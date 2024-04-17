from __future__ import annotations

import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
import transformers
from lightning import LightningModule
from lm_eval import utils
from lm_eval.models.utils import Grouper, chunks
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, pad_and_concat, generate
from tqdm import tqdm
from transformers import GPT2Config

from lm_eval.utils import eval_logger
from lm_eval.task_input_stats import TaskInputStats

LOGGER = eval_logger


def _get_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible.

    Does not use an instantiated HF AutoConfig
    """
    _torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else dtype
    return _torch_dtype


def get_config(litmodule: LightningModule):
    if hasattr(litmodule.model, "model_config"):
        config = GPT2Config(
            n_embd=litmodule.model.model_config.n_embd,
            n_layer=litmodule.model.model_config.n_layer,
            n_head=litmodule.model.model_config.n_head,
            vocab_size=litmodule.model.model_config.vocab_size,
            n_positions=litmodule.model.model_config.block_size,
        )
    elif hasattr(litmodule.model, "gpt"):
        config = GPT2Config(
            n_embd=litmodule.model.gpt.model_config.n_embd,
            n_layer=litmodule.model.gpt.model_config.n_layer,
            n_head=litmodule.model.gpt.model_config.n_head,
            vocab_size=litmodule.model.gpt.model_config.vocab_size,
            n_positions=litmodule.model.gpt.model_config.block_size,
        )
    elif hasattr(litmodule.model, "vocab_size"):
        config = GPT2Config(
            n_embd=litmodule.model.embed_dim,
            n_layer=litmodule.model.num_layers,
            n_head=litmodule.model.num_heads,
            vocab_size=litmodule.model.vocab_size,
            n_positions=litmodule.model.block_size,
        )
    else:
        raise NotImplementedError("The configuration for the model being evaluated is not recognized.")

    return config

@register_model("slalom")
class SlalomHFLM(LM):
    """Base class for models from the HuggingFace transformers library."""

    _DEFAULT_MAX_LENGTH = 2048
    AUTO_MODEL_CLASS = None

    def __init__(
        self,
        device: str = "cuda",
        tokenizer: str = None,
        batch_size: int | str = 1,
        max_length: int = None,
        litmodule: LightningModule = None,
        max_batch_size: int = 64,
    ):
        """
        Args:
            device: Device to use for the model. Default: ``"cuda"``.
            tokenizer: Name of the tokenizer to use. Default: ``None``.
            batch_size: Batch size to use. Default: ``1``.
            max_length: Maximum length of the input sequence. Default: ``None``.
            litmodule: LightningModule to use. Default: ``None``.
            max_batch_size: Maximum batch size to use. Default: ``64``.
        """
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(batch_size, int | str)

        device_list = set(["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())])
        if device and device in device_list:
            if device == "cuda":
                device = f"cuda:{litmodule.device.index}"
            self._device = torch.device(device)
            LOGGER.info(f"Using device '{device}'")
        else:
            LOGGER.info("Device not specified")
            LOGGER.info(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.litmodule = litmodule

        self.config = get_config(self.litmodule)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        self._max_length = max_length
        # setup for automatic batch size detection
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size


        self.task_input_stats = TaskInputStats()

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: list[int]):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.litmodule(inps)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        # whole_enc = self.tok_encode(context + continuation)
        # context_enc = self.tok_encode(context, add_special_tokens=False)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _model_generate(self, context, max_length):
        return generate(self.litmodule.model, context, block_size=self.config.n_positions, max_new_tokens=max_length)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(continuation)
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=True)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        loglikelihoods = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            LOGGER.info("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            LOGGER.info(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for (string,) in tqdm([req.args for req in requests], disable=(self.rank != 0)):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = self.accelerator.gather(mytensor).detach().cpu().numpy().tolist()

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> tuple[list[int], list[int]]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        self.tokenizer.pad_token = self.tokenizer.eos_token
        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )

        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = defaultdict(list)
        re_ords = {}

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer([req.args for req in reqs], _collate)

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        if self.batch_size == "auto":
            # using rolling window with maximum context
            LOGGER.info("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            LOGGER.info(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        for key, re_ord in re_ords.items():
            _chunks = chunks(
                re_ord.get_reordered(),
                n=(
                    self.batch_size
                    if self.batch_size != "auto"
                    else adaptive_batch_size
                    if adaptive_batch_size is not None
                    else 0
                ),
                fn=self._batch_scheduler if self.batch_size == "auto" and not adaptive_batch_size else None,
            )
            for chunk in _chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]
                # unpack our keyword arguments.
                until = None
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [kwargs]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                            )
                else:
                    raise ValueError(f"Expected `kwargs` to be of type `dict` but got {kwargs}")
                if not until:
                    until = [self.tok_decode(self.eot_token_id)]
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks

                # set the max length in tokens of inputs ("context_enc")
                max_ctx_len = self.max_length - max_gen_toks

                # encode, pad, and truncate contexts for this batch
                context_enc, attn_masks = self.tok_batch_encode(
                    contexts,
                    left_truncate_len=max_ctx_len,
                )
                context_enc = context_enc.to(self.device)

                if "max_length" not in kwargs:
                    kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

                # perform batched generation
                cont = self._model_generate(context=context_enc, max_length=max_gen_toks)

                cont_toks_list = cont.tolist()
                for cont_toks, context in zip(cont_toks_list, contexts):
                    # discard context + left-padding toks if using causal decoder-only LM
                    if transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS:
                        cont_toks = cont_toks[context_enc.shape[1] :]

                    s = self.tok_decode(cont_toks)

                    # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                    for term in until:
                        if len(term) > 0:
                            # ignore '' separator,
                            # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                            s = s.split(term)[0]

                    res[key].append(s)

                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                    pbar.update(1)
            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)

    def _select_cont_toks(self, logits: list[int], contlen: int = None, inplen: int = None):
        assert contlen and inplen, "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]
        return logits

    def _batch_scheduler(self, pos: int, n_reordered_requests: list[int]):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (self.batch_sizes[sched - 1] == self.max_batch_size):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        LOGGER.info(f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size")
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        LOGGER.info(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> list[tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = self.batch_size if self.batch_size != "auto" else override_bs if override_bs is not None else 0
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
            else None
        )

        _chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)

        chunk_idx = 0
        print_freq = int(len(requests) / 20) # print every 5%
        if print_freq == 0:
            print_freq = 1
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)))
        for chunk in _chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            _stats = []
            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape

                padding_len_inp = max(padding_len_inp, inplen) if padding_len_inp is not None else inplen

                _stats.append({
                    "context_enc": len(context_enc),
                    "continuation_enc": len(continuation_enc),
                    "inp_padding_len": padding_len_inp,
                    "inplen": inplen
                })

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            self.task_input_stats.log(_stats)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}

            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")  # [batch, padding_len_inp]

            self.task_input_stats.log(
                {
                    "batched_inps_0": batched_inps.shape[0],
                    "batched_inps_1": batched_inps.shape[1]
                }
            )

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (cache_key, _, _), logits, inplen, cont_toks in zip(chunk, multi_logits, inplens, cont_toks_list):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                res.append(answer)

                self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)

                if disable_tqdm:
                    chunk_idx += 1
                    if chunk_idx % print_freq == 0:
                        LOGGER.info("Processed %d out of %d chunks (%.2f%%)...", chunk_idx, len(requests), (chunk_idx + 1) * 100 / len(requests))
        pbar.close()

        return re_ord.get_original(res)

