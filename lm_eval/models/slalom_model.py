import torch
from torch import nn
from transformers import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


class OutputSlalomClass:
    def __init__(self, logits):
        self.logits = list(logits)


class SlalomModel(nn.Module, GenerationMixin):
    def __init__(self, litmodule, config, device):
        super().__init__()
        self.litmodule = litmodule
        self.config = config
        self.main_input_name = 'input_ids'
        self.generation_config = GenerationConfig(
            do_sample=False,
            early_stopping=True,
            num_return_sequences=1
        )

        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask=None, token_type_ids=None, past_key_values=None,
                use_cache=False, **args):
        if use_cache:
            logits, layer_past = self.litmodule(input_ids, past_key_values=past_key_values, use_cache=use_cache)
        else:
            logits = self.litmodule(input_ids)
            layer_past = None

        return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=layer_past,
                hidden_states=None,
                attentions=None,
            )



    def can_generate(self) -> bool:
        """
            Returns whether this model can generate sequences with `.generate()`.

            Returns:
                `bool`: Whether this model can generate sequences with `.generate()`.
            """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
        if "GenerationMixin" in str(self.prepare_inputs_for_generation.__func__):
            return False
        return True

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs
