from lm_eval.utils import eval_logger

LOGGER = eval_logger

class TaskInputStats:
    def __init__(self):
        self._stats: list = list()

    def generate_report(self):
        stats = self._stats
        items_inp = 0
        items_batch = 0

        for item in stats:
            if not isinstance(item, list):
                item = [item]
            for i in item:
                if "context_enc" in i:
                    items_inp += 1
                elif "batched_inps_1" in i:
                    items_batch += 1
                else:
                    raise ValueError("Unexpected data package in stats")

        if items_inp != items_batch:
            LOGGER.warning(f"Uneven number of stats packets items_inp={items_inp} vs items_batch={items_batch}")

        context_enc_size = dict()
        continuation_enc_size = dict()


        inplen = None
        inp_padding_len = None

        total_prompts = 0
        padded_prompts = 0
        truncated_prompts = 0

        for item in stats:
            for i in item:
                if "context_enc" in i:
                    if i["context_enc"] in context_enc_size:
                        context_enc_size[i["context_enc"]] += 1
                    else:
                        context_enc_size[i["context_enc"]] = 1

                    if i["continuation_enc"] in continuation_enc_size:
                        continuation_enc_size[i["continuation_enc"]] += 1
                    else:
                        continuation_enc_size[i["continuation_enc"]] = 1

                    if inplen is None:
                        inplen = i["inplen"]

                    if inp_padding_len is None:
                        inp_padding_len = i["inp_padding_len"]

        for item in stats:
            for i in item:
                if "context_enc" in i:
                    total_prompts += 1
                    if inplen < i["context_enc"]:
                        truncated_prompts += 1
                    if inplen > i["context_enc"]:
                        padded_prompts += 1

        LOGGER.info("----------------------------------------------------------------")
        LOGGER.info("Requests (context) counts by token size. Length(tokens) = count")
        for key, value in context_enc_size.items():
            LOGGER.info(f"{key}={value}")

        LOGGER.info("Responses (continuation) counts by token size. Length(tokens) = count")
        for key, value in continuation_enc_size.items():
            LOGGER.info(f"{key}={value}")

        LOGGER.info("----------------------------------------------------------------")
        LOGGER.info(f"Total prompts: {total_prompts}")
        LOGGER.info(f"Padded prompts: {padded_prompts}")
        LOGGER.info(f"Truncated prompts: {truncated_prompts}")
        LOGGER.info("----------------------------------------------------------------")

    def reset(self):
        self._stats = list()

    def log(self, data: dict):
        self._stats.append(data)
