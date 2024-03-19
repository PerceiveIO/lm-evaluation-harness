import io
import numpy as np

from collections import Counter

from lm_eval.utils import eval_logger

LOGGER = eval_logger


class TaskInputStats:
    def __init__(self):
        self._stats: list = list()

    def data_as_histogram(self, a, bins=40, width=120) -> str:
        h, b = np.histogram(a, bins)
        output = io.StringIO()

        for i in range (0, bins):
            print('{:12.0f}  | {:{width}s} {}'.format(
                b[i],
                '#'*int(width*h[i]/np.amax(h)),
                h[i],
                width=width), file=output)
        print('{:12.0f}  |'.format(b[bins]), file=output)

        contents = output.getvalue()
        output.close()
        return contents

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

        context_enc_ = list()
        continuation_enc_ = list()

        inplen = None
        total_prompts = 0
        padded_prompts = 0
        truncated_prompts = 0

        for item in stats:
            for i in item:
                if "context_enc" in i:
                    context_enc_.append(i["context_enc"])
                    continuation_enc_.append(i["continuation_enc"])

                    if inplen is None:
                        inplen = i["inplen"]

        for item in stats:
            for i in item:
                if "context_enc" in i:
                    total_prompts += 1
                    if inplen < i["context_enc"]:
                        truncated_prompts += 1
                    if inplen > i["context_enc"]:
                        padded_prompts += 1

        LOGGER.info("----------------------------------------------------------------")
        LOGGER.info(f"Requests (context) counts by token size. Length(tokens) | count\n{self.data_as_histogram(context_enc_, bins=min(30, 1 + len(Counter(context_enc_).keys())))}")
        LOGGER.info(f"Responses (continuation) counts by token size. Length(tokens) | count\n{self.data_as_histogram(continuation_enc_, bins=min(30, 1 + len(Counter(continuation_enc_).keys())))}")
        LOGGER.info("----------------------------------------------------------------")
        LOGGER.info(f"Total prompts: {total_prompts}")
        LOGGER.info(f"Padded prompts: {padded_prompts}")
        LOGGER.info(f"Truncated prompts: {truncated_prompts}")
        LOGGER.info("----------------------------------------------------------------")

    def reset(self):
        self._stats = list()

    def log(self, data: dict):
        self._stats.append(data)
