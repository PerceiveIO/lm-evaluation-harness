from __future__ import annotations

import json
import os
from pathlib import Path

import huggingface_hub
import lm_eval.evaluator
import wandb
from copy import deepcopy
from lightning import LightningModule
from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lm_eval import utils
from lm_eval.tasks import TaskManager

import lm_eval
import lm_eval.api
import lm_eval.api.metrics
import lm_eval.models
import lm_eval.tasks

from lm_eval.evaluator import evaluate

from lm_eval.utils import positional_deprecated
from lm_eval.evaluator_utils import run_task_tests
from lm_eval.models.slalom import SlalomHFLM
from lm_eval.utils import eval_logger

import numpy as np
import tinyBenchmarks as tb

LOGGER = eval_logger


def get_task_names(tasks_arg: str) -> list[str]:
    """Retrieve a list of task names based on the provided argument.

    Args:
        tasks_arg: A string representing either a directory path containing YAML files,
                        a comma-separated list of task names, or None to use all available tasks.
    Returns:
        list of task names obtained from the specified argument.
    """
    task_manager = TaskManager()

    if tasks_arg is None:
        return task_manager.all_tasks

    task_path = Path(tasks_arg)
    if task_path.is_dir():
        return [utils.load_yaml_config(str(yaml_file)) for yaml_file in task_path.glob("*.yaml")]

    tasks_list = tasks_arg.split(",")
    task_names = task_manager.match_tasks(tasks_list)
    task_names += [utils.load_yaml_config(task) for task in tasks_list if Path(task).is_file()]
    task_missing = [task for task in tasks_list if task not in task_names and "*" not in task]

    if task_missing:
        missing = ", ".join(task_missing)
        raise ValueError(f"Tasks {missing} were not found. Try `lm-eval --tasks list` for list of available tasks.")

    return task_names


def run_evaluation(
    litmodule: LightningModule,
    tokenizer_name: str,
    tasks: str,
    num_fewshot: int,
    batch_size: int,
    max_batch_size: int,
    model_args: str,
    limit: int | float,
    check_integrity: bool,
    write_out: bool,
    log_samples: bool,
) -> dict[str, str]:
    """Run evaluation on specified task.

    Args:
        litmodule : Lightning LMM model to evaluate on.
        tokenizer_name: The name of the tokenizer to use
        tasks: Name of the task to evaluate
        num_fewshot : Number of examples in the few-shot context.
        batch_size: Batch size for model
        max_batch_size: Maximal batch size to try with automatic batch size detection
        model_args: String arguments for each model class, see LM.create_from_arg_string.
        limit: Limit the number of examples per task (only use this for testing),
            If <1, limit is a percentage of the total number of examples.
        check_integrity: Whether to run the relevant part of the test suite for the tasks
        write_out: If True, write out an example document and model input for checking task integrity
        log_samples: If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    Returns:
         A dictionary containing evaluation results.
    """
    if limit:
        LOGGER.warning(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    task_names = get_task_names(tasks_arg=tasks)

    LOGGER.info('Running task(s) "%s" ...', ",".join(task_names))

    results = simple_evaluate(
        litmodule=litmodule,
        tokenizer_name=tokenizer_name,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        model_args=model_args,
        limit=limit,
        check_integrity=check_integrity,
        write_out=write_out,
        log_samples=log_samples,
    )

    # filter out verbose "configs", "versions", etc. for MMLU task
    result_to_log = {k: v for k, v in results.items() if k in ("results", "groups", "config")}
    LOGGER.info('Raw result for "%s":\n%s', ",".join(task_names), json.dumps(result_to_log, indent=2))

    return results


def make_summary(results: dict) -> str:
    """Generate a summary string from the evaluation results.

    Args:
        results: A dictionary containing evaluation results for different tasks.

    Returns:
        str: A formatted summary string of the evaluation results.
    """
    summary = [
        "",
        "==============================",
        "Finished Evaluation. Results:",
        "==============================",
    ]

    for task_name in results:
        result = results[task_name]["results"]
        if task_name == "mmlu" and "mmlu" in result:
            # Only include the MMLU result, since this is very verbose otherwise
            result = {"mmlu": result["mmlu"]}
        summary.append(json.dumps(result, indent=2))

    return "\n".join(summary)


def make_wandb_table(results, column: str = "results"):
    """Generate wandb table of results."""
    result_dict = deepcopy(results)
    if column == "results":
        column_name = "Tasks"
    elif column == "groups":
        column_name = "Groups"

    all_headers = [
        "Level",
        column_name,
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "Value",
        "Stderr",
    ]

    values = []
    full_name = []

    for k, dic in result_dict[column].items():
        version = result_dict["versions"].get(k, "N/A")
        n = str(result_dict["n-shot"][k])

        if "alias" in dic:
            k = dic.pop("alias")

        level = k.find("-")
        if level < 0:
            level = 0
        k = k.replace("-", "").strip()
        if len(full_name) <= level:
            full_name.append(k)
        else:
            full_name[level] = k
            del full_name[level + 1 :]
        k = ".".join(full_name)

        for (mf), v in dic.items():
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" + "," + f in dic:
                se = dic[m + "_stderr" + "," + f]
                values.append([level, k, str(version), f, n, m, v, se])
            else:
                values.append([level, k, str(version), f, n, m, v, ""])
            k = ""
            version = ""

    return wandb.Table(columns=all_headers, data=values)


# Not currently used, but kept as an alternative parsing implementation
def wandb_table_from_markdown_table_str(data: str) -> wandb.Table:
    """Convert MarkDown table generated by lm_eval.utils.make_table() to W&B Table
    Args:
        data: String containing markdown formatted table. Example:
        |                 Tasks                 |Version|Filter|n-shot|Metric|Value |   |Stderr|
        |---------------------------------------|-------|------|-----:|------|-----:|---|-----:|
        |mmlu                                   |N/A    |none  |     0|acc   |0.2295|±  |0.0400|
        | - humanities                          |N/A    |none  |     5|acc   |0.2421|±  |0.0312|
        |  - formal_logic                       |      0|none  |     5|acc   |0.2857|±  |0.0404|
        |  - high_school_european_history       |      0|none  |     5|acc   |0.2182|±  |0.0323|
        |  - high_school_us_history             |      0|none  |     5|acc   |0.2500|±  |0.0304|
        |  - high_school_world_history          |      0|none  |     5|acc   |0.2700|±  |0.0289|
        |  - international_law                  |      0|none  |     5|acc   |0.2397|±  |0.0390|
        ...
    Returns:
        wandb.Table: W&B Table instance
    """
    lines = data.split("\n")

    columns = ["Level"] + [x.strip() for x in lines[0].split("|")][1:-1]
    rows = []
    full_name = []
    for line in lines[2:]:
        row = []
        for idx, val in enumerate(line.split("|")[1:-1]):
            if idx == 0:
                level = val.find("-")
                if level < 0:
                    level = 0
                row.append(level)
                name = val.replace("-", "").strip()
                if len(full_name) <= level:
                    full_name.append(name)
                else:
                    full_name[level] = name
                    del full_name[level + 1 :]
                row.append(".".join(full_name))
            else:
                row.append(val.strip())
        if len(row) == len(columns):
            rows.append(row)
    _table = wandb.Table(columns=columns, data=rows)
    return _table

def slalom_evaluate(cfg: dict, litmodule: LightningModule, logger: LightningLogger) -> dict:
    """Run evaluation on the specified tasks.

    Args:
        cfg : Configuration object containing evaluation parameters.
        litmodule : The LightningModule to be evaluated.
    """
    if "llama" in cfg.tokenizer_name:
        # Login to Hugging Face Hub with the specified token for access to Llama2 HF Tokenizer.
        huggingface_hub.login(os.environ["HUGGINGFACE_TOKEN"])

    tasks = cfg.tasks.split(",")
    few_shots = str(cfg.num_fewshot).split(",")

    if cfg.log_samples is False and any(s.startswith("tiny_") for s in tasks):
        assert RuntimeError("Set log_samples to True for TinyBenchmark evaluation.")

    results = {}
    for task, num_shot in zip(tasks, few_shots, strict=True):
        results[task] = run_evaluation(
            litmodule=litmodule,
            tokenizer_name=cfg.tokenizer_name,
            tasks=task,
            num_fewshot=int(num_shot),
            batch_size=cfg.batch_size,
            max_batch_size=cfg.max_batch_size,
            model_args=cfg.model_args,
            limit=cfg.limit,
            check_integrity=cfg.check_integrity,
            write_out=cfg.write_out,
            log_samples=cfg.log_samples,
        )
        # copy dict because make_table modifies it
        results_table = lm_eval.utils.make_table(deepcopy(results[task]))
        LOGGER.info('Results table for "%s":\n%s', task, results_table)

        metrics = {}
        if task in results[task]["results"]:
            for k, v in results[task]["results"][task].items():
                if k == "alias":
                    continue
                metric_key = f"{task}.{k.replace(',none', '')}"
                metrics[metric_key] = v

        if isinstance(logger, WandbLogger):
            logger.experiment.log(
                {f"Evaluation Results for {task}": make_wandb_table(results[task]), **metrics}
            )

    summary = make_summary(results)
    LOGGER.info(summary)

    return results

@positional_deprecated
def simple_evaluate(
    litmodule: LightningModule,
    tokenizer_name: str,
    tasks: str,
    num_fewshot: int,
    batch_size: int = 1,
    max_batch_size: int = None,
    model_args: str = None,
    limit: int | float = None,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = False,
) -> dict[str, str]:
    """Instantiate and evaluate a model on a list of tasks.

    Args:
        litmodule : Lightning LMM model to evaluate on.
        tokenizer_name: The name of the tokenizer to use
        tasks: Name of the task to evaluate
        num_fewshot : Number of examples in the few-shot context.
        batch_size: Batch size for model
        max_batch_size: Maximal batch size to try with automatic batch size detection
        model_args: String arguments for each model class, see LM.create_from_arg_string.
        limit: Limit the number of examples per task (only use this for testing),
            If <1, limit is a percentage of the total number of examples.
        check_integrity: Whether to run the relevant part of the test suite for the tasks
        write_out: If True, write out an example document and model input for checking task integrity
        log_samples: If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    Returns:
         A dictionary containing evaluation results.
    """

    if tasks is None:
        tasks = []

    assert tasks != [], "No tasks specified"

    if model_args is None:
        model_args = ""
    lm = SlalomHFLM.create_from_arg_string(
        model_args,
        {
            "batch_size": batch_size,
            "max_batch_size": max_batch_size,
            "litmodule": litmodule,
            "tokenizer": tokenizer_name,
        },
    )

    task_dict = lm_eval.tasks.get_task_dict(tasks)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if type(task_obj) is tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue

        config = task_obj._config

        if num_fewshot is not None:
            if config["num_fewshot"] != 0:
                default_num_fewshot = config["num_fewshot"]
                LOGGER.info(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj._config["num_fewshot"] = num_fewshot

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(lm=lm, task_dict=task_dict, limit=limit, write_out=write_out, log_samples=log_samples)

    # add info about the model and few shot config
    results["config"] = {
        "model": model_args if isinstance(model_args, str) else model_args.model.config._name_or_path,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "batch_sizes": list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else [],
        "limit": limit,
    }

    return results
