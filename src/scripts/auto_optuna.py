"""
让任意科研python运行、使用配置文件指定参数的科研实验自动变成可以被Optuna调参的实验。

This script supports two modes:
1. `tune`: For efficient hyperparameter exploration using CMA-ES Sampler and Hyperband Pruner.
2. `ablation`: For reproducible ablation studies using Grid Search and no pruning.
"""

# %%
import argparse
import json
import operator
import shutil
import subprocess
import sys
import uuid
from functools import reduce
from pathlib import Path

import optuna
import yaml
from typing import Any, Dict, Optional


from skinfra.experiment import (
    load_config,
    save_config,
    load_overlaying_config,
    iterate_path_hierarchy,
)

# %%
from dotenv import load_dotenv

load_dotenv()
import os


# %%
def set_nested_key(d: Dict, key_path: str, value):
    """Sets a value in a nested dictionary using a dot-separated path."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


# %%
def objective(
    trial: optuna.Trial, base_experiment_path: str, optuna_config: Dict
) -> float:
    """
    The Optuna objective function.
    """
    # 1. Create a unique directory for this trial that conforms to the expected path structure.
    base_path = Path(base_experiment_path)
    trial_naming_hyperparameters = optuna_config.get("trial_naming_hyperparameters", [])
    if trial_naming_hyperparameters:
        # name_parts = [f"{base_path.name}"]
        name_parts = []
        for param in trial_naming_hyperparameters:
            if param in trial.params:
                name_parts.append(f"{param}={trial.params[param]}")
        name_parts.append(f"trial-{trial.number}-{uuid.uuid4().hex[:8]}")
        trial_name = "/".join(name_parts)
    else:
        trial_name = f"{base_path.name}-trial-{trial.number}-{uuid.uuid4().hex[:8]}"
    # trial_path = base_path.parent.parent / "optuna_sweeps" / trial_name
    trial_path = base_path / "auto_optuna" / trial_name
    trial_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Starting Trial {trial.number} ---\nPath: {trial_path}")

    # Copy the template directory to the new trial path.
    shutil.copytree(
        base_experiment_path,
        trial_path,
        ignore=shutil.ignore_patterns(
            "optuna_sweep", "results.json", "*.log", "__pycache__"
        ),
    )

    # 2. Suggest hyperparameters and modify config files
    for config_filename, params in optuna_config.get("files", {}).items():
        config_path = trial_path / config_filename
        if not config_path.exists():
            print(
                f"[Trial {trial.number}] Warning: Config file '{config_filename}' not found. Skipping."
            )
            continue

        # filetype = config_filename.split(".")[-1]
        # TODO 用户指定了一些参数，但是是overlay继承的，不是本身就有的，我们是否应该支持调参？
        # config_data: Dict = load_config(config_path)
        config_data: Dict = load_overlaying_config(config_path)

        for key_path, suggest_config in params.items():
            suggest_type = suggest_config["type"]
            suggest_args = suggest_config["args"]

            suggest_method = getattr(trial, f"suggest_{suggest_type}")
            value = suggest_method(**suggest_args)

            set_nested_key(config_data, key_path, value)
            print(f"[Trial {trial.number}] Set {key_path} = {value}")
        # Save the modified config back to file
        # 为了最终让人清晰看到所有可复现参数，把overlay的参数也写进去，防止因为optuna选择的路径不同导致overlay到不同的东西。
        save_config(config_data, config_path)

    # 3. Launch run.py as a subprocess and capture its output
    # command list避免shell injection问题，但是我们信任输入者，应该尽可能支持的运行方式差不多，所以使用SHELL更好，特别是有环境变量的情况下。
    study_name = optuna_config.get("study_name", "auto_optuna_study")
    os.environ["SWANLAB_PROJECT_NAME"] = study_name
    os.environ["SWANLAB_EXPERIMENT_NAME"] = f"trial_{trial.number}"

    command_template: str = optuna_config.get(
        "command_template", "python run.py {experiment_path}"
    )
    command: str = command_template.format(experiment_path=str(trial_path))

    print(
        f"[Trial {trial.number}] Executing command: \n{'-' * 40}\n{command}\n{'-' * 40}\n"
    )

    log_path = trial_path / f"trial_{trial.number}.log"
    process = subprocess.Popen(
        command,
        shell=True,
        # cwd=trial_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    # 4. Monitor the output in real-time
    final_value = None
    intermediate_metric_key = optuna_config["intermediate_metric"]
    final_metric_key = optuna_config["final_metric"]
    print(
        f"[Trial {trial.number}] Monitoring output for intermediate metric '{intermediate_metric_key}' and final metric '{final_metric_key}'"
    )

    # Open log file to save the output
    with open(log_path, "w") as log_file:
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            log_file.write(line)

            try:
                log_data = json.loads(line)

                if intermediate_metric_key in log_data:
                    step = log_data.get("step")
                    metric_value = log_data[intermediate_metric_key]
                    trial.report(metric_value, step)
                    print(
                        f"[Trial {trial.number}] Reported intermediate metric: Step {step}, {intermediate_metric_key}: {metric_value}"
                    )
                    if trial.should_prune():
                        print(f"[Trial {trial.number}] Pruning trial.")
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise optuna.TrialPruned()
                if final_metric_key in log_data:
                    final_value = log_data[final_metric_key]
                    print(f"[Trial {trial.number}] Found final metric: {final_value}")
            except (json.JSONDecodeError, TypeError):
                continue

    process.wait()
    print(f"[Trial {trial.number}] Finished with exit code: {process.returncode}")

    if final_value is None:
        print(
            f"[Trial {trial.number}] Could not find final metric '{final_metric_key}' in log. Assuming failure."
        )
        raise RuntimeError("Final metric could not be determined.")
    return final_value


# %%
from fastcore.script import call_parse
# https://fastcore.fast.ai/script.html

# 如果没有call parse，那就是直接运行这个函数。


@call_parse
def auto_optuna(
    path_experiment: str = ".",  # Path to the base experiment directory containing configs (train_params.json, optuna_config.yaml, etc.)
    mode: str = "ablation",  # Set the operation mode: 'tune' for hyperparameter search, 'ablation' for grid search.
):
    """Auto Optuna Hyperparameter Optimization Script."""
    # --- Load Optuna Configuration ---
    optuna_config = load_overlaying_config(
        path_experiment,
        os.getenv("OPTUNA_CONFIG_FILE", "optuna_config.yaml"),
        verbose=True,
    )
    if optuna_config is None:
        raise ValueError(
            f"Optuna configuration file not found in experiment path: {path_experiment} with mode {mode}"
        )
    print("Loaded Optuna configuration:")
    print(yaml.dump(optuna_config, indent=2, sort_keys=False))

    # --- Setup Optuna Study ---
    path_experiment_name = Path(path_experiment).name
    study_name = optuna_config.get("study_name", "auto_optuna_study")
    study_name = f"{study_name}_{path_experiment_name}_{mode}"  # TODO 是否需要配置
    optuna_config.study_name = study_name

    n_trials = optuna_config.get("n_trials", 20)
    direction = optuna_config.get("direction", "maximize")

    storage_path = Path(os.getenv("OPTUNA_STORAGE_DIR", "."))
    storage_path.mkdir(exist_ok=True)
    storage_name = f"sqlite:///{storage_path / 'auto_optuna_studies.db'}"

    # --- Configure Sampler and Pruner based on mode ---
    if mode == "ablation":
        print(
            "\n[Mode: Ablation] Using Grid Search Sampler and disabling pruning for reproducibility."
        )
        search_space = {}
        for _, params in optuna_config.get("files", {}).items():
            for _, suggest_config in params.items():
                if suggest_config["type"] != "categorical":
                    raise ValueError(
                        f"In 'ablation' mode, all hyperparameters must be of type 'categorical'."
                    )
                param_name = suggest_config["args"]["name"]
                choices = suggest_config["args"]["choices"]
                search_space[param_name] = choices

        sampler = optuna.samplers.GridSampler(search_space)
        pruner = optuna.pruners.NopPruner()

        grid_size = (
            reduce(operator.mul, (len(v) for v in search_space.values()), 1)
            if search_space
            else 1
        )
        if n_trials != grid_size:
            print(
                f"Warning: In Grid Search mode, n_trials ({n_trials}) does not match the grid size ({grid_size}). Adjusting n_trials to {grid_size}."
            )
            n_trials = grid_size

    elif mode == "tune":
        print(
            "\n[Mode: Tune] Using CMA-ES Sampler and Hyperband Pruner for efficient tuning."
        )
        sampler = optuna.samplers.CmaEsSampler(consider_pruned_trials=True)
        pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction=direction,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    # --- Run Optimization ---
    try:
        study.optimize(
            lambda trial: objective(trial, path_experiment, optuna_config),
            n_trials=n_trials,
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during the optimization: {e}")
        print("The study will now conclude. You may be able to resume it later.")

    # --- Print Results ---
    print("\n--- Sweep Complete ---")
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    failed_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.FAIL]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")
    print(f"  Complete trials: {len(complete_trials)}")

    if study.best_trial:
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No successful trials were completed.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path_experiment",
        type=str,
        help="Path to the base experiment directory containing configs (train_params.json, optuna_config.yaml, etc.)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ablation",
        choices=["tune", "ablation"],
        help="Set the operation mode: 'tune' for hyperparameter search, 'ablation' for grid search.",
    )
    args = parser.parse_args()
    path_experiment = args.path_experiment
    mode = args.mode
    return auto_optuna(path_experiment, mode)


if __name__ == "__main__":
    main()
