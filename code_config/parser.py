"""
this function parser is used to analyze the configuration file to get an opt

New config layout (required):
file_config/
  base.json
  train.json / test.json / evaluation.json
  experiments.json                 # ONLY: {"work": "...", "work_list": [...]}
  experiments/
    <work>.json                    # one work per file, filename = work name
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from code_util.dataset.prepare import generate_paths_from_dict
from code_util.util import deep_update

config_root = "./file_config"


# =========================================================
# Basic IO
# =========================================================
def get_timestamp():
    return datetime.now().strftime("%y%m%d_%H%M%S")


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt", encoding="utf-8") as handle:
        json.dump(content, handle, indent=4, sort_keys=False, ensure_ascii=False)


def parse_json_file(json_path: str):
    """Parse json with // line comments support."""
    json_str = ""
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("//")[0] + "\n"
            json_str += line
    return json.loads(json_str)


# =========================================================
# Dict helpers
# =========================================================
class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    """convert to NoneDict, which returns None for missing key."""
    if isinstance(opt, dict):
        new_opt = {}
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    if isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


# =========================================================
# Experiments loading (NEW)
# =========================================================
def _load_experiment_index():
    """
    Load experiments.json which ONLY contains:
      - work: str
      - work_list: list[str] (optional)
    """
    index_path = os.path.join(config_root, "experiments.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"[Config] experiments.json not found: {index_path}")
    index_opt = parse_json_file(index_path)

    # hard guard: avoid old-format experiments.json silently passing
    forbidden_keys = [k for k in index_opt.keys() if k not in ("work", "work_list")]
    if forbidden_keys:
        raise ValueError(
            "[Config] experiments.json should only contain keys: 'work' and 'work_list'. "
            f"Found extra keys: {forbidden_keys}. "
            "Please move each work config into file_config/experiments/<work>.json"
        )
    return index_opt


def _resolve_work_name(cmdline_opt: dict, experiment_index: dict) -> str:
    """
    work name priority:
      1) command line --work
      2) experiments.json["work"]
    """
    work_name = experiment_index.get("work", None)
    if cmdline_opt.get("work") is not None:
        work_name = cmdline_opt["work"]
    if not work_name:
        raise ValueError("[Config] Missing work name: no --work and experiments.json has no 'work'.")
    return work_name


def _load_work_config(work_name: str, experiment_index: dict, strict_work_list: bool = True) -> dict:
    """
    Load file_config/experiments/{work_name}.json

    strict_work_list:
      - True: if work_list exists and work_name not in it -> raise
      - False: skip this validation
    """
    work_list = experiment_index.get("work_list", None)
    if strict_work_list and isinstance(work_list, list) and len(work_list) > 0:
        if work_name not in work_list:
            raise ValueError(f"[Config] work='{work_name}' not in work_list: {work_list}")

    work_path = os.path.join(config_root, "experiments", f"{work_name}.json")
    if not os.path.exists(work_path):
        raise FileNotFoundError(
            f"[Config] Work config not found: {work_path}\n"
            f"Please create: file_config/experiments/{work_name}.json"
        )
    return parse_json_file(work_path)


# =========================================================
# Main parse
# =========================================================
def parse(status="train", status_config=None, common_config=None, save=True, val=False):
    """
    Read config with the order:
      base.json
        -> {status}.json (or given status_config)
        -> experiments/<work>.json : general
        -> common_config (optional)
        -> experiments/<work>.json : {status}
        -> --config (optional)
        -> manual_manipulate
    """
    parser = init_parser(status)
    cmdline_opt = vars(parser.parse_args())

    # -----------------------------------------------------
    # base config
    # -----------------------------------------------------
    base_config_path = os.path.join(config_root, "base.json")
    base_opt = parse_json_file(base_config_path)

    # -----------------------------------------------------
    # status config: train/test/evaluation.json
    # -----------------------------------------------------
    if status_config is not None:
        status_opt = status_config
    else:
        status_path = os.path.join(config_root, f"{status}.json")
        status_opt = parse_json_file(status_path)

    # old behavior: merge status_opt["validation"] when val=True
    if val is True:
        if "validation" not in status_opt:
            raise KeyError("[Config] val=True but status config has no key 'validation'.")
        status_opt = deep_update(status_opt, status_opt["validation"])

    # -----------------------------------------------------
    # experiment config (NEW structure)
    # -----------------------------------------------------
    if status in ("train", "validation", "test"):
        exp_index = _load_experiment_index()
        work_name = _resolve_work_name(cmdline_opt, exp_index)
        experiment_opt = _load_work_config(work_name, exp_index, strict_work_list=True)

        # cmdline GPU override goes to experiment.general.model.gpu_ids (same as before)
        if cmdline_opt.get("gpu") is not None:
            experiment_opt.setdefault("general", {})
            experiment_opt["general"].setdefault("model", {})
            experiment_opt["general"]["model"]["gpu_ids"] = [cmdline_opt["gpu"]]

        experiment_general_opt = experiment_opt.get("general", {})
        experiment_status_opt = experiment_opt.get(status, {})
        common_opt_temp = deep_update(base_opt, experiment_general_opt)

    elif status == "evaluation":
        # evaluation does not require experiment config
        experiment_general_opt = {}
        experiment_status_opt = {}
        common_opt_temp = base_opt

        if cmdline_opt.get("recons") is True:
            status_opt.setdefault("reconstruction", {})
            status_opt["reconstruction"]["conduct_reconstruction"] = True
        if cmdline_opt.get("metrics") is True:
            status_opt.setdefault("metrics", {})
            status_opt["metrics"]["calculate_metrics"] = True
        if cmdline_opt.get("name") is not None:
            status_opt["name"] = cmdline_opt["name"]
        if cmdline_opt.get("gpu") is not None:
            status_opt["gpu"] = cmdline_opt["gpu"]

    else:
        raise ValueError(f"Invalid status: {status}")

    # -----------------------------------------------------
    # merge: base -> status -> exp.general
    # -----------------------------------------------------
    final_opt = deep_update(base_opt, status_opt)
    final_opt = deep_update(final_opt, experiment_general_opt)

    # -----------------------------------------------------
    # common config
    # -----------------------------------------------------
    if common_config is not None:
        common_opt = common_config
        common_opt = deep_update(common_opt_temp, common_opt)
        final_opt = deep_update(final_opt, common_opt)
    else:
        common_opt = common_opt_temp

    # merge exp.{status}
    final_opt = deep_update(final_opt, experiment_status_opt)

    # -----------------------------------------------------
    # cmdline / function config override
    # -----------------------------------------------------
    if cmdline_opt.get("config") is not None:
        config_path = cmdline_opt["config"]
        config_opt = parse_json_file(config_path)
        final_opt = deep_update(final_opt, config_opt)

    # -----------------------------------------------------
    # manual manipulation
    # -----------------------------------------------------
    final_opt = manual_manipulate(final_opt)

    # -----------------------------------------------------
    # derived paths
    # -----------------------------------------------------
    final_opt["work_relative_path"] = construct_work_relative_path(final_opt, status)
    final_opt["dataset"]["dataset_position"] = [
        os.path.join(final_opt["dataset"]["dataroot"], rel)
        for rel in generate_paths_from_dict(final_opt["dataset"]["info"])
    ]

    if status == "train":
        formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        final_opt["work_dir"] = os.path.join(
            final_opt["record"]["record_dir"],
            final_opt["work_relative_path"],
            formatted_time,
        )
    elif status in ("test", "evaluation"):
        final_opt["work_dir"] = os.path.join(
            final_opt["result"]["result_dir"],
            final_opt["work_relative_path"],
            final_opt["result"]["test_epoch"],
        )
    else:
        raise ValueError(f"Invalid status: {status}")

    os.makedirs(final_opt["work_dir"], exist_ok=True)

    # -----------------------------------------------------
    # save configuration
    # -----------------------------------------------------
    if save:
        config_path = os.path.join(final_opt["work_dir"], f"{status}_config.json")
        with open(config_path, "w", encoding="utf-8") as json_file:
            json.dump(final_opt, json_file, indent=4, ensure_ascii=False)

        common_config_path = os.path.join(final_opt["work_dir"], "common_config.json")
        with open(common_config_path, "w", encoding="utf-8") as json_file:
            json.dump(common_opt, json_file, indent=4, ensure_ascii=False)
    else:
        common_config_path = None

    return final_opt, common_opt


# =========================================================
# CLI parser
# =========================================================
def init_parser(status):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="configuration file")
    parser.add_argument("--gpu", type=int, default=None, help="gpu id")

    if status in ("train", "test", "validation"):
        parser.add_argument("--work", type=str, default=None, help="work name")
        parser.add_argument("--epoch", type=int, default=None, help="epoch number")
    elif status == "evaluation":
        parser.add_argument("--recons", action="store_true", help="conduct_reconstruction")
        parser.add_argument("--metrics", action="store_true", help="calculate_metrics")
        parser.add_argument("--name", type=str, default=None, help="experiment name")
    else:
        raise ValueError(f"Invalid status: {status}")

    return parser


# =========================================================
# Existing project functions (kept)
# =========================================================
def construct_work_relative_path(config, phase="train"):
    """
    Construct work relative path from dataset.info
    """
    dataset_info = config["dataset"]["info"]

    dataset_relative_path = ""
    for key in dataset_info:
        value = dataset_info[key]
        if isinstance(value, list):
            value = "_".join(str(v) for v in value)
        dataset_relative_path = os.path.join(dataset_relative_path, value)

    # keep original behavior
    dim = config["model"]["dim"]

    work_relative_path = os.path.join(dataset_relative_path, dim, config["name"])
    return work_relative_path


def manual_manipulate(opt):
    """manipulate the base opt here"""
    manual_opt = {
        "Task1": {  # MRI->CT
            "clip": {
                "use_clip_A": True,
                "use_clip_B": True,
                "clip_type_A": "99ptile",
                "clip_level_A": "patient",
                "clip_level_B": "population",
                "clip_range_B": [-1024, 2000],
            },
            "norm": {
                "use_norm_A": True,
                "use_norm_B": True,
                "norm_type_A": "99ptile",
                "norm_type_B": "minmax",
                "norm_level_A": "patient",
                "norm_level_B": "population",
                "minmax_norm_range_B": [-1024, 2000],
            },
        },
        "Task2": {  # CBCT->CT
            "clip": {
                "use_clip_A": True,
                "use_clip_B": True,
                "clip_level_A": "population",
                "clip_level_B": "population",
                "clip_range_A": [-1024, 2000],
                "clip_range_B": [-1024, 2000],
            },
            "norm": {
                "use_norm_A": True,
                "use_norm_B": True,
                "norm_type_A": "minmax",
                "norm_type_B": "minmax",
                "norm_level_A": "population",
                "norm_level_B": "population",
                "minmax_norm_range_A": [-1024, 2000],
                "minmax_norm_range_B": [-1024, 2000],
            },
        },
        "Task3": {  # CT->MRI
            "clip": {
                "use_clip_A": True,
                "use_clip_B": False,
                "clip_level_A": "population",
                "clip_range_A": [-1024, 2000],
            },
            "norm": {
                "use_norm_A": True,
                "use_norm_B": False,
                "norm_type_A": "minmax",
                "norm_level_A": "population",
                "minmax_norm_range_A": [-1024, 2000],
            },
        },
    }

    task = opt["dataset"]["info"]["task"]
    if task == "Task1":
        opt["preprocess"]["clip"] = manual_opt["Task1"]["clip"]
        opt["preprocess"]["norm"] = manual_opt["Task1"]["norm"]
    elif task == "Task2":
        opt["preprocess"]["clip"] = manual_opt["Task2"]["clip"]
        opt["preprocess"]["norm"] = manual_opt["Task2"]["norm"]
    elif task == "Task3":
        opt["preprocess"]["clip"] = manual_opt["Task3"]["clip"]
        opt["preprocess"]["norm"] = manual_opt["Task3"]["norm"]
        if "segmentation" in opt:
            opt["segmentation"]["task"] = "total"
            opt["segmentation"]["modality"] = "mr_processed"
        if "metrics" in opt:
            opt["metrics"]["image_similarity"]["dynamic_range"] = [-1, 1]
    else:
        raise ValueError(f"Invalid task: {task}")

    dataset_path = os.path.join(opt["dataset"]["dataroot"], opt["dataset"]["info"]["name"])
    if not os.path.exists(dataset_path):
        opt["dataset"]["dataroot"] = opt["dataset"]["dataroot_alternate"]

    return opt