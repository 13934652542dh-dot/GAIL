"""读取正式实验输入，并构造环境、MIP 和强化学习共用的数据视图。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
FORMAL_INPUT_DIR = PROJECT_DIR / "data" / "正式实验输入"
WT_OUTPUT_FILE = FORMAL_INPUT_DIR / "WT的出力数据.pkl"
SYSTEM_OUTPUT_FILE = FORMAL_INPUT_DIR / "系统电气参数.pkl"
DEFAULT_PKL_PATH = WT_OUTPUT_FILE


def load_dictionary(input_file: str | Path) -> dict:
    """读取一个 pkl 字典。

    输入：pkl 文件路径。
    输出：文件中保存的原始字典。
    """

    with Path(input_file).open("rb") as file:
        return pickle.load(file)


def build_window_view(wt_data: dict) -> dict:
    """把 train 和 test 转换为程序统一使用的窗口数据。

    输入：顶层包含 train 和 test 的 WT 出力字典。
    输出：依次排列的窗口编号、数据集划分、风机出力、B 和 E。

    按 train 在前、test 在后的顺序拼接数组，并在内存中生成连续窗口编号，
    不修改原始 pkl 文件。
    """

    train = wt_data["train"]
    test = wt_data["test"]
    train_count = len(train["WT_power_mw"])
    test_count = len(test["WT_power_mw"])
    return {
        "window_ids": np.arange(train_count + test_count, dtype=np.int64),
        "dataset_split": np.concatenate([
            np.full(train_count, "train", dtype="<U5"),
            np.full(test_count, "test", dtype="<U4"),
        ]),
        "current_power": np.concatenate([train["WT_power_mw"], test["WT_power_mw"]]),
        "B": np.concatenate([train["B"], test["B"]]),
        "E": np.concatenate([train["E"], test["E"]]),
    }


def load_experiment_data(pkl_path: str | Path = DEFAULT_PKL_PATH) -> dict:
    """读取正式实验所需的系统参数和窗口数据。

    输入：WT 出力数据 pkl 路径；系统参数从同一目录自动读取。
    输出：包含 ``system_data`` 和 ``window_data`` 的字典。

    主要步骤：读取两个正式 pkl，再把 train/test 转换为统一窗口视图。
    """

    wt_path = Path(pkl_path)
    wt_data = load_dictionary(wt_path)
    system_data = load_dictionary(wt_path.with_name(SYSTEM_OUTPUT_FILE.name))
    return {
        "system_data": system_data,
        "window_data": build_window_view(wt_data),
    }
