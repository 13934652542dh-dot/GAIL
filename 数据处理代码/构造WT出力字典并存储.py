"""构造并保存正式实验所需的两个字典。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESS_DIR = PROJECT_DIR / "data" / "处理过程"
FORMAL_INPUT_DIR = PROJECT_DIR / "data" / "正式实验输入"
PARAMETER_FILE = PROCESS_DIR / "基本电气参数.xlsx"
WT_OUTPUT_FILE = FORMAL_INPUT_DIR / "WT的出力数据.pkl"
SYSTEM_OUTPUT_FILE = FORMAL_INPUT_DIR / "系统电气参数.pkl"


def read_system_parameters(parameter_file: Path = PARAMETER_FILE) -> dict:
    """读取基本电气参数并生成系统电气参数字典。

    输入：包含“传统机组”“新能源机组”“负荷”“线路”四个页面的 Excel 路径。
    输出：供恢复环境和 MIP 使用的系统电气参数字典。

    主要步骤：
    1. 读取四个参数表；
    2. 将节点编号转换为程序使用的 0 起始索引；
    3. 按固定顺序合并传统机组和风电机组参数；
    4. 组装母线、机组、负荷和线路数据。
    """

    traditional = pd.read_excel(parameter_file, sheet_name="传统机组")
    wind = pd.read_excel(parameter_file, sheet_name="新能源机组")
    loads = pd.read_excel(parameter_file, sheet_name="负荷")
    lines = pd.read_excel(parameter_file, sheet_name="线路")
    generator_bus = np.concatenate([
        traditional["所在节点"].to_numpy(dtype=int) - 1,
        wind["所在节点"].to_numpy(dtype=int) - 1,
    ])
    black_start_index = int(
        np.flatnonzero(traditional["是否黑启动"].to_numpy(dtype=int))[0]
    )
    initial_power = np.zeros(len(generator_bus), dtype=float)
    initial_power[black_start_index] = 100.0
    return {
        "bus_count": 39,
        "horizon": 38,
        "interval_hours": 0.25,
        "total_load_mw": float(loads["容量"].sum()),
        "generators": {
            "bus_index": generator_bus,
            "capacity_mw": np.concatenate([
                traditional["容量"].to_numpy(dtype=float),
                wind["容量"].to_numpy(dtype=float),
            ]),
            "ramp_mw_per_step": np.concatenate([
                traditional["爬坡率"].to_numpy(dtype=float),
                np.zeros(len(wind), dtype=float),
            ]),
            "type": np.array(["CG"] * len(traditional) + ["WT"] * len(wind)),
            "black_start_index": black_start_index,
            "initial_power_mw": initial_power,
        },
        "loads": {
            "bus_index": loads["所在节点"].to_numpy(dtype=int) - 1,
            "capacity_mw": loads["容量"].to_numpy(dtype=float),
        },
        "branches": {
            "endpoints": lines[["起始节点", "结束节点"]].to_numpy(dtype=int) - 1,
        },
    }


def build_wt_split_dictionary(
    power_mw: np.ndarray,
    b_values: np.ndarray,
    e_values: np.ndarray,
) -> dict:
    """生成一个 train 或 test 字典。

    输入：形状为 ``(样本数, 38, 6)`` 的风机出力、形状均为
    ``(样本数,)`` 的 B 数组和 E 数组。
    输出：只包含 ``WT_power_mw``、``B`` 和 ``E`` 的窗口字典。

    说明：train/test 已经由上层字典键表达，因此不再重复保存
    ``sample_type`` 和 ``sample_id``。

    主要步骤：将窗口出力、B 和 E 按统一字段名放入一个字典，不复制或
    改变输入数组内容。
    """

    return {
        "WT_power_mw": power_mw,
        "B": b_values,
        "E": e_values,
    }


def save_dictionary(data: dict, output_file: Path) -> None:
    """将一个字典保存为 pkl 文件。

    输入：待保存的字典和目标 pkl 路径。
    输出：无；在目标路径写出 pkl 文件。

    主要步骤：先创建目标目录，再使用最高协议序列化字典，保证 NumPy
    数组等数据结构可以被后续环境和求解器直接读取。
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
