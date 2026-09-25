"""提取六台风机的 38 步窗口，并划分 train/test。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "处理过程" / "六台风机15分钟仿真可用出力.csv"
TIME_COLUMN = "时间戳"
POWER_COLUMNS = [f"WT{i}仿真可用出力（MW）" for i in range(1, 7)]
WINDOW_LENGTH = 38
WINDOW_STEP = 1
TRAIN_RATIO = 0.7


def read_wt_output(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """读取固定的风机出力表。

    输入：连续 15 分钟出力 CSV 路径。
    输出：按时间升序排列、包含时间戳和六台风机出力的 DataFrame。

    主要步骤：读取指定列、解析时间戳、按时间排序，并重新生成连续索引。
    """

    return pd.read_csv(
        input_file,
        encoding="utf-8-sig",
        parse_dates=[TIME_COLUMN],
        usecols=[TIME_COLUMN, *POWER_COLUMNS],
    ).sort_values(TIME_COLUMN).reset_index(drop=True)


def extract_wt_windows(
    data: pd.DataFrame,
    window_length: int = WINDOW_LENGTH,
    window_step: int = WINDOW_STEP,
) -> np.ndarray:
    """按固定步长提取风机出力窗口。

    输入：连续风机出力表、窗口长度（默认 38 个时步）和窗口起点间隔
    （默认 1 个时步）。
    输出：形状为 ``(窗口数, 38, 6)`` 的风机出力数组，第三维依次为 WT1 到 WT6。
    power_windows[窗口编号, 时步编号, 风机编号]

    主要步骤：按 ``window_step`` 生成窗口起点，截取连续的
    ``window_length`` 个时步，并将每个窗口转换为 NumPy 数组。
    """

    starts = range(0, len(data) - window_length + 1, window_step)
    return np.stack([
        data.iloc[start : start + window_length][POWER_COLUMNS].to_numpy(dtype=np.float32)
        for start in starts
    ])


def split_windows(
    power_windows: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """按时间顺序划分已经提取的窗口。

    输入：全部窗口数组和训练集比例。
    输出：``(train_windows, test_windows)``；默认比例为 7:3。

    主要步骤：按照窗口在原始时间序列中的顺序计算切分位置，前段作为
    train，后段作为 test。
    """

    split_index = int(len(power_windows) * train_ratio)
    return power_windows[:split_index], power_windows[split_index:]


def calculate_b(power_windows: np.ndarray) -> np.ndarray:
    """计算每个窗口的 B。

    输入：形状为 ``(窗口数, 38, 6)`` 的风机出力数组。
    输出：每个窗口一个 B 值的数组。

    公式为 ``B_i = sum(t=1..38) sum(w=1..6) P_WT(i,t,w)``，单位为
    MW·step；如果换算为 15 分钟风电能量，则为 ``0.25 * B_i MWh``。

    主要步骤：沿时步和风机两个维度求和，并将结果保存为 float32。
    这里不再额外保留两位小数，以避免无必要的精度损失。
    """

    return power_windows.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)


def create_empty_e(sample_count: int) -> np.ndarray:
    """为尚未求解专家策略的窗口创建 E 占位数组。

    输入：窗口数量。
    输出：长度相同、全部为 NaN 的 E 数组。

    E 的实验定义为 ``E_i = 0.25 * sum(t=1..38) P_load(i,t)``，单位为
    MWh；预处理阶段没有专家恢复负荷数据，因此不擅自填入数值。

    主要步骤：创建指定长度的 float32 数组，并用 NaN 标记待后续拟合或
    写回的参考恢复电量。
    """

    return np.full(sample_count, np.nan, dtype=np.float32)
