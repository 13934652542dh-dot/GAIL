"""
MIP 专家数据求解与回放校验。

两件事，对应两个公开函数：
    solve_mip           按顺序求解顶部配置的 train 连续区间。
    verify_mip_in_env   读取已保存的专家数据，在恢复环境中逐步回放。

同一批专家数据同时用于 B 拟合 E 和 BC/GAIL 模仿学习，不重复求解。

执行流程：
    1. 读取系统参数和 train 窗口；
    2. 按数据原始顺序选取顶部配置的 train 连续区间；
    3. 单进程依次调用 MIP.py，单个模型使用 Gurobi 全部可用线程；
    4. 保存线路闭合顺序、负荷恢复轨迹、目标值、B 和 E；
    5. 在恢复环境中回放专家轨迹并保存校验结果。
"""

import pickle
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from 建模代码.MIP import build_model
from 建模代码.恢复环境 import RestorationEnv


# ======================================================================
# 文件定位与完整实验流程
# ======================================================================
# 本文件是专家数据入口，直接运行时依次完成：
#   选择当前 train 顺序区间 -> 逐窗口调用 MIP.py -> 保存专家 pkl
#   -> 在 RestorationEnv 中逐动作回放 -> 保存回放 pkl。
#
# 三个文件的分工：
#   MIP.py       ：只建立并求解一个窗口，不选样本、不保存文件；
#   solve_MIP.py ：选择一批窗口、批量求解、命名和保存、触发回放；
#   恢复环境.py  ：按 RL 状态转移规则重放专家动作，检查两套实现是否一致。
#
# 本文件不会训练 BC、GAIL 或 PPO。保存的专家 pkl 后续同时供 E' 拟合、BC 和
# GAIL 使用，因此同一批窗口只需要进行一次 MIP 求解。
RESULTS_DIR = PROJECT_DIR / "data" / "专家求解数据" / "train"
DEFAULT_PKL_PATH = PROJECT_DIR / "data" / "正式实验输入" / "WT的出力数据.pkl"
SYSTEM_OUTPUT_FILE = PROJECT_DIR / "data" / "正式实验输入" / "系统电气参数.pkl"

# ======================================================================
# 本次专家批次：通常只需要修改下面两行
# ======================================================================
# SAMPLE_START 和 SAMPLE_END 表示 train 字典内的顺序位置，且首尾都包含；
# train 字典的窗口编号从 0 开始，程序直接使用该局部编号作为 window_id。
# 正式实验只求解 train 顺序中的前 100 个场景；该批结果
# 同时用于 E(B) 拟合以及 BC/GAIL 预训练，不再求解额外专家批次。
SAMPLE_START = 0
SAMPLE_END = 99
MIP_RESULTS_FILE = (
    RESULTS_DIR / f"MIP_{DEFAULT_PKL_PATH.stem}_{SAMPLE_START}-{SAMPLE_END}.pkl"
)

# 当前配置对应的专家文件示例：
#   data/专家求解数据/train/MIP_WT的出力数据_0-99.pkl
# 对应回放文件由 verify_mip_in_env 自动命名为：
#   data/专家求解数据/train/MIP回放_WT的出力数据_0-99.pkl

# ======================================================================
# 关键实验参数
# ======================================================================
# TIME_LIMIT：每个窗口的最长求解时间，单位为秒；达到时限后保留当前可行解。
# 达到时限不等于必然失败；只要 Gurobi 已找到可行解，该轨迹仍会被保存。
TIME_LIMIT = 20.0

# MIP_GAP：Gurobi 的相对最优间隙；0.1 表示上下界差距达到 10% 时允许停止。
MIP_GAP = 0.1

# GUROBI_THREADS：0 表示由 Gurobi 自动使用当前机器允许的全部线程。
# Python 层仍是单进程逐窗口求解；“全部线程”只作用于当前一个 Gurobi 模型。
GUROBI_THREADS = 0


def solve_mip(
    pkl_path: str | Path = DEFAULT_PKL_PATH,
    time_limit: float = TIME_LIMIT,
    gap: float = MIP_GAP,
) -> Path:
    """串行求解train中的配置区间，并保存MIP专家数据。

    输入：
        pkl_path：正式实验输入数据文件路径。
        time_limit：每个窗口的最大求解时间，默认 20 秒。
        gap：每个窗口的相对目标间隙，默认 0.1。
    输出：
        Path，专家数据文件的保存路径。

    主要步骤：
        1. 读取系统参数和全部风电窗口；
        2. 从train字典顺序选取配置区间；
        3. 为每个窗口调用 MIP.py 中的 build_model；
        4. 提取线路闭合顺序、负荷恢复轨迹和目标值；
        5. 根据负荷恢复轨迹计算恢复电量 E；
        6. 按抽样顺序保存全部窗口的专家结果。

    每个结果同时包含 B 和 E，可供线性拟合使用；线路顺序和负荷轨迹
    可供 BC、GAIL 等模仿学习方法使用，因此不需要重复求解。
    """

    # ================================================================
    # 1. 读取正式实验数据
    # ================================================================
    # 只读取 train 字典，不读取 WT 文件中的 test 字典；MIP专家数据只来自 train。
    wt_path = Path(pkl_path)
    with wt_path.open("rb") as file:
        train_data = pickle.load(file)["train"]
    with wt_path.with_name(SYSTEM_OUTPUT_FILE.name).open("rb") as file:
        system = pickle.load(file)
    windows = {
        "window_ids": np.arange(len(train_data["B"]), dtype=np.int64),
        "current_power": train_data["WT_power_mw"],
        "B": train_data["B"],
        "E": train_data["E"],
    }
    # ================================================================
    # 2. 确定本次需要求解的窗口
    # ================================================================
    all_window_ids = np.asarray(windows["window_ids"], dtype=np.int64)
    # windows 只包含 train；当前区间由顶部两项参数确定，并进入保存文件名。
    positions = np.arange(len(all_window_ids))[SAMPLE_START:SAMPLE_END + 1]
    window_ids = all_window_ids[positions]
    sample_count = len(window_ids)

    sampled_windows = [
        {
            "window_ids": np.asarray([windows["window_ids"][position]]),
            "current_power": np.asarray(
                windows["current_power"][position:position + 1]
            ),
            "B": np.asarray([windows["B"][position]]),
            "E": np.asarray([windows["E"][position]]),
        }
        for position in positions
    ]

    # ================================================================
    # 3. 单进程依次调用 MIP.py 求解各窗口
    # ================================================================
    # 每次只向 build_model 传入一个窗口，所以 position 参数固定使用 0。
    # results 与本批窗口等长，即使某个窗口无解，也保留它的位置和 window_id。
    results = [None] * sample_count
    # 批量开始前直接设置一次Gurobi参数，后续所有窗口沿用同一配置。
    gp.setParam("TimeLimit", time_limit)
    gp.setParam("MIPGap", gap)
    gp.setParam("Threads", GUROBI_THREADS)
    for index, sampled_window in enumerate(sampled_windows):
        result = build_model(system, sampled_window, 0)

        # ============================================================
        # 4. 整理当前窗口的求解结果
        # ============================================================
        if result is None:
            # Gurobi 未找到任何可行解时保留窗口编号，并用 None 对齐结果位置。
            # 代码、数据或许可证错误不会在这里吞掉，而是直接中止并显示原始异常。
            results[index] = {
                "window_id": int(window_ids[index]),
                "order": None,
                "load_trajectory": None,
                "obj_val": None,
                "B": float(windows["B"][positions[index]]),
                "E_mwh": None,
            }
        else:
            order, load_trajectory, objective = result
            load_trajectory = np.asarray(load_trajectory, dtype=np.int8)
            load_capacity = np.asarray(
                system["loads"]["capacity_mw"], dtype=float
            )
            interval_hours = float(system["interval_hours"])
            results[index] = {
                # window_id 是 train 字典内的局部编号，与本批顺序范围一致。
                "window_id": int(window_ids[index]),
                # order 和 load_trajectory 是 BC/GAIL 使用的专家动作来源。
                "order": np.asarray(order, dtype=np.int64),
                "load_trajectory": load_trajectory,
                # obj_val 是 MIP 的 38 步原始目标函数值。
                "obj_val": float(objective),
                # B 和 E_mwh 用于后续拟合奖励归一化基准 E'=alpha+beta*B。
                "B": float(windows["B"][positions[index]]),
                "E_mwh": float(
                    (load_trajectory * load_capacity / 2.0).sum()
                    * interval_hours
                ),
            }

        print(
            f"  [{index + 1}/{sample_count}] window_id={window_ids[index]} "
            f"{'不可行' if result is None else '可行'}",
            flush=True,
        )

    # ================================================================
    # 5. 保存统一专家数据文件
    # ================================================================
    # 专家结果固定写入 data/专家求解数据/train/；本项目不求解 test 专家数据。
    # 文件名含 train 顺序范围，后续批次不会覆盖已有结果。
    result_file = (
        RESULTS_DIR
        / f"MIP_{Path(pkl_path).stem}_{SAMPLE_START}-{SAMPLE_END}.pkl"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with result_file.open("wb") as file:
        pickle.dump(results, file)

    feasible_count = sum(result["order"] is not None for result in results)
    print(
        f"MIP 求解完成：{feasible_count}/{sample_count} 个窗口可行，"
        f"已保存至 {result_file}"
    )
    return result_file


def verify_mip_in_env(
    mip_path: str | Path = MIP_RESULTS_FILE,
    pkl_path: str | Path = DEFAULT_PKL_PATH,
) -> Path:
    """读取 MIP 专家数据，在恢复环境中逐步回放全部轨迹。

    输入：
        mip_path：solve_mip 保存的专家数据文件路径。
        pkl_path：建立恢复环境所需的正式实验输入文件路径。

    输出：
        Path，环境回放结果文件的保存路径。

    主要步骤：
        1. 读取已保存的 MIP 专家数据；
        2. 使用可行窗口建立恢复环境；
        3. 将线路编号和负荷挡位组合成完整动作；
        4. 在环境中逐时步执行专家动作；
        5. 累计环境返回的奖励，还原后与 MIP 目标值比较；
        6. 保存每个窗口的回放结果。
    """

    # ================================================================
    # 1. 读取专家数据并建立恢复环境
    # ================================================================
    # 回放不再调用 Gurobi，也不修改专家文件，只验证已保存轨迹能否被环境接受。
    with Path(mip_path).open("rb") as file:
        mip_results = pickle.load(file)

    wt_path = Path(pkl_path)
    with wt_path.open("rb") as file:
        train_data = pickle.load(file)["train"]
    with wt_path.with_name(SYSTEM_OUTPUT_FILE.name).open("rb") as file:
        system = pickle.load(file)
    windows = {
        "window_ids": np.arange(len(train_data["B"]), dtype=np.int64),
        "current_power": train_data["WT_power_mw"],
        "B": train_data["B"],
        "E": train_data["E"],
    }
    feasible_ids = [
        result["window_id"]
        for result in mip_results
        if result["order"] is not None
    ]
    if not feasible_ids:
        raise ValueError("专家数据中没有可供环境回放的可行窗口")
    env = RestorationEnv(system, windows, window_ids=feasible_ids)

    # ================================================================
    # 2. 逐窗口回放线路动作和负荷恢复动作
    # ================================================================
    # MIP 的 order 与 load_trajectory 会在这里还原为环境的完整联合动作：
    # [线路编号, 负荷1挡位, ..., 负荷21挡位]。
    results = []
    try:
        for result in mip_results:
            if result["order"] is None:
                results.append({
                    "window_id": result["window_id"],
                    "mip_obj": None,
                    "env_reward": None,
                    "reward_objective": None,
                    "diff": None,
                    "match": None,
                })
                continue

            window_id = int(result["window_id"])
            env.reset(options={"window_id": window_id})
            # 每行动作为：[闭合线路编号, 21 个负荷的恢复挡位]。
            trajectory = np.column_stack((
                np.asarray(result["order"], dtype=np.int64),
                np.asarray(result["load_trajectory"], dtype=np.int8),
            ))
            reward_sum = 0.0
            for action in trajectory:
                _, reward, _, _, _ = env.step(action)
                reward_sum += float(reward)

            # 同一窗口的 E' 在全轨迹内不变，因此累计奖励乘回 E'
            # 就是与 MIP ObjVal 同口径的未归一化目标。
            reward_objective = reward_sum * float(
                env.reference_energy[env.window_position]
            )
            difference = reward_objective - float(result["obj_val"])
            tolerance = max(1e-6 * abs(float(result["obj_val"])), 1e-4)
            match = abs(difference) <= tolerance
            results.append({
                "window_id": window_id,
                "mip_obj": result["obj_val"],
                "env_reward": reward_sum,
                "reward_objective": reward_objective,
                "diff": difference,
                "match": match,
            })
    finally:
        env.close()

    # ================================================================
    # 3. 保存环境回放结果
    # ================================================================
    # 回放结果与专家文件位于同一目录，并在原文件名前增加“MIP回放_”。
    # 每条记录保存 MIP 目标、环境奖励、还原目标及差值，不替换原专家数据。
    data_name = Path(mip_path).stem.removeprefix("MIP_")
    verify_file = Path(mip_path).parent / f"MIP回放_{data_name}.pkl"
    with verify_file.open("wb") as file:
        pickle.dump(results, file)
    feasible_count = sum(result["mip_obj"] is not None for result in results)
    match_count = sum(bool(result["match"]) for result in results)
    print(
        f"MIP-环境回放完成：{match_count}/{feasible_count} 个可行窗口通过，"
        f"已保存至 {verify_file}"
    )
    return verify_file


if __name__ == "__main__":
    # 直接运行本文件时，求解当前配置批次，再回放该批专家轨迹。
    # 修改 SAMPLE_START/SAMPLE_END 后，下列默认路径会随模块常量自动更新。
    mip_path = solve_mip()
    verify_mip_in_env(mip_path)
