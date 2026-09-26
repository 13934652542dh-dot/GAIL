"""
MIP 专家数据求解与回放校验。

两件事，对应两个公开函数：
    solve_mip           随机抽取 train 窗口，或求解指定的窗口编号。
    verify_mip_in_env   读取已保存的专家数据，在恢复环境中逐步回放。

同一批专家数据同时用于 B 拟合 E 和 BC/GAIL 模仿学习，不重复求解。

执行流程：
    1. 读取系统参数和 train 窗口；
    2. 使用固定随机种子无放回抽取 500 个窗口；
    3. 并行调用 MIP.py 求解每个窗口；
    4. 保存线路闭合顺序、负荷恢复轨迹、目标值、B 和 E；
    5. 在恢复环境中回放专家轨迹并保存校验结果。
"""

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gurobipy as gp
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from 建模代码.MIP import build_model
from 建模代码.恢复环境 import RestorationEnv
from 建模代码.读取字典数据 import DEFAULT_PKL_PATH, load_experiment_data


RESULTS_DIR = PROJECT_DIR / "data" / "专家求解数据" / "train"
TEST_RESULTS_DIR = PROJECT_DIR / "data" / "专家求解数据" / "test对比专家数据"
DATA_NAME = Path(DEFAULT_PKL_PATH).stem
EXPERIMENT_NAME = f"MIP_{DATA_NAME}"
MIP_RESULTS_FILE = RESULTS_DIR / f"MIP_{DATA_NAME}.pkl"

# ======================================================================
# 关键实验参数
# ======================================================================
# RANDOM_SEED：固定 train 窗口的随机抽样结果，保证重复运行使用同一批窗口。
RANDOM_SEED = 10

# SAMPLE_COUNT：本次专家求解使用的训练窗口数量。
SAMPLE_COUNT = 500

# TIME_LIMIT：每个窗口的最长求解时间，单位为秒；达到时限后保留当前可行解。
TIME_LIMIT = 20.0

# MIP_GAP：Gurobi 的相对最优间隙；0.1 表示上下界差距达到 10% 时允许停止。
MIP_GAP = 0.1

# MAX_WORKERS：同时运行的 MIP 求解进程数；4 表示同时求解 4 个窗口。
MAX_WORKERS = 4

# THREADS_PER_WORKER：每个 Gurobi 进程使用 4 个线程，4 个进程共使用约 16 个线程。
THREADS_PER_WORKER = 4


def set_mip_parameters(time_limit: float, gap: float, threads: int) -> None:
    """设置一个并行进程中所有 MIP 子问题的求解参数。

    输入：
        time_limit：单个窗口允许的最大求解时间，单位为秒。
        gap：Gurobi 的相对目标间隙，例如 0.1 表示 10%。
        threads：每个 Gurobi 求解进程内部使用的线程数。

    输出：
        无返回值，参数直接写入当前子进程的 Gurobi 环境。

    该函数作为进程池初始化器，每个子进程只执行一次，避免在 MIP.py
    中重复设置参数，使 MIP.py 只负责数学模型的建立和求解。
    """

    gp.setParam("TimeLimit", time_limit)
    gp.setParam("MIPGap", gap)
    gp.setParam("Threads", threads)


def solve_mip(
    pkl_path: str | Path = DEFAULT_PKL_PATH,
    time_limit: float = TIME_LIMIT,
    gap: float = MIP_GAP,
    seed: int = RANDOM_SEED,
    sample_count: int = SAMPLE_COUNT,
    window_ids: list[int] | np.ndarray | None = None,
) -> Path:
    """并行求解随机抽取或直接指定的窗口，并保存 MIP 专家数据。

    输入：
        pkl_path：正式实验输入数据文件路径。
        time_limit：每个窗口的最大求解时间，默认 20 秒。
        gap：每个窗口的相对目标间隙，默认 0.1。
        seed：训练窗口抽样使用的随机种子，默认 10。
        sample_count：无放回抽取的训练窗口数量，默认 500。
        window_ids：需要直接求解的窗口编号；不传时从 train 集随机抽样。

    输出：
        Path，专家数据文件的保存路径。

    主要步骤：
        1. 读取系统参数和全部风电窗口；
        2. 从 train 集固定抽样，或读取调用方指定的窗口编号；
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
    data = load_experiment_data(pkl_path)
    system = data["system_data"]
    windows = data["window_data"]
    # ================================================================
    # 2. 确定本次需要求解的窗口
    # ================================================================
    all_window_ids = np.asarray(windows["window_ids"], dtype=np.int64)
    if window_ids is None:
        train_positions = np.flatnonzero(
            np.asarray(windows["dataset_split"]) == "train"
        )
        positions = np.random.default_rng(seed).choice(
            train_positions, size=sample_count, replace=False
        )
        window_ids = all_window_ids[positions]
    else:
        window_ids = np.asarray(window_ids, dtype=np.int64)
        position_by_id = {
            int(window_id): position
            for position, window_id in enumerate(all_window_ids)
        }
        positions = np.asarray(
            [position_by_id[int(window_id)] for window_id in window_ids],
            dtype=np.int64,
        )
        sample_count = len(window_ids)

    selected_splits = np.asarray(windows["dataset_split"])[positions]
    sampled_windows = [
        {
            "window_ids": np.asarray([windows["window_ids"][position]]),
            "dataset_split": np.asarray([windows["dataset_split"][position]]),
            "current_power": np.asarray(
                windows["current_power"][position:position + 1]
            ),
            "B": np.asarray([windows["B"][position]]),
            "E": np.asarray([windows["E"][position]]),
        }
        for position in positions
    ]

    # ================================================================
    # 3. 并行调用 MIP.py 求解各窗口
    # ================================================================
    results = [None] * sample_count
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=set_mip_parameters,
        initargs=(time_limit, gap, THREADS_PER_WORKER),
    ) as executor:
        futures = {
            executor.submit(build_model, system, sampled_window, 0): index
            for index, sampled_window in enumerate(sampled_windows)
        }

        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception as error:
                print(f"  [警告] 样本 {index} MIP 求解异常：{error}")
                result = None

            # ========================================================
            # 4. 整理当前窗口的求解结果
            # ========================================================
            if result is None:
                # E 为整个恢复过程中各负荷恢复电量的总和，单位为 MWh。
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
                    "window_id": int(window_ids[index]),
                    "order": np.asarray(order, dtype=np.int64),
                    "load_trajectory": load_trajectory,
                    "obj_val": float(objective),
                    "B": float(windows["B"][positions[index]]),
                    "E_mwh": float(
                        (load_trajectory * load_capacity / 2.0).sum()
                        * interval_hours
                    ),
                }

            completed += 1
            print(
                f"  [{completed}/{sample_count}] window_id={window_ids[index]} "
                f"{'不可行' if result is None else '可行'}",
                flush=True,
            )

    # ================================================================
    # 5. 保存统一专家数据文件
    # ================================================================
    data_name = Path(pkl_path).stem
    result_dir = RESULTS_DIR if selected_splits[0] == "train" else TEST_RESULTS_DIR
    result_file = result_dir / f"MIP_{data_name}.pkl"
    result_dir.mkdir(parents=True, exist_ok=True)
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
        5. 比较环境累计目标值和 MIP 目标值；
        6. 保存每个窗口的回放结果。
    """

    # ================================================================
    # 1. 读取专家数据并建立恢复环境
    # ================================================================
    with Path(mip_path).open("rb") as file:
        mip_results = pickle.load(file)

    data = load_experiment_data(pkl_path)
    feasible_ids = [
        result["window_id"]
        for result in mip_results
        if result["order"] is not None
    ]
    if not feasible_ids:
        raise ValueError("专家数据中没有可供环境回放的可行窗口")
    all_window_ids = np.asarray(data["window_data"]["window_ids"], dtype=np.int64)
    position_by_id = {
        int(window_id): position
        for position, window_id in enumerate(all_window_ids)
    }
    first_position = position_by_id[int(feasible_ids[0])]
    dataset_split = str(data["window_data"]["dataset_split"][first_position])
    env = RestorationEnv(
        data["system_data"], data["window_data"], dataset_split, window_ids=feasible_ids
    )

    # ================================================================
    # 2. 逐窗口回放线路动作和负荷恢复动作
    # ================================================================
    results = []
    try:
        for result in mip_results:
            if result["order"] is None:
                results.append({
                    "window_id": result["window_id"],
                    "mip_obj": None,
                    "env_obj": None,
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
            info = {}
            valid = True
            for action in trajectory:
                _, _, _, truncated, info = env.step(action)
                if truncated:
                    valid = False
                    break

            env_objective = (
                float(info["cumulative_objective"]) if valid else None
            )
            difference = (
                env_objective - float(result["obj_val"])
                if valid else None
            )
            # 环境采用最大可用出力，目标值允许略高于 MIP 当前可行解。
            tolerance = max(1e-6 * abs(float(result["obj_val"])), 1e-4)
            match = valid and difference >= -tolerance
            results.append({
                "window_id": window_id,
                "mip_obj": result["obj_val"],
                "env_obj": env_objective,
                "diff": difference,
                "match": match,
            })
    finally:
        env.close()

    # ================================================================
    # 3. 保存环境回放结果
    # ================================================================
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
    # 直接运行本文件时，先求解 500 个训练窗口，再回放已保存的专家轨迹。
    data_path = DEFAULT_PKL_PATH
    mip_path = RESULTS_DIR / f"MIP_{Path(data_path).stem}.pkl"

    solve_mip(data_path)
    verify_mip_in_env(mip_path, data_path)
