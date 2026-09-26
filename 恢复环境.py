"""38 步恢复环境：动作、状态转移和修订稿目标奖励。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HORIZON = 38
FUTURE_WIND_STEPS = 18
SHORTAGE_PENALTY = 5.0


def positions_for_window_ids(
    window_data: dict,
    window_ids: np.ndarray | list[int],
    dataset_split: str | None = None,
) -> np.ndarray:
    """把窗口 ID 转换为数据位置，并检查重复、缺失和数据划分。"""

    requested_ids = np.asarray(window_ids, dtype=np.int64).reshape(-1)
    if len(np.unique(requested_ids)) != len(requested_ids):
        raise ValueError("窗口 ID 清单中存在重复值")

    all_ids = np.asarray(window_data["window_ids"], dtype=np.int64)
    splits = np.asarray(window_data["dataset_split"])
    position_by_id = {
        int(window_id): position for position, window_id in enumerate(all_ids)
    }
    missing = [
        int(window_id)
        for window_id in requested_ids
        if int(window_id) not in position_by_id
    ]
    if missing:
        raise ValueError(f"窗口 ID 不在数据集中：{missing[:10]}")

    positions = np.asarray(
        [position_by_id[int(window_id)] for window_id in requested_ids], dtype=int
    )
    if dataset_split is not None:
        wrong_split = requested_ids[splits[positions] != dataset_split]
        if len(wrong_split):
            raise ValueError(
                f"窗口 ID 不属于 {dataset_split} 划分：{wrong_split[:10].tolist()}"
            )
    return positions


class RestorationEnv(gym.Env):
    """用一个 38 步风电窗口执行网架恢复。

    输入：系统参数、窗口数组、数据划分和可选的固定窗口 ID。
    输出：Gymnasium 环境；每个动作包含 1 个线路编号和 21 个负荷挡位。
    负荷挡位 0/1/2 表示 0%/50%/100%，动作前已通电的母线才能升档。
    当前奖励为修订稿目标的归一化值：``(P_L-5*s)/E_ref``。
    """

    def __init__(
        self,
        system_data: dict,
        wind_data: dict,
        dataset_split: str = "train",
        window_ids: np.ndarray | list[int] | None = None,
    ):
        """读取系统数据并建立动作、观测空间。

        输入：``system_data``、``wind_data``、``train/test`` 划分和可选窗口池。
        输出：无；环境创建后还需要调用 ``reset`` 才能执行动作。
        """

        super().__init__()
        self.wind = wind_data
        self.positions = np.flatnonzero(
            np.asarray(wind_data["dataset_split"]) == dataset_split
        )
        if window_ids is not None:
            self.positions = positions_for_window_ids(
                wind_data, window_ids, dataset_split
            )
        if len(self.positions) == 0:
            raise ValueError(f"{dataset_split} 没有可用恢复窗口")

        generators = system_data["generators"]
        loads = system_data["loads"]
        branches = system_data["branches"]
        self.generator_bus = np.asarray(generators["bus_index"], dtype=int)
        self.generator_capacity = np.asarray(generators["capacity_mw"], dtype=float)
        self.generator_ramp = np.asarray(generators["ramp_mw_per_step"], dtype=float)
        self.is_wind = np.asarray(generators["type"]) == "WT"
        self.initial_power = np.asarray(
            generators["initial_power_mw"], dtype=float
        )
        if self.initial_power.shape != self.generator_capacity.shape:
            raise ValueError("initial_power_mw 与机组数量不一致")
        if not np.allclose(self.initial_power, 0.0):
            raise ValueError("修订实验要求所有机组初始出力为 0 MW")
        self.black_start_index = int(generators["black_start_index"])
        self.black_start_bus = int(self.generator_bus[self.black_start_index])
        self.load_bus = np.asarray(loads["bus_index"], dtype=int)
        self.load_capacity = np.asarray(loads["capacity_mw"], dtype=float)
        self.branch_endpoints = np.asarray(branches["endpoints"], dtype=int)
        self.bus_count = int(system_data["bus_count"])
        self.generator_count = len(self.generator_bus)
        self.load_count = len(self.load_bus)
        self.branch_count = len(self.branch_endpoints)
        self.total_load_mw = float(system_data["total_load_mw"])
        self.interval_hours = float(system_data["interval_hours"])
        self.horizon = int(system_data["horizon"])
        if self.horizon != HORIZON:
            raise ValueError(f"环境要求 horizon={HORIZON}，当前为 {self.horizon}")

        reference = np.asarray(wind_data["E"], dtype=float)
        fallback = max(self.total_load_mw * self.horizon, 1.0)
        self.reference_energy = np.where(
            np.isfinite(reference) & (reference > 0), reference, fallback
        )
        wind_capacity = self.generator_capacity[self.is_wind]
        self.action_space = spaces.MultiDiscrete(
            np.asarray([self.branch_count] + [3] * self.load_count, dtype=np.int64)
        )
        self.observation_space = spaces.Dict({
            "line_state": spaces.Box(0, 1, (2 * self.branch_count,), np.float32),
            "bus_state": spaces.Box(0, 1, (2 * self.bus_count,), np.float32),
            "load_state": spaces.Box(0, 1, (3 * self.load_count,), np.float32),
            "time_fraction": spaces.Box(0, 1, (1,), np.float32),
            "future_wind_mw": spaces.Box(
                low=np.zeros(FUTURE_WIND_STEPS * len(wind_capacity), dtype=np.float32),
                high=np.tile(wind_capacity, FUTURE_WIND_STEPS).astype(np.float32),
                dtype=np.float32,
            ),
        })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """选择窗口并恢复到 t=0 初态。

        输入：可选随机种子；``options['window_id']`` 可固定测试窗口。
        输出：初始观测和空信息字典。
        """

        super().reset(seed=seed)
        window_id = (options or {}).get("window_id")
        if window_id is None:
            self.window_position = int(
                self.positions[self.np_random.integers(len(self.positions))]
            )
        else:
            matches = self.positions[
                self.wind["window_ids"][self.positions] == int(window_id)
            ]
            if len(matches) != 1:
                raise ValueError(f"窗口 ID 不在当前环境：{window_id}")
            self.window_position = int(matches[0])

        self.current_t = 0
        self.line_closed = np.zeros(self.branch_count, dtype=bool)
        self.bus_energized = np.zeros(self.bus_count, dtype=bool)
        self.bus_energized[self.black_start_bus] = True
        self.generator_start_t = np.full(self.generator_count, -1, dtype=int)
        self.generator_start_t[self.black_start_index] = 0
        self.load_level = np.zeros(self.load_count, dtype=np.int8)
        self.cumulative_objective = 0.0
        self.cumulative_energy_mwh = 0.0
        self.generation = self._available_generation(0)
        return self._observation(), {}

    def step(self, action: np.ndarray):
        """执行一个闭线和负荷挡位动作。

        输入：长度为 22 的联合动作，第一项是线路编号，后 21 项是挡位。
        输出：下一观测、归一化目标奖励、终止标志、截断标志和指标信息。
        关键顺序：先按动作前状态检查负荷，再计算下一状态出力，最后写入状态。
        """

        action = np.asarray(action, dtype=np.int64).reshape(-1)
        reason = self._check_action(action)
        if reason is not None:
            return self._invalid_action(reason)

        branch = int(action[0])
        requested_level = action[1:].astype(np.int8)
        left, right = self.branch_endpoints[branch]
        new_bus = int(right if self.bus_energized[left] else left)
        next_t = self.current_t + 1
        next_generation = self._available_generation(next_t)
        generation_total = float(next_generation.sum())

        previous_load = self._load_mw(self.load_level)
        requested_load = self._load_mw(requested_level)
        deficit = max(0.0, requested_load - generation_total)
        objective_step = requested_load - SHORTAGE_PENALTY * deficit

        self.line_closed[branch] = True
        self.bus_energized[new_bus] = True
        newly_started = (self.generator_start_t < 0) & self.bus_energized[self.generator_bus]
        self.generator_start_t[newly_started] = next_t
        self.load_level = requested_level
        self.current_t = next_t
        self.generation = next_generation
        self.cumulative_objective += objective_step
        self.cumulative_energy_mwh += min(requested_load, generation_total) * self.interval_hours

        info = {
            "objective_step": objective_step,
            "cumulative_objective": self.cumulative_objective,
            "requested_load_mw": requested_load,
            "generation_mw": generation_total,
            "deficit_mw": deficit,
            "cumulative_energy_mwh": self.cumulative_energy_mwh,
            "restoration_rate": min(requested_load, generation_total) / self.total_load_mw,
        }
        return (
            self._observation(),
            float(objective_step / self.reference_energy[self.window_position]),
            self.current_t >= self.horizon,
            False,
            info,
        )

    def action_masks(self) -> np.ndarray:
        """生成 MaskablePPO 的 109 维动作掩码。

        输出：46 个线路动作和 21×3 个挡位动作的布尔数组。
        规则：线路必须是一端通电的未闭合线路；挡位只能保持或上升，
        且升档负荷的母线必须在动作前已通电。
        """

        endpoints = self.bus_energized[self.branch_endpoints]
        branch_mask = (~self.line_closed) & (endpoints.sum(axis=1) == 1)
        eligible = self.bus_energized[self.load_bus]
        levels = np.arange(3)[None, :]
        load_mask = (levels >= self.load_level[:, None]) & eligible[:, None]
        load_mask[np.arange(self.load_count), self.load_level] = True
        return np.concatenate((branch_mask, load_mask.reshape(-1)))

    def _check_action(self, action: np.ndarray) -> str | None:
        """检查动作形状、线路前沿和负荷挡位合法性。

        输入：待执行的整数动作。
        输出：合法时返回 ``None``，非法时返回中文原因。
        """

        if len(action) != 1 + self.load_count:
            return f"动作长度应为 {1 + self.load_count}"
        branch = int(action[0])
        if branch < 0 or branch >= self.branch_count:
            return "线路编号越界"
        if np.any(action[1:] < 0) or np.any(action[1:] > 2):
            return "负荷挡位必须为 0、1 或 2"
        mask = self.action_masks()
        offsets = self.branch_count + np.arange(self.load_count) * 3
        selected = np.concatenate(([mask[branch]], mask[offsets + action[1:]]))
        if not selected.all():
            return "线路不是前沿线路，或负荷挡位违反恢复约束"
        return None

    def _available_generation(self, time_index: int) -> np.ndarray:
        """计算当前状态可用的最大机组出力。

        输入：状态时步 ``time_index``，取值 0..38。
        输出：各机组最大可用出力数组。
        常规机组按 p(0)=0 和向上爬坡率计算；风电按状态前母线和窗口出力计算。
        """

        started = self.generator_start_t >= 0
        elapsed = np.maximum(time_index - self.generator_start_t, 0)
        generation = np.minimum(
            self.generator_capacity, self.generator_ramp * elapsed
        ) * started
        wind_started = started[self.is_wind] & (elapsed[self.is_wind] >= 1)
        if time_index > 0:
            wind_power = self.wind["current_power"][self.window_position, time_index - 1]
            generation[self.is_wind] = np.where(wind_started, wind_power, 0.0)
        else:
            generation[self.is_wind] = 0.0
        return generation

    def _load_mw(self, level: np.ndarray) -> float:
        """把 0/1/2 负荷挡位转换为 MW。

        输入：负荷挡位数组；输出：``sum(0.5*Pmax*k)``。
        """

        return float(np.dot(level / 2.0, self.load_capacity))

    def _invalid_action(self, reason: str):
        """返回非法动作的固定惩罚并保持状态不变。

        输入：非法原因；输出：当前观测、惩罚、截断标志和原因信息。
        """

        return (
            self._observation(),
            -1.0,
            False,
            True,
            {
                "invalid_action": reason,
                "cumulative_objective": self.cumulative_objective,
                "cumulative_energy_mwh": self.cumulative_energy_mwh,
            },
        )

    def _observation(self) -> dict[str, np.ndarray]:
        """把当前恢复状态编码成策略网络的字典观测。"""

        future = np.minimum(
            self.current_t + np.arange(FUTURE_WIND_STEPS), self.horizon - 1
        )
        wind = self.wind["current_power"][self.window_position, future].astype(np.float32)
        return {
            "line_state": np.eye(2, dtype=np.float32)[self.line_closed.astype(int)].reshape(-1),
            "bus_state": np.eye(2, dtype=np.float32)[self.bus_energized.astype(int)].reshape(-1),
            "load_state": np.eye(3, dtype=np.float32)[self.load_level].reshape(-1),
            "time_fraction": np.asarray([self.current_t / self.horizon], dtype=np.float32),
            "future_wind_mw": wind.reshape(-1),
        }
