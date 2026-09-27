"""38 步恢复环境：动作、状态转移和修订稿目标奖励。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ======================================================================
# 文件定位与实验口径
# ======================================================================
# 本文件定义同一个恢复问题在 Gymnasium 中的逐步交互形式，本身不保存任何文件。
# 主要由三类流程调用：
#   1. solve_MIP.py：回放 MIP 专家轨迹，检查专家动作是否满足环境约束；
#   2. BC/GAIL：把专家轨迹转换成“观测—动作”预训练样本；
#   3. PPO、BC+PPO、GAIL+PPO：在指定窗口池上训练并统一测试。
#
# 与 MIP.py 的共同口径：38 步、每步闭合一条前沿线路、负荷只能保持或升档、
# 单步原始目标均为 P_L-5s。主要区别是：MIP 同时优化机组出力，环境直接采用
# 当前状态下的最大可用出力；环境再用 E' 归一化单步奖励，供 PPO 稳定训练。
# info 中的 cumulative_objective 仍保留未除以 E' 的原始目标，便于和 MIP 比较。

# 策略观测包含从当前时步起的 18 步风电信息；接近末尾时重复最后一个时步。
FUTURE_WIND_STEPS = 18
# 缺额惩罚系数与 MIP.py 相同：单步原始目标为 P_L-5s。
SHORTAGE_PENALTY = 5.0


class RestorationEnv(gym.Env):
    """用一个 38 步风电窗口执行网架恢复。

    输入：系统参数、单一划分的窗口数组和可选的固定窗口 ID。
    输出：Gymnasium 环境；每个动作包含 1 个线路编号和 21 个负荷挡位。
    负荷挡位 0/1/2 表示 0%/50%/100%，动作前已通电的母线才能升档。
    当前奖励为修订稿目标的归一化值：``(P_L-5*s)/E_ref``。
    """

    def __init__(
        self,
        system_data: dict,
        wind_data: dict,
        window_ids: np.ndarray | list[int] | None = None,
    ):
        """读取系统数据并建立动作、观测空间。

        输入：``system_data`` 为电网参数，``wind_data`` 为已经选定的 train
        或 test 窗口数组；``window_ids`` 可进一步限制窗口池。
        输出：无；对象创建后需调用 ``reset`` 才能执行动作。
        作用：保存恢复问题状态，并提供 Gymnasium 的动作、观测和奖励接口。
        关键参数：环境固定使用 38 步、负荷挡位 0/1/2 和一端通电的线路动作。
        步骤：确定窗口池，读取系统参数，建立动作空间和观测空间。
        """

        super().__init__()
        self.wind = wind_data
        # train/test 在读取数据时已经分开；未传 window_ids 时使用当前字典全部窗口。
        # 传入后只使用清单中的窗口。
        # 训练 reset 随机抽取池中窗口，测试和专家回放则显式指定 window_id。
        self.positions = np.arange(len(wind_data["window_ids"]), dtype=int)
        if window_ids is not None:
            requested_ids = np.asarray(window_ids, dtype=np.int64).reshape(-1)
            position_by_id = {
                int(window_id): position
                for position, window_id in enumerate(wind_data["window_ids"])
            }
            self.positions = np.asarray(
                [position_by_id[int(window_id)] for window_id in requested_ids],
                dtype=int,
            )
        self.position_by_window_id = {
            int(wind_data["window_ids"][position]): int(position)
            for position in self.positions
        }

        # 读取机组、负荷和线路的正式实验参数。
        generators = system_data["generators"]
        loads = system_data["loads"]
        branches = system_data["branches"]
        self.generator_bus = np.asarray(generators["bus_index"], dtype=int)
        self.generator_capacity = np.asarray(generators["capacity_mw"], dtype=float)
        self.generator_ramp = np.asarray(generators["ramp_mw_per_step"], dtype=float)
        self.is_wind = np.asarray(generators["type"]) == "WT"
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

        # E' 是强化学习奖励的统一归一化基准。
        # 训练工具会按当前专家版本的拟合方程替换 wind_data["E"]。
        # E' 只改变奖励尺度，不改变 cumulative_objective 的原始目标口径。
        self.reference_energy = np.asarray(wind_data["E"], dtype=float)
        wind_capacity = self.generator_capacity[self.is_wind]
        # 联合动作有 1+load_count 个分量：第一项选线路，其余项选各负荷挡位。
        self.action_space = spaces.MultiDiscrete(
            np.asarray([self.branch_count] + [3] * self.load_count, dtype=np.int64)
        )
        # 线路、母线和负荷使用 one-hot，另加入归一化时步和未来风电功率。
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
        """选择窗口并恢复到 ``t=0`` 初态。

        输入：可选随机种子；``options['window_id']`` 可固定测试窗口。
        输出：初始观测和空信息字典，符合 Gymnasium ``reset`` 接口。
        步骤：选定窗口，清零线路、母线和负荷状态，设置黑启动机组并计算初始出力。
        """

        super().reset(seed=seed)
        window_id = (options or {}).get("window_id")

        # 训练时随机抽样，测试和专家回放时使用指定窗口。
        if window_id is None:
            self.window_position = int(
                self.positions[self.np_random.integers(len(self.positions))]
            )
        else:
            self.window_position = self.position_by_window_id[int(window_id)]

        # 所有设备恢复到 t=0 的固定初始状态。
        # 初始仅黑启动母线通电，线路全断、负荷为 0 挡，黑启动机组在 t=0 启动。
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
            # 非法动作不更新状态，给固定-1奖励并截断当前episode。
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

        branch = int(action[0])
        requested_level = action[1:].astype(np.int8)

        # 根据动作前母线状态确定本步新通电母线。
        # 合法前沿线路恰有一端通电，因此另一端就是动作后新增的通电母线。
        left, right = self.branch_endpoints[branch]
        new_bus = int(right if self.bus_energized[left] else left)
        next_t = self.current_t + 1
        next_generation = self._available_generation(next_t)
        generation_total = float(next_generation.sum())

        # 计算动作后的负荷缺额和本步原始目标值。
        # max(0, 负荷-出力) 保证缺额不小于 0，因此不存在负缺额增加目标值。
        requested_load = float(np.dot(requested_level / 2.0, self.load_capacity))
        deficit = max(0.0, requested_load - generation_total)
        objective_step = requested_load - SHORTAGE_PENALTY * deficit

        # 写入线路、母线、机组和负荷的新状态。
        self.line_closed[branch] = True
        self.bus_energized[new_bus] = True
        newly_started = (self.generator_start_t < 0) & self.bus_energized[self.generator_bus]
        self.generator_start_t[newly_started] = next_t
        self.load_level = requested_level
        self.current_t = next_t
        self.generation = next_generation
        self.cumulative_objective += objective_step
        self.cumulative_energy_mwh += min(requested_load, generation_total) * self.interval_hours

        # 同时返回训练奖励和测试需要的物理指标。
        # reward 使用 objective_step/E'；info 保留原始目标及缺额，供最终评价汇总。
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
            # 只归一化训练奖励，不改变状态转移、动作约束和原始目标统计。
            float(objective_step / self.reference_energy[self.window_position]),
            self.current_t >= self.horizon,
            False,
            info,
        )

    def action_masks(self) -> np.ndarray:
        """生成 MaskablePPO 的动作掩码。

        输入：无，直接读取当前环境状态。
        输出：线路动作和每个负荷挡位动作组成的布尔数组。
        作用：屏蔽不满足恢复规则的动作，避免训练反复采样非法动作。
        步骤：筛选一端通电且未闭合的线路，再筛选已通电母线上的保持/升档动作。
        """

        endpoints = self.bus_energized[self.branch_endpoints]

        # 线路只能从当前已通电区域向外扩展。
        # 两端状态和为 1 表示恰有一端通电；已经闭合的线路同时被排除。
        branch_mask = (~self.line_closed) & (endpoints.sum(axis=1) == 1)

        # 负荷只能保持或升档，升档母线必须已经通电。
        # 最后一行赋值保证“保持当前挡位”始终合法，即使母线暂未通电。
        eligible = self.bus_energized[self.load_bus]
        levels = np.arange(3)[None, :]
        load_mask = (levels >= self.load_level[:, None]) & eligible[:, None]
        load_mask[np.arange(self.load_count), self.load_level] = True
        return np.concatenate((branch_mask, load_mask.reshape(-1)))

    def _check_action(self, action: np.ndarray) -> str | None:
        """检查动作空间格式和当前恢复约束。

        输入：长度为 ``1 + load_count`` 的联合整数动作。
        输出：动作合法时返回 ``None``，否则返回中文失败原因。
        步骤：先用动作空间检查长度和取值，再用当前掩码检查恢复约束。
        """

        # action_space 同时检查动作维度、线路编号和负荷挡位范围。
        if not self.action_space.contains(action):
            return "动作格式或取值超出动作空间"

        # 动作掩码检查线路前沿和负荷恢复状态约束。
        # 这不是重复审核：action_space 检查格式/范围，mask 检查当前状态下能否执行。
        branch = int(action[0])
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

        # 常规机组按启动后的爬坡时间计算最大可用出力。
        # 机组所在母线通电后记录启动时步，elapsed 决定当前最多可爬升多少 MW。
        generation = np.minimum(
            self.generator_capacity, self.generator_ramp * elapsed
        ) * started
        # 风电机组启动一个时步后使用窗口中的可用出力。
        # 风电不用常规机组爬坡值，而由当前窗口对应时步的可用功率覆盖。
        wind_started = started[self.is_wind] & (elapsed[self.is_wind] >= 1)
        if time_index > 0:
            wind_power = self.wind["current_power"][self.window_position, time_index - 1]
            generation[self.is_wind] = np.where(wind_started, wind_power, 0.0)
        else:
            generation[self.is_wind] = 0.0
        return generation

    def _observation(self) -> dict[str, np.ndarray]:
        """把当前恢复状态编码为策略网络的字典观测。

        输入：无，读取当前线路、母线、负荷、时步和未来风电状态。
        输出：包含五类定长 ``float32`` 数组的观测字典。
        作用：将离散恢复状态转换为 ``observation_space`` 规定的输入格式。
        步骤：对离散状态做 one-hot 编码，归一化时步，并拼接未来风电窗口。
        """

        future = np.minimum(
            self.current_t + np.arange(FUTURE_WIND_STEPS), self.horizon - 1
        )
        wind = self.wind["current_power"][self.window_position, future].astype(np.float32)

        # 离散状态使用 one-hot，连续状态统一为 float32。
        # 本函数只构造内存中的策略输入，不会把观测单独写入磁盘。
        return {
            "line_state": np.eye(2, dtype=np.float32)[self.line_closed.astype(int)].reshape(-1),
            "bus_state": np.eye(2, dtype=np.float32)[self.bus_energized.astype(int)].reshape(-1),
            "load_state": np.eye(3, dtype=np.float32)[self.load_level].reshape(-1),
            "time_fraction": np.asarray([self.current_t / self.horizon], dtype=np.float32),
            "future_wind_mw": wind.reshape(-1),
        }
