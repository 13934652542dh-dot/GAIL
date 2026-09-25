"""IEEE 39 节点系统恢复的 MaskablePPO 环境。"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# 观测中的完美预知风电长度；只影响策略可见信息，不改变38步恢复时域。
# 增大它会增加输入维度和网络计算量，减小它会减少可用的未来信息。
FUTURE_WIND_STEPS = 18
# PPO 奖励由归一化供电量、机组恢复、负荷恢复和约束惩罚组成。
# 这些系数只改变学习信号的相对权重，不改变环境状态转移或 MIP 专家目标。
REWARD = {
    "generator": 0.05,          # 新通电机组的恢复奖励
    "load": 0.10,               # 新恢复负荷的奖励
    "invalid_action": 0.05,     # 线路/负荷拓扑动作非法时的惩罚
    "deficit": 2.0,             # 供需缺额惩罚的权重
    "deficit_exponent": 5.0,    # 供需缺额惩罚的指数系数
}


class RestorationEnv(gym.Env):
    """在一个 38 步风电窗口内执行线路和负荷恢复。

    输入：系统电气参数字典、WT 出力窗口字典，以及使用的 train/test 数据类型。
    输出：符合 Gymnasium 接口并支持 MaskablePPO 动作掩码的恢复环境。
    """

    def __init__(
        self,
        system_data: dict,
        wind_data: dict,
        dataset_split: str = "train",
    ):
        """创建固定参数、动作空间和观测空间。

        输入：
            system_data：机组、负荷、线路和恢复时域参数；
            wind_data：全部 WT 出力窗口、窗口编号、数据类型和参考 E；
            dataset_split：当前环境使用 ``train`` 或 ``test`` 窗口。
        输出：无；初始化环境固定参数，不建立具体回合状态。
        步骤：筛选窗口，读取系统数组，再定义线路/负荷动作空间和状态观测空间。
        """

        super().__init__()

        # 1. 筛选当前训练或测试环境可以抽取的窗口。
        self.wind = wind_data
        self.positions = np.flatnonzero(
            wind_data["dataset_split"] == dataset_split
        )

        # 2. 提取状态转移所需的机组、负荷和线路参数。
        generators = system_data["generators"]
        loads = system_data["loads"]
        branches = system_data["branches"]

        self.generator_bus = generators["bus_index"]
        self.generator_capacity = generators["capacity_mw"]
        self.generator_ramp = generators["ramp_mw_per_step"]
        self.generator_initial_power = generators["initial_power_mw"]
        self.is_wind = generators["type"] == "WT"
        self.black_start_index = int(generators["black_start_index"])
        self.black_start_bus = int(self.generator_bus[self.black_start_index])

        self.load_bus = loads["bus_index"]
        self.load_capacity = loads["capacity_mw"]
        self.branch_endpoints = branches["endpoints"]

        self.bus_count = int(system_data["bus_count"])
        self.generator_count = len(self.generator_bus)
        self.load_count = len(self.load_bus)
        self.branch_count = len(self.branch_endpoints)
        self.total_load_mw = float(system_data["total_load_mw"])
        self.horizon = int(system_data["horizon"])
        self.interval_hours = float(system_data["interval_hours"])
        self.reference_energy_mwh = np.asarray(wind_data["E"], dtype=float)

        # 3. 定义一个线路动作、21 个负荷挡位动作以及字典观测空间。
        wind_capacity = self.generator_capacity[self.is_wind]
        self.action_space = spaces.MultiDiscrete(
            np.array([self.branch_count] + [3] * self.load_count, dtype=np.int64)
        )
        self.observation_space = spaces.Dict({
            "line_state": spaces.Box(
                0.0, 1.0, (2 * self.branch_count,), dtype=np.float32
            ),
            "bus_state": spaces.Box(
                0.0, 1.0, (2 * self.bus_count,), dtype=np.float32
            ),
            "load_state": spaces.Box(
                0.0, 1.0, (3 * self.load_count,), dtype=np.float32
            ),
            "time_fraction": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "future_wind_mw": spaces.Box(
                low=np.zeros(FUTURE_WIND_STEPS * len(wind_capacity), dtype=np.float32),
                high=np.tile(wind_capacity, FUTURE_WIND_STEPS).astype(np.float32),
                dtype=np.float32,
            ),
        })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """选择窗口并建立一个恢复回合的初始状态。

        输入：随机种子；``options`` 可通过 ``window_id`` 指定测试窗口。
        输出：初始观测字典和空的附加信息字典。
        步骤：选择窗口，初始化黑启动母线和机组，并清零线路、负荷、时间和累计电量。
        """

        super().reset(seed=seed)

        # 1. 训练时随机选窗口；测试时按 window_id 依次选择固定窗口。
        window_id = (options or {}).get("window_id")
        if window_id is None:
            self.window_position = int(
                self.positions[self.np_random.integers(len(self.positions))]
            )
        else:
            self.window_position = int(self.positions[
                self.wind["window_ids"][self.positions] == int(window_id)
            ][0])

        # 2. 建立 t=0 的黑启动状态。
        self.current_t = 0
        self.line_closed = np.zeros(self.branch_count, dtype=bool)
        self.bus_energized = np.zeros(self.bus_count, dtype=bool)
        self.bus_energized[self.black_start_bus] = True
        # -1 表示机组所在母线尚未通电；非负值表示母线首次通电的状态时步。
        self.generator_start_t = np.full(self.generator_count, -1, dtype=int)
        self.generator_start_t[self.black_start_index] = 0
        self.generator_started = self.generator_start_t >= 0
        self.load_level = np.zeros(self.load_count, dtype=np.int8)
        self.generation = self._generation_at(0)
        self.cumulative_energy_mwh = 0.0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        """执行一个恢复时步的线路和负荷联合动作。

        输入：22 维动作；第一个元素为线路编号，后 21 个元素为负荷目标挡位。
        输出：下一观测、当前奖励、终止标志、截断标志和测试结果信息。
        步骤：检查动作掩码，闭合一条前沿线路，按动作前母线状态检查负荷挡位，
        更新状态；若恢复负荷超过可用出力，不修改动作，而是计算缺额并在奖励中
        施加指数惩罚。
        """

        # 1. 拆分线路动作和负荷挡位动作。
        action = np.asarray(action, dtype=np.int64)
        branch = int(action[0])
        requested_level = action[1:].astype(np.int8)

        # 2. MaskablePPO 正常训练不会选择被掩码排除的动作。
        mask = self.action_masks()
        load_offset = self.branch_count + np.arange(self.load_count) * 3
        selected = np.concatenate((
            [mask[branch]],
            mask[load_offset + requested_level],
        ))
        if not selected.all():
            return self._invalid_action("动作违反线路前沿或负荷单调约束")

        # 3. 闭合选定的前沿线路，并计算下一时步的机组可用出力。
        left, right = self.branch_endpoints[branch]
        new_bus = int(right if self.bus_energized[left] else left)
        next_t = self.current_t + 1
        next_generation = self._generation_at(next_t)
        generation_total = float(next_generation.sum())

        next_bus_state = self.bus_energized.copy()
        next_bus_state[new_bus] = True
        # 4. 负荷约束对应修订稿 C4a-C4b：
        #    k(d,t+1) >= k(d,t)，且只有动作前已经通电的母线才能投入负荷。
        #    这里不再逐个“尽量接纳”请求，而是整组检查 C6 供需约束。
        ready_load = self.bus_energized[self.load_bus]
        if np.any(requested_level < self.load_level):
            return self._invalid_action("负荷挡位不能下降")
        if np.any((requested_level > 0) & ~ready_load):
            return self._invalid_action("负荷所在母线在动作前尚未通电")
        previous_load = self._load_mw(self.load_level)
        requested_load = self._load_mw(requested_level)

        # 5. 写入新的线路、母线、机组、负荷和时间状态。
        self.line_closed[branch] = True
        self.bus_energized = next_bus_state
        new_generators = self.generator_start_t < 0
        new_generators &= self.bus_energized[self.generator_bus]
        self.generator_start_t[new_generators] = next_t
        self.generator_started = self.generator_start_t >= 0
        self.generation = next_generation
        self.load_level = requested_level
        self.current_t = next_t

        # 6. PPO允许策略暂时请求超过出力的负荷；这不是拓扑非法动作。
        #    实际供电量按可用出力截断，未切除的负荷形成缺额并受到指数惩罚。
        connected_load = requested_load
        served_load = min(connected_load, generation_total)
        deficit = max(0.0, connected_load - generation_total)
        new_load = max(0.0, connected_load - previous_load)
        self.cumulative_energy_mwh += served_load * self.interval_hours
        reward = self._reward(
            served_load=served_load,
            deficit=deficit,
            new_generators=int(new_generators.sum()),
            new_load=new_load,
        )
        info = {
            "cumulative_energy_mwh": self.cumulative_energy_mwh,
            "restoration_rate": served_load / self.total_load_mw,
            "connected_load_mw": connected_load,
            "served_load_mw": served_load,
            "deficit_mw": deficit,
        }
        return (
            self._observation(),
            reward,
            self.current_t == self.horizon,
            False,
            info,
        )

    def action_masks(self) -> np.ndarray:
        """生成 MaskablePPO 使用的动作掩码。

        输入：无；使用当前线路、母线和负荷状态。
        输出：109 维布尔数组，包括 46 个线路动作和 21×3 个负荷挡位动作。
        步骤：保留一端通电的未闭合线路，并禁止负荷降档或在未通电区域升档。
        """

        # 线路只能连接一个已通电端和一个未通电端。
        endpoints = self.bus_energized[self.branch_endpoints]
        branch_mask = (~self.line_closed) & (endpoints.sum(axis=1) == 1)

        # C4b 使用动作前的 u(b,t)，所以新线路在本动作刚通电的母线不能同时投入负荷。
        eligible_load = self.bus_energized[self.load_bus]
        levels = np.arange(3)[None, :]
        load_mask = (levels >= self.load_level[:, None]) & eligible_load[:, None]
        load_mask[np.arange(self.load_count), self.load_level] = True
        return np.concatenate((branch_mask, load_mask.reshape(-1)))

    def _generation_at(self, time_index: int) -> np.ndarray:
        """计算指定恢复时步各机组的可用出力。

        输入：目标恢复时步 ``time_index``。
        输出：长度为机组数量的 MW 出力数组。
        步骤：传统机组按启动时长和爬坡率计算；风电机组读取当前窗口对应时步的出力。
        """

        # 传统机组按恢复后的经过时步爬坡，并受额定容量限制。
        elapsed = np.maximum(time_index - self.generator_start_t, 0)
        started = self.generator_start_t >= 0
        generation = np.minimum(
            self.generator_capacity,
            self.generator_initial_power + self.generator_ramp * elapsed,
        ) * started

        # 风电母线恢复一个时步后，直接使用对应窗口的可用出力。
        wind_active = started[self.is_wind] & (elapsed[self.is_wind] >= 1)
        generation[self.is_wind] = np.where(
            wind_active,
            self.wind["current_power"][self.window_position, time_index - 1],
            0.0,
        )
        return generation

    def _load_mw(self, level: np.ndarray) -> float:
        """将负荷挡位转换为总恢复负荷功率。

        输入：21 个负荷的 0/1/2 挡位数组。
        输出：对应 0%/50%/100% 挡位的总负荷功率，单位 MW。
        步骤：将挡位除以 2 后与各负荷容量做点积。
        """

        return float(np.dot(level / 2.0, self.load_capacity))

    def _reward(
        self,
        served_load: float,
        deficit: float,
        new_generators: int,
        new_load: float,
    ) -> float:
        """计算 PPO 当前时步的复合奖励。

        输入：
            served_load：实际供电负荷功率，单位 MW；
            deficit：请求负荷减去可用总出力的缺额，单位 MW；
            new_generators：本步新通电的机组数量；
            new_load：本步新增恢复负荷功率，单位 MW。
        输出：当前时步的标量奖励。
        步骤：先计算归一化供电量主奖励，再加入机组/负荷恢复奖励，
        最后对未切除的供需缺额施加指数惩罚。

        数学关系：
        1. 负荷请求功率：``P_conn(t)=sum_d 0.5*P_d^max*k(d,t)``；
        2. 可用总出力：``P_av(t)=sum_g p_av(g,t)``；
        3. 实际供电功率：``P_served(t)=min(P_conn(t),P_av(t))``；
        4. 供需缺额：``D(t)=max(0,P_conn(t)-P_av(t))``；
        5. 新增负荷：``Delta_P_load(t)=max(0,P_conn(t)-P_conn(t-1))``；
        6. 新通电机组数：
           ``N_new_g(t)=sum_g 1[u_(b(g),t)=1 and u_(b(g),t-1)=0]``；
        7. WT 特征：``B_i=sum_(t=1..38) sum_(w=1..6) P_WT_av(i,t,w)``；
        8. 1000 个专家窗口拟合参考值：``E_i^ref=alpha+beta*B_i``；
        9. 合法联合动作的总奖励：
           ``r_t = Delta_t*P_served(t)/E_i^ref``
           ``      + 0.05*N_new_g(t)/N_g``
           ``      + 0.10*Delta_P_load(t)/P_load_total``
           ``      - 2.0*(exp(5.0*D(t)/P_load_total)-1)``；
        10. 非法动作的奖励：``r_t=-0.05``，状态不更新并截断本回合；
            非法动作包括线路不是前沿线路、负荷挡位下降或未通电母线投入负荷。
        11. 一个完整回合的奖励：``R_episode=sum_(t=1..38) r_t``。

        因此“需要切负荷但仍保持过高负荷挡位”时，``D(t)>0``，缺额惩罚
        随缺额指数增长；供需缺额不是拓扑非法动作，不会触发 ``-0.05``。

        ``E_i^MIP`` 是修订稿中的优化目标；pkl 中的 ``E`` 是用 ``E(B)``
        拟合得到的归一化参考值，二者含义不同但单位都为 MWh。
        """

        reference_energy = self.reference_energy_mwh[self.window_position]
        reward = served_load * self.interval_hours / reference_energy
        reward += REWARD["generator"] * new_generators / self.generator_count
        reward += REWARD["load"] * new_load / self.total_load_mw
        if deficit > 0.0:
            reward -= REWARD["deficit"] * (
                np.exp(REWARD["deficit_exponent"] * deficit / self.total_load_mw)
                - 1.0
            )
        return float(reward)

    def _invalid_action(self, reason: str):
        """处理不满足 MDP 约束的联合动作。

        输入：说明违反哪一条拓扑、负荷或供需约束的文字。
        输出：当前观测、固定惩罚、未终止、已截断和简短信息字典。
        步骤：不改变状态，记录原因，并截断本回合；供需不足不在这里处理，
        而是在 ``_reward`` 中作为缺额指数惩罚处理。
        """

        current_load = self._load_mw(self.load_level)
        return self._observation(), -REWARD["invalid_action"], False, True, {
            "cumulative_energy_mwh": self.cumulative_energy_mwh,
            "restoration_rate": current_load / self.total_load_mw,
            "invalid_action": reason,
        }

    def _observation(self) -> dict[str, np.ndarray]:
        """构造 PPO 策略网络使用的状态观测。

        输入：无；读取当前线路、母线、负荷、时间和风电窗口状态。
        输出：包含线路、母线、负荷 one-hot 状态、时间比例和未来风电的字典。
        步骤：截取未来 18 步风电，窗口末端重复最后一步，再组装并展平各状态数组。
        """

        # 窗口末端不足 18 步时，重复第 38 步风电出力。
        future_indices = np.minimum(
            self.current_t + np.arange(FUTURE_WIND_STEPS),
            self.horizon - 1,
        )
        future_wind = self.wind["current_power"][
            self.window_position, future_indices
        ].astype(np.float32)
        return {
            "line_state": np.eye(2, dtype=np.float32)[
                self.line_closed.astype(int)
            ].reshape(-1),
            "bus_state": np.eye(2, dtype=np.float32)[
                self.bus_energized.astype(int)
            ].reshape(-1),
            "load_state": np.eye(3, dtype=np.float32)[self.load_level].reshape(-1),
            "time_fraction": np.array(
                [self.current_t / self.horizon], dtype=np.float32
            ),
            "future_wind_mw": future_wind.reshape(-1),
        }
