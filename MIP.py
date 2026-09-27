"""按照《恢复问题数学模型_终稿.md》建立并求解恢复 MILP。"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB


# ======================================================================
# 文件定位与实验口径
# ======================================================================
# 1. 本文件只负责“求解一个风电窗口”，不负责选择专家样本，也不写结果文件。
# 2. 批量实验由 solve_MIP.py 逐窗口调用 build_model，并在那里统一保存专家数据。
# 3. system 是固定电网参数；windows 通常只含一个窗口，所以批量入口传入 position=0。
# 4. 一个窗口包含 38 个恢复动作：每步闭合一条线路，同时确定全部负荷挡位。
# 5. 返回的线路顺序和负荷轨迹既用于环境回放，也作为 BC/GAIL 的专家示范。
# 6. 本文件属于专家求解口径；后续强化学习训练不会修改这里的模型。
HORIZON = 38
SHORTAGE_PENALTY = 5.0


def build_model(
    system: dict,
    windows: dict,
    position: int,
) -> tuple[list[int], list[list[int]], float] | None:
    """建立并求解一个风电窗口的 MIP 模型。

    输入：系统参数、风电窗口数据、窗口位置。
    输出：有可行解时返回 38 步线路闭合顺序、逐步负荷挡位和目标值；
    无可行解时返回 None。轨迹第 t 行对应动作 t 和动作后的负荷状态 t+1。
    """

    # ==================================================================
    # 1. 集合与参数
    # ==================================================================
    # system 保存不随窗口变化的电网参数；windows 保存当前窗口的风电时序。
    # wind_power 第一维对应 38 个动作时步，第二维对应各台风电机组。
    generators = system["generators"]
    loads = system["loads"]
    endpoints = np.asarray(system["branches"]["endpoints"], dtype=int)
    gen_bus = np.asarray(generators["bus_index"], dtype=int)                 # b(g)
    gen_capacity = np.asarray(generators["capacity_mw"], dtype=float)       # P_g^max
    gen_ramp = np.asarray(generators["ramp_mw_per_step"], dtype=float)      # R_g
    gen_type = np.asarray(generators["type"])
    load_bus = np.asarray(loads["bus_index"], dtype=int)                    # b(d)
    load_capacity = np.asarray(loads["capacity_mw"], dtype=float)           # P_d^max
    wind_power = np.asarray(windows["current_power"][position], dtype=float)

    B = list(range(int(system["bus_count"])))       # 母线集合 B
    L = list(range(len(endpoints)))                  # 线路集合 L
    G = list(range(len(gen_bus)))                    # 机组集合 G
    D = list(range(len(load_bus)))                   # 负荷集合 D
    G_conv = [generator for generator in G if gen_type[generator] != "WT"]   # 常规机组集合 G_conv
    G_WT = [generator for generator in G if gen_type[generator] == "WT"]     # 风电机组集合 G_WT
    T = list(range(HORIZON + 1))                     # 状态时步 t=0,...,H
    T_action = list(range(HORIZON))                  # 动作时步 t=0,...,H-1
    T_after = list(range(1, HORIZON + 1))            # 动作后的状态 t=1,...,H
    delta_L = {bus: [line for line in L if bus in endpoints[line]] for bus in B}  # 母线关联线路集合 δ_L(b)

    black_bus = int(gen_bus[int(generators["black_start_index"])])          # b_BS
    wind_position = {generator: index for index, generator in enumerate(G_WT)}

    # ==================================================================
    # 2. 决策变量
    # ==================================================================
    # 状态变量使用 t=0,...,38；线路动作只在 t=0,...,37 发生。
    # 因而动作 t 执行后得到状态 t+1，最终提取 38 行负荷轨迹。
    model = gp.Model("restoration_revision")
    bus_on = model.addVars(B, T, vtype=GRB.BINARY, name="u")
    line_action = model.addVars(L, T_action, vtype=GRB.BINARY, name="y")
    load_level = model.addVars(D, T, vtype=GRB.INTEGER, lb=0, ub=2, name="k")
    generation = model.addVars(G, T, vtype=GRB.CONTINUOUS, lb=0.0, name="p")
    shortage = model.addVars(T_action, lb=0.0, name="shortage")

    # u[b,t]：母线 b 在状态 t 是否通电，0/1 变量。
    # y[l,t]：动作时步 t 是否闭合线路 l，0/1 变量。
    # k[d,t]：负荷 d 的挡位，0/1/2 对应 0%/50%/100%。
    # p[g,t]：机组 g 的有功出力，单位 MW。
    # shortage[t-1]：数学模型中的 s_t，状态 t 的负荷缺额。
    # shortage 创建时 lb=0，因此缺额变量不可能小于 0；目标中的 -5s 只会扣分。

    # ==================================================================
    # 3. C0 初始状态约束
    # ==================================================================
    # C0a：初始时仅黑启动母线通电。
    for bus in B:
        model.addConstr(bus_on[bus, 0] == int(bus == black_bus), name=f"C0a_{bus}")

    # C0b：初始时所有机组出力为零。
    for generator in G:
        model.addConstr(generation[generator, 0] == 0.0, name=f"C0b_{generator}")

    # C0c：初始时所有负荷均未投入。
    for load in D:
        model.addConstr(load_level[load, 0] == 0, name=f"C0c_{load}")

    # ==================================================================
    # 4. C1-C2 线路动作约束
    # ==================================================================
    for step in T_action:
        # C1：每个动作时步恰好闭合一条线路。
        model.addConstr(gp.quicksum(line_action[line, step] for line in L) == 1, name=f"C1_{step}")

        for line in L:
            left, right = endpoints[line]
            # C2a：所选线路不能连接两个未通电母线。
            model.addConstr(line_action[line, step] <= bus_on[int(left), step] + bus_on[int(right), step], name=f"C2a_{line}_{step}")
            # C2b：所选线路不能连接两个已通电母线，避免形成环路。
            model.addConstr(line_action[line, step] <= 2 - bus_on[int(left), step] - bus_on[int(right), step], name=f"C2b_{line}_{step}")

    # ==================================================================
    # 5. C3 母线通电约束
    # ==================================================================
    for step in T_after:
        previous_step = step - 1

        for bus in B:
            # C3a：母线通电状态不可回退。
            model.addConstr(bus_on[bus, step] >= bus_on[bus, previous_step], name=f"C3a_{bus}_{step}")
            # 辅助约束：没有闭合关联线路时，未通电母线不能自行通电。
            model.addConstr(bus_on[bus, step] <= bus_on[bus, previous_step] + gp.quicksum(line_action[line, previous_step] for line in delta_L[bus]), name=f"C3_link_{bus}_{step}")

        for line in L:
            left, right = endpoints[line]
            # C3b：闭合线路的两个端点在下一状态通电。
            model.addConstr(bus_on[int(left), step] >= line_action[line, previous_step], name=f"C3b_left_{line}_{step}")
            model.addConstr(bus_on[int(right), step] >= line_action[line, previous_step], name=f"C3b_right_{line}_{step}")

        # ==============================================================
        # 6. C4 负荷恢复约束
        # ==============================================================
        for load in D:
            # C4a：负荷挡位不可回退。
            model.addConstr(load_level[load, step] >= load_level[load, previous_step], name=f"C4a_{load}_{step}")
            # C4b：负荷所在母线在前一状态已通电，负荷才可投入。
            model.addConstr(load_level[load, step] <= 2 * bus_on[int(load_bus[load]), previous_step], name=f"C4b_{load}_{step}")

        # ==============================================================
        # 7. C5 机组出力与供需约束
        # ==============================================================
        # C5a：常规机组出力上限及母线通电延迟。
        for generator in G_conv:
            model.addConstr(generation[generator, step] <= gen_capacity[generator] * bus_on[int(gen_bus[generator]), previous_step], name=f"C5a_conv_{generator}_{step}")

        # C5a：风电机组可用出力上限及母线通电延迟。
        for generator in G_WT:
            model.addConstr(generation[generator, step] <= wind_power[previous_step, wind_position[generator]] * bus_on[int(gen_bus[generator]), previous_step], name=f"C5a_WT_{generator}_{step}")

        # C5b：常规机组向上爬坡量不超过 R_g。
        for generator in G_conv:
            model.addConstr(generation[generator, step] - generation[generator, previous_step] <= gen_ramp[generator], name=f"C5b_{generator}_{step}")

        # C4c：P_L(t)=Σ_d(P_d^max/2)k_d,t。
        load_power = gp.quicksum(0.5 * load_capacity[load] * load_level[load, step] for load in D)
        generation_power = gp.quicksum(generation[generator, step] for generator in G)

        # C5c：s_t >= P_L(t)-Σ_g p_g,t；目标惩罚 s_t，因此最优时取实际缺额。
        model.addConstr(shortage[previous_step] >= load_power - generation_power, name=f"C5c_{step}")

    # ==================================================================
    # 8. 目标函数
    # ==================================================================
    # max J=Σ_t[P_L(t)-5s_t]。
    # P_L 越大越好，缺额 s 越大惩罚越重。由于 s>=0，不存在“负缺额增加目标值”。
    # ObjVal 是 38 步原始目标之和，不是强化学习中除以 E' 后的归一化奖励。
    restored_load = gp.quicksum(0.5 * load_capacity[load] * load_level[load, step] for load in D for step in T_after)
    model.setObjective(restored_load - SHORTAGE_PENALTY * gp.quicksum(shortage[step] for step in T_action), GRB.MAXIMIZE)

    # ==================================================================
    # 9. 求解
    # ==================================================================
    # TimeLimit、MIPGap 和 Threads 由 solve_MIP.py 在批量求解前统一设置，
    # 从而保证一批专家窗口使用同一套 Gurobi 参数。
    # ===== 求解 =====
    model.optimize()

    # ===== 提取结果 =====
    if model.SolCount == 0:
        # 返回 None 后，solve_MIP.py 仍会为该窗口保存一条无可用解记录。
        return None

    # order：长度 HORIZON 的列表，order[t] 是动作时步 t 闭合的线路编号。
    order = [None] * HORIZON
    for step in T_action:
        for line in L:
            if round(line_action[line, step].X) == 1:
                order[step] = line
                break

    # load_trajectory：第 t 行是动作后状态 t+1 的全部负荷挡位。load_trajectory[t][d]：第t步动作完成后，负荷d的挡位
    # 下游会把 order[t] 与 load_trajectory[t] 横向拼成一条完整专家动作。
    load_trajectory = []
    for step in T_after:
        load_trajectory.append([round(load_level[load, step].X) for load in D])

    # 本函数只把结果交给调用方；实际 pkl 文件由 solve_MIP.py 创建。
    return order, load_trajectory, model.ObjVal
