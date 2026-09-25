"""纯 PPO 的训练、定期评估和最终模型测试入口。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from 读取字典数据 import DEFAULT_PKL_PATH, load_experiment_data
from 恢复环境 import RestorationEnv


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "data" / "强化学习数据" / "纯PPO" / "训练"
TEST_DIR = PROJECT_DIR / "data" / "强化学习数据" / "纯PPO" / "测试"

TOTAL_TIMESTEPS = 300_000
# 回合每次最多38个动作；300000 是正式训练步数，不是专家求解样本数。
EVAL_FREQ = 3_000
# 评估回调的频率按单个子环境换算，375 次回调对应 8*375=3000 个总环境步。
EVAL_EPISODES = 30
NUM_ENVS = 8
ROLLOUT_STEPS = EVAL_FREQ // NUM_ENVS
# 每次 rollout 收集 375*8=3000 个样本，batch=1000 时恰好分成3个 minibatch。
# 减小 batch 会增加每轮梯度更新次数、计算开销和梯度噪声；增大 batch 会减少更新次数、
# 通常使梯度更平稳但占用更多内存。改动时应让 batch_size 不超过 rollout 样本数。
BATCH_SIZE = 1_000
SEED = 0
DEVICE = "cpu"


def train_ppo(pkl_path: Path = DEFAULT_PKL_PATH) -> Path:
    """训练纯 PPO，并保存最佳模型、最终模型和 TensorBoard 日志。

    输入：正式实验的 WT 出力数据 pkl 路径。
    输出：训练到 300000 步时保存的最终模型路径。
    步骤：读取数据，创建训练和评估环境，每 3000 步评估一次，最后保存模型和日志。
    """

    # 1. 读取系统参数和 train/test 风电窗口。
    experiment_data = load_experiment_data("all", pkl_path)
    system_data = experiment_data["system_data"]
    window_data = experiment_data["window_data"]

    # 2. 创建最佳模型和 TensorBoard 输出目录。
    best_model_dir = TRAIN_DIR / "最佳模型"
    tensorboard_dir = TRAIN_DIR / "TensorBoard"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # 3. 使用 8 个并行 train 环境采样，使用 1 个 test 环境定期评估。
    train_env = make_vec_env(
        RestorationEnv,
        n_envs=NUM_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "system_data": system_data,
            "wind_data": window_data,
            "dataset_split": "train",
        },
    )
    eval_env = make_vec_env(
        RestorationEnv,
        n_envs=1,
        seed=SEED + 1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "system_data": system_data,
            "wind_data": window_data,
            "dataset_split": "test",
        },
    )

    # 4. 每 3000 个总环境步评估一次，并保留平均奖励最高的模型。
    eval_callback = MaskableEvalCallback(
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        eval_freq=EVAL_FREQ // NUM_ENVS,
        best_model_save_path=str(best_model_dir),
        log_path=str(best_model_dir),
        deterministic=True,
    )
    # 5. 创建适用于字典观测和动作掩码的纯 PPO 模型。
    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        n_steps=ROLLOUT_STEPS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device=DEVICE,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
    )

    # 6. 训练 300000 步，并保存最后一个训练步对应的最终模型。
    final_model_path = TRAIN_DIR / "最终模型"
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            tb_log_name="纯PPO",
            progress_bar=True,
        )
        model.save(str(final_model_path))
    finally:
        train_env.close()
        eval_env.close()

    return final_model_path.with_suffix(".zip")


def test_ppo(
    model_path: Path,
    pkl_path: Path = DEFAULT_PKL_PATH,
) -> Path:
    """使用训练完成的最终模型依次测试全部 test 窗口。

    输入：最终 PPO 模型路径和正式实验的 WT 出力数据 pkl 路径。
    输出：测试汇总结果路径，同时保存每个 test 窗口的测试结果。
    步骤：加载模型，固定遍历全部 test 窗口，执行 38 步恢复并保存统计结果。
    """

    # 1. 读取系统参数并取得全部 test 窗口位置。
    experiment_data = load_experiment_data("all", pkl_path)
    system_data = experiment_data["system_data"]
    window_data = experiment_data["window_data"]
    test_positions = np.flatnonzero(window_data["dataset_split"] == "test")

    # 2. 加载最终模型并创建单个测试环境。
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    env = RestorationEnv(system_data, window_data, "test")
    model = MaskablePPO.load(str(model_path), device=DEVICE)

    # 3. 每个 test 窗口只测试一次，并使用确定性动作。
    results = []
    try:
        for position in test_positions:
            window_id = int(window_data["window_ids"][position])
            observation, _ = env.reset(options={"window_id": window_id})
            episode_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    action_masks=env.action_masks(),
                    deterministic=True,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
            results.append({
                "window_id": window_id,
                "episode_reward": episode_reward,
                "cumulative_energy_mwh": float(info["cumulative_energy_mwh"]),
                "terminal_restoration_rate": float(info["restoration_rate"]),
            })
    finally:
        env.close()

    # 4. 计算全部 test 窗口的平均结果并保存 JSON。
    rewards = np.asarray([item["episode_reward"] for item in results], dtype=float)
    energies = np.asarray([item["cumulative_energy_mwh"] for item in results], dtype=float)
    restoration_rates = np.asarray(
        [item["terminal_restoration_rate"] for item in results], dtype=float
    )
    summary = {
        "model_path": str(model_path),
        "test_count": len(results),
        "mean_episode_reward": float(rewards.mean()),
        "mean_cumulative_energy_mwh": float(energies.mean()),
        "mean_terminal_restoration_rate": float(restoration_rates.mean()),
    }

    summary_path = TEST_DIR / "测试汇总.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (TEST_DIR / "逐窗口测试结果.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary_path


def main() -> None:
    """运行完整的纯 PPO 实验。

    输入：无；使用代码中设置的默认正式数据路径和训练参数。
    输出：无；生成最佳模型、最终模型、TensorBoard 日志和 test 测试结果。
    步骤：先训练 300000 步，再使用最终模型测试全部 test 窗口并打印保存位置。
    """

    # 点击运行本文件时，依次完成训练和最终模型测试。
    final_model_path = train_ppo()
    summary_path = test_ppo(final_model_path)
    print(f"最终模型：{final_model_path}")
    print(f"测试结果：{summary_path}")


if __name__ == "__main__":
    main()
