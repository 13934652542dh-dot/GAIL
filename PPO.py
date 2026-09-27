"""纯 PPO：直接运行本文件，先训练，再统一测试。"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from 训练测试入口.训练工具 import (
    create_ppo,
    evaluate,
    load_index_ids,
    load_data,
    make_training_env,
    window_id_label,
)


OUTPUT_DIR = PROJECT_DIR / "data" / "强化学习数据" / "PPO"
DEFAULT_PKL_PATH = PROJECT_DIR / "data" / "正式实验输入" / "WT的出力数据.pkl"
PPO_TRAIN_INDEX_FILE = (
    PROJECT_DIR / "data" / "强化学习数据" / "固定窗口清单"
    / "PPO训练_17637-18136.json"
)
TEST_INDEX_FILE = (
    PROJECT_DIR / "data" / "强化学习数据" / "固定窗口清单"
    / "统一测试_8215-8714.json"
)
TRAIN_SEED = 0
# 纯PPO只使用这个专家范围拟合的E(B)作为奖励归一化基准。
REFERENCE_START = 0
REFERENCE_END = 1999
TOTAL_TIMESTEPS = 300_000
NUM_ENVS = 8
ROLLOUT_STEPS = 375
BATCH_SIZE = 1_000


def train(pkl_path: Path = DEFAULT_PKL_PATH) -> Path:
    """固定抽取 train 窗口并训练纯 PPO。

    输入：WT 出力 pkl 路径；系统参数从同目录读取。
    输出：训练完成的模型 zip 路径。
    作用：使用固定训练窗口和统一恢复环境训练基线 PPO。
    步骤：加载数据并保存窗口清单，创建向量环境，训练、保存模型并关闭环境。
    """

    # 三种 PPO 方法共用同一批、且与专家集不重叠的训练窗口。
    system, windows = load_data("train", REFERENCE_START, REFERENCE_END, pkl_path)
    window_ids = load_index_ids(PPO_TRAIN_INDEX_FILE)
    ppo_label = window_id_label(window_ids)

    # 文件名先记录E(B)专家范围，最后记录实际PPO训练窗口编号。
    train_dir = OUTPUT_DIR / "训练"
    model_name = (
        f"PPO_seed{TRAIN_SEED}_E参考{REFERENCE_START}-{REFERENCE_END}_PPO窗口{ppo_label}"
    )
    env = make_training_env(system, windows, window_ids, NUM_ENVS, TRAIN_SEED)
    model = create_ppo(
        env,
        train_dir / "TensorBoard" / model_name,
        TRAIN_SEED,
        ROLLOUT_STEPS,
        BATCH_SIZE,
    )
    # 精确编号同时写入模型本身；非连续窗口不只依赖文件名中的范围摘要。
    model.reward_reference_expert_range = [REFERENCE_START, REFERENCE_END]
    model.ppo_train_window_ids = window_ids.tolist()
    model_path = train_dir / model_name
    try:
        # 训练结束后保存可直接用于统一测试的最终模型。
        model.learn(TOTAL_TIMESTEPS, tb_log_name="PPO", progress_bar=True)
        model.save(str(model_path))
    finally:
        env.close()
    return model_path.with_suffix(".zip")


if __name__ == "__main__":
    # 入口固定执行训练和统一测试，不再增加只转发一次的main/test包装函数。
    model_path = train()
    result_path = evaluate(
        "PPO",
        model_path,
        OUTPUT_DIR / "测试",
        REFERENCE_START,
        REFERENCE_END,
        None,
        None,
        TEST_INDEX_FILE,
        PPO_TRAIN_INDEX_FILE,
        DEFAULT_PKL_PATH,
    )
    print(f"最终模型：{model_path}")
    print(f"测试结果：{result_path}")
