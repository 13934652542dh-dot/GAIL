"""数据预处理唯一入口。"""

from __future__ import annotations

from 提取与划分WT出力窗口 import (
    calculate_b, create_empty_e, extract_wt_windows, read_wt_output, split_windows,
)
from 构造WT出力字典并存储 import (
    SYSTEM_OUTPUT_FILE, WT_OUTPUT_FILE, build_wt_split_dictionary,
    read_system_parameters, save_dictionary,
)


def main() -> None:
    """按实验顺序生成正式实验所需的两个 pkl 文件。

    输入：固定的六台风机 CSV 和基本电气参数 Excel。
    输出：``WT的出力数据.pkl`` 与 ``系统电气参数.pkl``。

    主要步骤：
    1. 读取并排序连续风机出力；
    2. 按固定长度和步长提取全部窗口，再按时间顺序划分 train/test；
    3. 计算每个窗口的 B，并为尚未拟合的 E 建立 NaN 占位；
    4. 保存风机窗口数据和电气系统参数。
    """

    # 1. 读取固定 CSV。
    wt_table = read_wt_output()

    # 2. 每隔 1 个时步提取一个长度为 38 个时步的窗口。
    all_power = extract_wt_windows(wt_table)

    # 3. 按窗口时间顺序将全部窗口划分为 70% train 和 30% test。
    train_power, test_power = split_windows(all_power)

    # 4. 计算 B；B 是 38 步、6 台风机出力的总和。
    train_b = calculate_b(train_power)
    test_b = calculate_b(test_power)

    # 5. E 必须等待专家求解，当前只建立 NaN 占位，不填充任何数值。
    train_e = create_empty_e(len(train_power))
    test_e = create_empty_e(len(test_power))

    # 6. train/test 顶层键已经表达数据划分，不重复保存样本类型和编号。
    wt_data = {
        "train": build_wt_split_dictionary(train_power, train_b, train_e),
        "test": build_wt_split_dictionary(test_power, test_b, test_e),
    }

    # 7. 读取基本电气参数并分别保存两个字典。
    system_data = read_system_parameters()
    save_dictionary(wt_data, WT_OUTPUT_FILE)
    save_dictionary(system_data, SYSTEM_OUTPUT_FILE)


if __name__ == "__main__":
    main()
