"""生成六台风机连续 15 分钟仿真可用出力。

本脚本负责从ERCOT原始ZIP中提取六台已经确定的风电机组，
将原始HSL处理成连续15分钟时间序列，再映射到仿真系统的机组容量。

处理顺序：
1. 扫描ZIP中的Gen_Resource_Data CSV；
2. 只保留时间戳、机组名称和HSL三类必要数据；
3. 将原始时间戳四舍五入到最近15分钟，并处理同一目标时刻的重复记录；
4. 在HSL层面对共同15分钟时间轴上的缺失点做前后点线性插值；
5. 使用HSL / 原始机组容量 × 仿真机组容量得到仿真MW出力；
6. 输出连续仿真出力CSV。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd


# -------------------------
# 1. 文件路径和资源映射
# -------------------------
# PROJECT_DIR：当前代码位于“数据处理代码”子目录，向上一级回到项目目录。
PROJECT_DIR = Path(__file__).resolve().parent.parent
# RAW_DATA_DIR：原始ERCOT ZIP文件的读取路径，只读不写。
RAW_DATA_DIR = Path(r"E:\GAIL_renewable_data\ercot\raw")
# OUTPUT_DIR：固定输入数据目录，只保存连续出力CSV和基本电气参数。
OUTPUT_DIR = PROJECT_DIR / "data" / "处理过程"
# OUTPUT_FILE：第一阶段生成的最终连续15分钟仿真MW出力，供下一段代码读取。
OUTPUT_FILE = OUTPUT_DIR / "六台风机15分钟仿真可用出力.csv"

# 原始容量用于把HSL转换为原始机组容量比例；仿真容量用于得到仿真系统MW出力。
# 这六台机组来自已确定的资源映射，本脚本不重新抽样、不改变映射关系。
RESOURCE_MAPPING = [
    {
        "simulation_unit": "WT1",
        "original_unit": "SRWE1_UNIT1",
        "simulation_capacity_mw": 646.0,
        "original_capacity_mw": 213.82,
    },
    {
        "simulation_unit": "WT2",
        "original_unit": "SSPURTWO_WIND_1",
        "simulation_capacity_mw": 725.0,
        "original_capacity_mw": 160.95,
    },
    {
        "simulation_unit": "WT3",
        "original_unit": "CHALUPA_UNIT1",
        "simulation_capacity_mw": 508.0,
        "original_capacity_mw": 173.25,
    },
    {
        "simulation_unit": "WT4",
        "original_unit": "CABEZON_WIND2",
        "simulation_capacity_mw": 687.0,
        "original_capacity_mw": 122.40,
    },
    {
        "simulation_unit": "WT5",
        "original_unit": "AQUILLA_U2_28",
        "simulation_capacity_mw": 564.0,
        "original_capacity_mw": 143.80,
    },
    {
        "simulation_unit": "WT6",
        "original_unit": "AJAXWIND_UNIT1",
        "simulation_capacity_mw": 865.0,
        "original_capacity_mw": 225.60,
    },
]
ORIGINAL_UNIT_NAMES = {item["original_unit"] for item in RESOURCE_MAPPING}


def read_zip_raw_data(data_dir: Path) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """从 ERCOT ZIP 文件读取六台指定机组的原始数据。

    输入：
        data_dir：存放ERCOT原始ZIP文件的目录。

    输出：
        raw_records：字典，键为原始机组名称，值为
        ``(SCED时间戳, HSL)``二元组列表。

    作用：只提取后续计算需要的时间戳、机组名称和 HSL，不写入文件。

    主要步骤：
    1. 遍历目录中的 ZIP 文件及其 Gen_Resource_Data CSV；
    2. 筛选六台目标机组和三个必要字段；
    3. 清理无效时间戳、缺失值和非数值 HSL；
    4. 按机组归类记录、去除完全重复项并按时间排序。
    """
    raw_records = {unit_name: [] for unit_name in ORIGINAL_UNIT_NAMES}
    # 外层循环：遍历原始数据目录中的每一个ZIP，目的是覆盖全部下载的源数据。
    # sorted保证每次运行的处理顺序稳定，便于复现和检查。
    for zip_path in sorted(data_dir.rglob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            # 内层循环：遍历当前ZIP中的所有成员文件，只挑选Gen_Resource_Data表。
            # ZIP里可能有很多无关文件，因此这里是“打开有用表格”的筛选位置。
            for member in archive.infolist():
                member_name = member.filename.replace("\\", "/")
                file_name = member_name.rsplit("/", 1)[-1].lower()
                if "gen_resource_data" not in file_name or not file_name.endswith(".csv"):
                    continue

                raw_bytes = archive.read(member)
                table = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
                required_columns = {"SCED Time Stamp", "Resource Name", "HSL"}
                if not required_columns.issubset(table.columns):
                    continue

                table = table.loc[
                    table["Resource Name"].isin(ORIGINAL_UNIT_NAMES),
                    ["SCED Time Stamp", "Resource Name", "HSL"],
                ].copy()
                table["SCED Time Stamp"] = pd.to_datetime(
                    table["SCED Time Stamp"], errors="coerce"
                )
                table["HSL"] = pd.to_numeric(table["HSL"], errors="coerce")
                table = table.dropna(subset=["SCED Time Stamp", "HSL"])

                # 逐行循环：把筛选后的三列原始数据放入按机组分类的字典，
                # 后续每台机组会独立进行时间四舍五入、去重和插值。
                for timestamp, unit_name, hsl in table.itertuples(index=False, name=None):
                    raw_records[unit_name].append((timestamp, float(hsl)))

    # 逐机组循环：删除跨ZIP文件完全重复的记录，并按时间排序，
    # 使每台机组形成一个可用于后续连续化的原始时间序列。
    for unit_name in ORIGINAL_UNIT_NAMES:
        raw_records[unit_name] = sorted(set(raw_records[unit_name]), key=lambda item: item[0])
        if not raw_records[unit_name]:
            raise ValueError(f"没有读取到机组 {unit_name} 的有效原始数据")

    return raw_records


def round_and_deduplicate(
    raw_records: list[tuple[pd.Timestamp, float]],
) -> pd.DataFrame:
    """整理单台机组的 15 分钟时间点。

    输入：
        raw_records：单台机组的原始``(时间戳, HSL)``列表。

    输出：
        DataFrame，包含“目标时间”和“HSL（MW）”两列，每个目标时间唯一。

    作用：只对时间戳对齐到最近 15 分钟；同一目标时刻只保留距离最近的记录。
    这里的四舍五入是时间轴对齐，不是对风机 MW 出力数值做四舍五入。

    主要步骤：计算每条记录的目标时刻和时间距离，按目标时刻、距离和原始
    时间排序，再对目标时刻去重。
    """
    table = pd.DataFrame(raw_records, columns=["原始时间", "HSL（MW）"])
    table["目标时间"] = table["原始时间"].dt.round("15min")
    table["距离秒"] = (
        table["原始时间"] - table["目标时间"]
    ).abs().dt.total_seconds()
    table = table.sort_values(["目标时间", "距离秒", "原始时间"])
    table = table.drop_duplicates(subset=["目标时间"], keep="first")
    return table[["目标时间", "HSL（MW）"]].sort_values("目标时间").reset_index(drop=True)


def build_continuous_simulation_output(
    raw_records: dict[str, list[tuple[pd.Timestamp, float]]],
) -> pd.DataFrame:
    """生成六台风机共同时间轴上的连续仿真出力。

    输入：
        raw_records：read_zip_raw_data返回的六台原始机组记录字典。

    输出：包含时间戳和WT1到WT6仿真MW出力的连续宽表。

    作用：建立共同 15 分钟时间轴、补齐内部缺失点，并按容量比例映射为仿真 MW。

    主要步骤：
    1. 对每台机组的原始数据进行时间对齐和去重；
    2. 取六台机组共同覆盖的时间范围，建立 15 分钟时间轴；
    3. 对内部缺失 HSL 做基于时间的线性插值；
    4. 按容量比例转换为仿真出力，并限制在 0 到仿真容量之间。
    """
    anchor_tables = {
        item["original_unit"]: round_and_deduplicate(raw_records[item["original_unit"]])
        for item in RESOURCE_MAPPING
    }

    # 共同边界避免某一台机组的起止时间造成其他机组的单边外推。
    start_time = max(table["目标时间"].min() for table in anchor_tables.values())
    end_time = min(table["目标时间"].max() for table in anchor_tables.values())
    target_time_index = pd.date_range(start_time, end_time, freq="15min")
    if len(target_time_index) == 0:
        raise ValueError("六台机组没有共同的15分钟有效时间范围")

    continuous_hsl = pd.DataFrame({"时间戳": target_time_index})
    # 逐机组循环：每台机组都经历“对齐时间轴→插值HSL→容量映射”三步，
    # 最终写入连续输出表中的一列仿真MW出力。
    for item in RESOURCE_MAPPING:
        original_unit = item["original_unit"]
        simulation_unit = item["simulation_unit"]
        anchor = anchor_tables[original_unit].set_index("目标时间")["HSL（MW）"]
        aligned_hsl = anchor.reindex(target_time_index)
        continuous_hsl_value = aligned_hsl.interpolate(method="time", limit_area="inside")
        if continuous_hsl_value.isna().any():
            raise ValueError(f"{original_unit}存在无法用前后点插值补全的端点缺失")

        original_capacity = float(item["original_capacity_mw"])
        simulation_capacity = float(item["simulation_capacity_mw"])
        # 使用容量映射公式，将原始机组 HSL 按容量比例转换为仿真 MW 出力。
        simulation_output = (
            continuous_hsl_value / original_capacity * simulation_capacity
        ).clip(lower=0.0, upper=simulation_capacity)
        continuous_hsl[f"{simulation_unit}仿真可用出力（MW）"] = simulation_output.to_numpy()

    return continuous_hsl


def save_wt_output(continuous_output: pd.DataFrame, output_file: Path = OUTPUT_FILE) -> None:
    """保存连续风机出力。

    输入：包含时间戳和六台风机出力的 DataFrame，以及输出 CSV 路径。
    输出：无；写出六台风机 15 分钟仿真可用出力 CSV。

    主要步骤：创建输出目录，并将 DataFrame 原样写出；不再对出力数值
    做两位小数格式化，避免保存阶段损失精度。
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)
    continuous_output.to_csv(output_file, index=False, encoding="utf-8-sig")


def main() -> None:
    """从 ZIP 原始数据生成六台风机连续 15 分钟 CSV。

    输入：``RAW_DATA_DIR`` 下的 ERCOT ZIP 文件。
    输出：``OUTPUT_FILE``，包含时间戳和六台风机仿真可用出力。
    流程：读取原始数据、整理时间点、建立连续出力、写出 CSV。

    当前机器没有原始 ZIP 时不要运行本入口；后续只需运行本函数即可生成 CSV。
    """

    raw_records = read_zip_raw_data(RAW_DATA_DIR)
    continuous_output = build_continuous_simulation_output(raw_records)
    save_wt_output(continuous_output, OUTPUT_FILE)


if __name__ == "__main__":
    main()
