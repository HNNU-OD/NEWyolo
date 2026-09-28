"""Read-only source conversion and merge into a new flat YOLO dataset."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
import yaml
from PIL import Image

OUTPUT = Path(__file__).resolve().parent
ORIGINAL = Path(r"D:\data\9.22\insulator_yolo")
IDD = Path(r"D:\data\open_insulator_datasets\Insulator_Defect_Detection\project")
VPMBGI = Path(r"D:\data\open_insulator_datasets\VPMBGI\dataset")
OLD_NAMES = ["污秽较为严重，但表面无明显放电", "破损", "污秽（放电痕迹）", "固定不牢固（倾斜）", "釉表面脱落"]
EXTRA_NAMES = ["绝缘子本体（状态未区分）", "破损缺陷（局部框）", "污闪缺陷（局部框）", "玻璃绝缘子破损（局部框）"]
SPLITS = ("train", "val", "test")


def hash_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def snapshot(roots):
    return {str(path): {"bytes": path.stat().st_size, "sha256": hash_file(path)}
            for root in roots for path in sorted(root.rglob("*")) if path.is_file()}


def txt_boxes(path, mapping):
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        assert len(values) == 5, (path, line)
        source_id = int(values[0])
        boxes.append((mapping[source_id], *map(float, values[1:])))
    return boxes


def read_records(include_original):
    records = []
    offset = 5 if include_original else 0
    if include_original:
        config = yaml.safe_load((ORIGINAL / "data.yaml").read_text(encoding="utf-8"))
        assert list(config["names"].values()) == OLD_NAMES
        for split in SPLITS:
            for line in (ORIGINAL / config[split]).read_text(encoding="utf-8-sig").splitlines():
                image = (ORIGINAL / line).resolve()
                label = ORIGINAL / "labels" / (image.stem + ".txt")
                records.append({"source": "original", "split": split, "image": image, "annotation": label,
                                "boxes": txt_boxes(label, {i: i for i in range(5)})})
    mapping = {"insulator": offset, "broken": offset + 1, "pollution-flashover": offset + 2}
    for split in ("train", "val"):
        for label in sorted((IDD / split / "ann").glob("*.json")):
            data = json.loads(label.read_text(encoding="utf-8"))
            width, height = data["size"]["width"], data["size"]["height"]
            assert width > 0 and height > 0
            image = IDD / split / "img" / label.name.removesuffix(".json")
            boxes = []
            for obj in data["objects"]:
                assert obj["geometryType"] == "rectangle"
                (x1, y1), (x2, y2) = obj["points"]["exterior"]
                assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, label
                boxes.append((mapping[obj["classTitle"]], (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height))
            records.append({"source": "idd", "split": split, "image": image, "annotation": label,
                            "boxes": boxes, "expected_size": (width, height)})
    for source_split, split in (("train", "train"), ("valid", "val")):
        for label in sorted((VPMBGI / source_split / "labels").glob("*.txt")):
            images = list((VPMBGI / source_split / "images").glob(label.stem + ".*"))
            assert len(images) == 1, label
            records.append({"source": "vpmbgi", "split": split, "image": images[0], "annotation": label,
                            "boxes": txt_boxes(label, {0: offset + 3})})
    return records


def verify_conversion(record, actual, offset):
    """Compare final TXT against source geometry independently of serialization."""
    if record["source"] == "idd":
        source = json.loads(record["annotation"].read_text(encoding="utf-8"))
        mapping = {"insulator": offset, "broken": offset + 1, "pollution-flashover": offset + 2}
        width, height = source["size"]["width"], source["size"]["height"]
        assert len(source["objects"]) == len(actual)
        for obj, values in zip(source["objects"], actual):
            cls, x, y, w, h = values
            assert cls == mapping[obj["classTitle"]]
            recovered = ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)
            expected = obj["points"]["exterior"][0] + obj["points"]["exterior"][1]
            assert all(abs(a - b) < 1e-5 for a, b in zip(recovered, expected))
    else:
        lines = record["annotation"].read_text(encoding="utf-8-sig").splitlines()
        assert len(lines) == len(actual)
        for line, values in zip(lines, actual):
            fields = line.split()
            expected_class = int(fields[0]) if record["source"] == "original" else offset + 3
            assert values[0] == expected_class
            assert all(abs(a - float(b)) <= 1e-9 for a, b in zip(values[1:], fields[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("all", "external"), default="all")
    args = parser.parse_args()
    include_original = args.scope == "all"
    assert not (OUTPUT / "images").exists(), "Output dataset already exists; refusing to overwrite"
    names = ([name + "（整体框）" for name in OLD_NAMES] if include_original else []) + EXTRA_NAMES
    offset = 5 if include_original else 0
    roots = ([ORIGINAL] if include_original else []) + [IDD, VPMBGI]
    print("Hashing source files before conversion...", flush=True)
    before = snapshot(roots)
    records = read_records(include_original)
    expected_counts = {"original": 920, "idd": 1440, "vpmbgi": 1010} if include_original else {"idd": 1440, "vpmbgi": 1010}
    assert dict(Counter(r["source"] for r in records)) == expected_counts
    hashes = {}
    for record in records:
        image = record["image"]
        assert image.is_file(), image
        fingerprint = before[str(image)]["sha256"]
        if fingerprint in hashes:
            assert hashes[fingerprint] == record["split"], "Identical image spans source splits; manual resolution required"
        hashes[fingerprint] = record["split"]
        record["output_name"] = record["source"] + "_" + image.name
    assert len({r["output_name"] for r in records}) == len(records)
    (OUTPUT / "images").mkdir()
    (OUTPUT / "labels").mkdir()
    manifests, image_lists = [], {split: [] for split in SPLITS}
    stats = {split: {"images": 0, "boxes": 0, "class_boxes": Counter()} for split in SPLITS}
    for index, record in enumerate(records, 1):
        image = record["image"]
        with Image.open(image) as picture:
            picture.load()
            if "expected_size" in record:
                assert picture.size == record["expected_size"], image
        dest_image = OUTPUT / "images" / record["output_name"]
        shutil.copy2(image, dest_image)
        assert hash_file(dest_image) == before[str(image)]["sha256"]
        label = OUTPUT / "labels" / (dest_image.stem + ".txt")
        lines = [str(box[0]) + " " + " ".join(f"{value:.10f}" for value in box[1:]) for box in record["boxes"]]
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        actual = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            cls, x, y, w, h = int(parts[0]), *map(float, parts[1:])
            assert 0 <= cls < len(names)
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) and w > 0 and h > 0
            assert x - w / 2 >= -1e-5 and y - h / 2 >= -1e-5 and x + w / 2 <= 1 + 1e-5 and y + h / 2 <= 1 + 1e-5
            actual.append((cls, x, y, w, h))
            stats[record["split"]]["class_boxes"][cls] += 1
        verify_conversion(record, actual, offset)
        stats[record["split"]]["images"] += 1
        stats[record["split"]]["boxes"] += len(actual)
        image_lists[record["split"]].append("./images/" + dest_image.name)
        manifests.append([dest_image.name, label.name, record["source"], record["split"], str(image), str(record["annotation"]), len(actual), before[str(image)]["sha256"]])
        if index % 250 == 0:
            print(f"Converted, copied and verified {index}/{len(records)} images", flush=True)
    for split, paths in image_lists.items():
        (OUTPUT / (split + ".txt")).write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    config = {"path": OUTPUT.as_posix(), "train": "train.txt", "val": "val.txt"}
    if image_lists["test"]:
        config["test"] = "test.txt"
    config.update({"nc": len(names), "names": dict(enumerate(names))})
    (OUTPUT / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (OUTPUT / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    with (OUTPUT / "source_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "label", "source", "split", "original_image", "original_annotation", "boxes", "image_sha256"])
        writer.writerows(manifests)
    totals = Counter()
    for values in stats.values():
        totals.update(values["class_boxes"])
    with (OUTPUT / "class_statistics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "class", "total_boxes", "train_boxes", "val_boxes", "test_boxes"])
        for cls, name in enumerate(names):
            writer.writerow([cls, name, totals[cls]] + [stats[s]["class_boxes"][cls] for s in SPLITS])
    assert sum(totals.values()) == (9734 if include_original else 6019)
    assert {p.stem for p in (OUTPUT / "images").iterdir()} == {p.stem for p in (OUTPUT / "labels").iterdir()}
    assert yaml.safe_load((OUTPUT / "data.yaml").read_text(encoding="utf-8")) == config
    seen = set()
    for split in SPLITS:
        for relative in (OUTPUT / (split + ".txt")).read_text(encoding="utf-8").splitlines():
            assert (OUTPUT / relative).is_file() and relative not in seen
            seen.add(relative)
    assert len(seen) == len(records)
    print("Verifying original files remained byte-for-byte unchanged...", flush=True)
    after = snapshot(roots)
    assert before == after, "Source files changed during conversion"
    (OUTPUT / "source_fingerprints.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"scope": args.scope, "images": len(records), "labels": len(records), "boxes": sum(totals.values()),
              "class_names": names, "class_boxes": dict(totals), "sources": expected_counts, "splits": stats,
              "source_files_checked": len(before), "original_files_unchanged": before == after,
              "source_images_copied_without_pixel_changes": True, "same_image_across_splits": False,
              "class_semantics": "Source distinctions retained; no automatic whole-object versus local-defect relabeling",
              "test_coverage": "Only original five classes have a held-out test split; external sources provide train/val only." if include_original else "External sources provide train/val only; no test split created."}
    (OUTPUT / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    class_rows = "\n".join(f"| {cls} | {name} | {totals[cls]} |" for cls, name in enumerate(names))
    readme = f"""# 绝缘子合并 YOLO 数据集

已在新目录转换并合并 {len(records)} 张图片、{len(records)} 个 TXT 标注、{sum(totals.values())} 个目标框。
所有图片在 images/，同名标注在 labels/；统一使用 data.yaml 训练。train.txt、val.txt、test.txt 负责划分，不按类别拆文件夹。
每行标注：class_id x_center y_center width height，坐标归一化到 0–1。

## 类别

保留原数据的整体框与外部缺陷局部框的语义差异。编号不能直接等同原始下载数据集的编号。

| 编号 | 类别 | 框数 |
|---|---|---:|
{class_rows}

## 来源与划分

- original_ 前缀（如包含）：9.22/insulator_yolo，原有五类绝缘子整体框。
- idd_ 前缀：Insulator_Defect_Detection/project，Supervisely JSON 已转换为 YOLO TXT；包含未区分状态的绝缘子本体、破损局部框、污闪局部框。
- vpmbgi_ 前缀：VPMBGI/dataset，玻璃绝缘子破损局部框；原 YOLO 类别重新编号。
- 沿用各来源的原始训练/验证/测试划分。训练 {stats['train']['images']} 张，验证 {stats['val']['images']} 张，测试 {stats['test']['images']} 张。
- {report['test_coverage']}
- source_manifest.csv 记录每个输出文件对应的源图片、标注及哈希；class_statistics.csv 给出分类别、分集合的数量。

## 原目录与回退

本次只在当前新目录写入数据，图片使用独立复制，没有使用硬链接。
对 {len(before)} 个源文件在转换前后计算 SHA-256，所有文件内容与文件集合完全一致，记录见 source_fingerprints.json。
需要回退时继续使用原来的 D:/data/9.22/insulator_yolo/data.yaml 即可；原始下载数据和 rawdata 不受影响。

## 训练前需了解

转换与合并已验证图片可读性、图片标注配对、框几何和类别编号；尚未开始训练。
来源间标注任务不同：原数据没有对全部缺陷局部单独画框，外部数据也没有逐个标注原有五种整体缺陷状态。因此这是保留来源语义的合并版本，不能把所有未标注类别都解释为确认不存在；若要训练可靠的统一九类模型，需要统一标注规则并补齐缺失标注。
原数据的“釉表面脱落”仍只有 4 个框，没有通过增强或重复文件虚增样本。
没有进行近重复检测；已核对字节相同图片不跨集合。

## 来源署名

外部两套数据均声明 CC BY 4.0，保留来源与作者署名：
- Insulator-Defect Detection，Jianfeng Zheng、Hang Wu、Han Zhang、Zhaoqi Wang、Weiyue Xu；Sensors 2022, 22(22), 8801；https://doi.org/10.3390/s22228801 ；原数据 https://doi.org/10.6084/m9.figshare.21200986 ；获取仓库 https://github.com/supervisely-ecosystem/aerial-power-infrastructure-detection-train-dataset 。本次转换 JSON 标注并重新编号，图片未改变。
- VPMBGI，phd-benel 及 README 所列原始数据贡献者；https://github.com/phd-benel/VPMBGI ；发布附件 https://github.com/phd-benel/VPMBGI/releases/tag/dataset 。本次重新编号标注，图片未改变。
- 许可链接：https://creativecommons.org/licenses/by/4.0/ 。完整来源说明仍保留在原下载目录。
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print("COMPLETE " + json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
