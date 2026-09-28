# 当前数据已调整为五类

已删除原编号1、2、3、4的全部202个标注框。保留图片及划分清单，原0/5/6/7/8分别映射到新0/1/2/3/4；当前共9532框、3370张图片。

删除后新增125份空标签，总空标签139份。空标签作为当前五类任务的背景图，不代表绝缘子状态正常。

所有活动YAML、classes.txt、统计表和训练脚本已同步。旧labels.cache已清理。官方检查：缺失0、损坏0。

旧runs目录、权重、run_summary.json及训练日志属于此前九类试跑，未修改，不能直接当作新五类训练结果或按新编号解释。五类模型需要重新训练。

旧128组跨集合近重复、整体框/局部框的标注差异仍存在。当前测试集只包含新类别0，不能评价全部五类。

备份：`D:\newYOLO\backups\before_remove_classes_1_2_3_4_20260928_140143.zip`

新配置：`D:/newYOLO/insulator_project/data_full.yaml`

新旧编号及数量详见class_removal_report.json。下方或其他早期说明属于历史记录；数据目录的build_dataset.py是原九类构建脚本，不用于重建当前五类数据。

---

# 绝缘子合并 YOLO 数据集

已在新目录转换并合并 3370 张图片、3370 个 TXT 标注、9734 个目标框。
所有图片在 images/，同名标注在 labels/；统一使用 data.yaml 训练。train.txt、val.txt、test.txt 负责划分，不按类别拆文件夹。
每行标注：class_id x_center y_center width height，坐标归一化到 0–1。

## 类别

保留原数据的整体框与外部缺陷局部框的语义差异。编号不能直接等同原始下载数据集的编号。

| 编号 | 类别 | 框数 |
|---|---|---:|
| 0 | 污秽较为严重，但表面无明显放电（整体框） | 3513 |
| 1 | 破损（整体框） | 105 |
| 2 | 污秽（放电痕迹）（整体框） | 54 |
| 3 | 固定不牢固（倾斜）（整体框） | 39 |
| 4 | 釉表面脱落（整体框） | 4 |
| 5 | 绝缘子本体（状态未区分） | 1645 |
| 6 | 破损缺陷（局部框） | 963 |
| 7 | 污闪缺陷（局部框） | 2223 |
| 8 | 玻璃绝缘子破损（局部框） | 1188 |

## 来源与划分

- original_ 前缀（如包含）：9.22/insulator_yolo，原有五类绝缘子整体框。
- idd_ 前缀：Insulator_Defect_Detection/project，Supervisely JSON 已转换为 YOLO TXT；包含未区分状态的绝缘子本体、破损局部框、污闪局部框。
- vpmbgi_ 前缀：VPMBGI/dataset，玻璃绝缘子破损局部框；原 YOLO 类别重新编号。
- 沿用各来源的原始训练/验证/测试划分。训练 2848 张，验证 383 张，测试 139 张。
- Only original five classes have a held-out test split; external sources provide train/val only.
- source_manifest.csv 记录每个输出文件对应的源图片、标注及哈希；class_statistics.csv 给出分类别、分集合的数量。

## 原目录与回退

本次只在当前新目录写入数据，图片使用独立复制，没有使用硬链接。
对 6752 个源文件在转换前后计算 SHA-256，所有文件内容与文件集合完全一致，记录见 source_fingerprints.json。
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
