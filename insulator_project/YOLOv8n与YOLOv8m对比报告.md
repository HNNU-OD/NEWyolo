# YOLOv8n 与 YOLOv8m：五轮对比

两者使用相同五类数据和训练参数：2848张训练图、383张验证图，5轮，imgsz=640，batch=4，workers=0，amp=False，seed=20260928，close_mosaic=0。
这是各自预训练权重微调5轮的阶段性比较，不代表最终收敛效果。

|指标|YOLOv8n|YOLOv8m|
|---|---:|---:|
|mAP50|0.6599|0.7314|
|mAP50-95|0.4222|0.4744|
|Precision|0.7222|0.7768|
|Recall|0.6208|0.6832|
|总流程耗时（分钟）|9.46|20.87|
|第2–5轮平均耗时（秒）|108.44|244.68|
|按相同速度估算100轮（小时）|3.01|6.80|
|GPU抽样平均功率（W）|23.9|39.5|
|GPU抽样最高温度（°C）|66|77|

|类别|n mAP50|m mAP50|m Recall|
|---|---:|---:|---:|
|severe_soiling_no_discharge_whole|0.683|0.727|0.711|
|insulator_unspecified|0.993|0.994|0.988|
|broken_local|0.487|0.652|0.573|
|pollution_flashover_local|0.434|0.610|0.502|
|broken_glass_local|0.703|0.674|0.642|

m模型权重：`D:\newYOLO\insulator_project\runs\full_yolov8m\weights\best.pt`
n模型权重：`D:\newYOLO\insulator_project\runs\full\weights\best.pt`

m预训练权重下载自Ultralytics官方Hugging Face页面：https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8m.pt ，已比对官方SHA256通过。
数据仍有128组跨训练/验证集合近重复、标注口径不一致及空标签需复核问题。指标可能偏乐观，比较不能证明对新场景的泛化能力。
推理示例采用与n模型相同的5张验证图和conf=0.25，未按m模型预测效果挑图。
GPU功率不含CPU、屏幕与电源损耗；不能直接当整机电费。此次5轮已结束，未启动100轮训练。
