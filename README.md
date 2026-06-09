# TiViT Extension

本项目在 TiViT 的时间序列图像化分类框架上，加入 **Activity Graph** 作为多通道时间序列的视觉表示，并可与 Mantis / MOMENT 等时间序列基础模型的 embedding 拼接，用于 UCR、UEA、UCI HAR 等分类实验。

当前 README 只保留复现实验所需的主线说明：**Activity Graph + TiViT**。原 TiViT 的分段灰度图、历史 line plot 路径、Mantis / MOMENT 融合和分析工具放在后面的补充说明中。
数据以及checkpoint网盘链接: https://pan.baidu.com/s/1nTc5D7ROKLDA0YRyPHjzeA 提取码: 1234

## 核心思路

Activity Graph 的目标是把一个样本的多通道时间序列 `(n, T)` 转成一张二维图，再输入冻结的视觉 Transformer：

```text
多通道时间序列 (B, n, T)
  -> Activity Graph (按通道访问序列重排并拼接)
  -> 归一化、resize 到 224 x 224、复制为 3 通道
  -> ViT / CLIP-ViT 提取视觉 embedding
  -> 可选拼接 Mantis / MOMENT embedding
  -> logistic regression / nearest centroid / random forest 分类
```

Activity Graph 相关实现位于 [src/tivit.py](src/tivit.py)，主要入口如下：

| 函数 | 作用 |
| --- | --- |
| `get_optimal_order(n)` | 生成通道访问序列，使通道两两邻接关系尽量被覆盖 |
| `generate_activity_graph(signals, mode)` | 将单个样本 `(n, T)` 转为二维 Activity Graph |
| `preprocess_graph(signals, mode, img_size)` | 将 `(n, T)` 或 `(B, n, T)` 转为 ViT 输入 `(B, 3, 224, 224)` |
| `BaseTiViT.forward()` | 根据 `--image_mode` 调用图像化路径并提取 ViT 表示 |
| `save_activity_graph_samples(...)` | 从 dataloader 中导出若干张 Activity Graph 样例图，便于检查图像化效果 |

默认使用 `multicolumn` 图结构。每一行由当前通道及其前后邻接通道拼接得到：

```text
row_k = concat(signal[id_list[k - 1]],
               signal[id_list[k]],
               signal[id_list[k + 1]])
```

## ![IMG](D:/HDMI/TiViT_Extension/img/IMG.png)



活力图显示模块用于把训练集中生成的 Activity Graph 保存为 PNG 样例，方便检查多通道时间序列被转换成视觉输入后的纹理、通道排列和整体形状。该模块不会参与分类训练，只用于实验可视化和调试。

显示模块由命令行参数 `--save_activity_graph_samples` 控制。当 `--image_mode activity_graph` 且该参数大于 0 时，`main.py` 会在每个数据集加载后调用 [src/utils.py](src/utils.py) 中的 `save_activity_graph_samples()`：

```text
dataloader batch
  -> preprocess_graph(batch, mode="multicolumn")
  -> save_image(...)
  -> result_dir/activity_graph_samples/<dataset>_sampleN.png
```

示例：

```bash
python main.py \
  --vit_1_name /path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --classifier_type logistic_regression \
  --datasets uea \
  --dataset_names BasicMotions \
  --data_dir /path/to/dataset \
  --result_dir /path/to/results \
  --save_activity_graph_samples 5
```

运行后会在本次实验目录下生成：

```text
activity_graph_samples/
|-- BasicMotions_sample0.png
|-- BasicMotions_sample1.png
`-- ...
```

这些图片对应 ViT 的实际输入前处理结果：二维 Activity Graph 先被按样本归一化，再 resize 到 `224 x 224`，最后复制成 3 通道图像。

## 仓库结构

```text
TiViT_Extension/
|-- main.py                    # 实验入口：加载数据、提取 embedding、训练分类器
|-- requirements.txt           # Python 依赖
|-- README.md                  # 项目说明
|-- test_graph_shape.py        # Activity Graph 形状验证脚本，不依赖 torch
|-- scripts/
|   |-- run_lineplot_ucr.sh    # UCR 实验脚本
|   |-- run_lineplot_uea.sh    # UEA / Activity Graph 实验脚本
|   |-- check_ts_file.py       # 检查 aeon/sktime .ts 文件格式
|   `-- repair_ts_labels.py    # 修复 .ts 标签分隔符问题
|-- src/
|   |-- arguments.py           # 命令行参数
|   |-- datautils.py           # UCR / UEA / UCI HAR 数据加载
|   |-- embedding.py           # TiViT / Mantis / MOMENT embedding 提取
|   |-- tivit.py               # TiViT、Activity Graph、ViT 前向逻辑
|   |-- classifier.py          # 分类器训练与评估
|   |-- analysis.py            # 表示分析
|   |-- mutual_knn.py          # mutual kNN alignment
|   `-- utils.py               # 随机种子、结果写入、样本导出等工具
|-- assets/                    # 文档和图示资源
`-- img/                       # README 图片资源
```

`data/`、`outputs/`、`Mantis-8M/`、`activity_graph_samples/` 和临时下载目录属于本地数据、模型缓存或实验产物，默认不纳入版本管理。

## 环境安装

建议使用 Python 3.11。

```bash
conda create -n tivit_env python=3.11
conda activate tivit_env
python -m pip install -r requirements.txt
```

如果服务器不能联网，可以提前准备 HuggingFace 模型缓存，然后启用离线模式：

```bash
export HF_HOME=/path/to/huggingface/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 数据准备

项目支持三类数据入口：

| 参数 | 数据 | 说明 |
| --- | --- | --- |
| `--datasets ucr` | UCR | 单变量时间序列分类数据集 |
| `--datasets uea` | UEA | 多变量时间序列分类数据集 |
| `--datasets uci` | UCI HAR | 人体活动识别数据集，加载 9 个惯性信号通道 |

UCR / UEA 数据目录建议组织为：

```text
dataset/
|-- UCR/
|   |-- ECG200/
|   `-- FordA/
`-- UEA/
    |-- BasicMotions/
    `-- SelfRegulationSCP1/
```

UCI HAR 可以直接把 `data_dir` 指向 `UCI HAR Dataset`，也可以指向它的上级目录。代码会自动查找 `train/Inertial Signals`。

## 数据处理流程

数据处理入口位于 [src/datautils.py](src/datautils.py)，由 `main.py` 调用 `get_dataloader(dataset, args)` 完成。最终送入模型的数据统一为 PyTorch `DataLoader`，每个 batch 只包含一个张量：

```text
batch: (B, n, T)
B = batch size
n = 时间序列通道数
T = 序列长度
```

不同数据源的处理流程如下：

| 数据源 | 读取方式 | 主要处理 | 输出 |
| --- | --- | --- | --- |
| UCR | `aeon.datasets.load_classification()` | 不等长样本 padding；缺失值线性插值；必要时做全局标准化 | `train_loader`, `train_labels`, `test_loader`, `test_labels` |
| UEA | `aeon.datasets.load_classification()` | 默认按等长、无缺失数据读取 | `train_loader`, `train_labels`, `test_loader`, `test_labels` |
| UCI HAR | 读取 `Inertial Signals` 下 9 个惯性信号文件 | 按 `body_acc`、`body_gyro`、`total_acc` 的 x/y/z 通道堆叠；标签从 1-6 转为 0-5 | `train_loader`, `train_labels`, `test_loader`, `test_labels` |

UCI HAR 的样本形状固定为：

```text
train: (7352, 9, 128)
test:  (2947, 9, 128)
```

9 个通道分别是：

```text
body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z
```

验证集不是从磁盘单独读取，而是在分类阶段由 [src/classifier.py](src/classifier.py) 按 `--val_ratio` 从官方训练集 embedding 中随机划分；划分索引会写入 `result_dir/splits/`，便于复现实验。

## TiViT 处理逻辑

TiViT 的模型构造入口是 [src/tivit.py](src/tivit.py) 中的 `get_tivit()`。它根据 `--vit_1_name` / `--vit_2_name` 加载视觉骨干，并按模型结构选择 HuggingFace ViT 路径或 OpenCLIP 路径。

在 Activity Graph 主线中，TiViT 接收的是完整多通道 batch：

```text
input batch: (B, n, T)
```

处理流程如下：

```text
BaseTiViT.forward(inputs)
  -> preprocess_graph(inputs, mode="multicolumn")
       每个样本 (n, T)
       -> get_optimal_order(n)
       -> generate_activity_graph(...)
       -> 按样本 min-max 归一化
       -> resize 到 (224, 224)
       -> repeat 到 3 通道
  -> ViT 输入: (B, 3, 224, 224)
  -> forward_vit(...)
  -> aggregate_hidden_representations(...)
  -> TiViT embedding: (B, D)
```

其中 `--aggregation mean` 会对 token 维度取平均，`--aggregation cls_token` 会取 CLS token 表示。视觉骨干默认作为冻结特征提取器使用，主流程训练的是后续分类器。

需要注意的是，Activity Graph 必须看到完整的多通道样本，因此 [src/embedding.py](src/embedding.py) 对 `image_mode in {"line_plot", "activity_graph"}` 的 TiViT 分支会直接把 `(B, n, T)` 送入模型，而不是逐通道拆开。

## 多模态融合

多模态融合位于 [src/embedding.py](src/embedding.py)。项目把不同模型都视为 embedding 提取器，先分别计算 train / test embedding，再在特征维度拼接。

```text
原始 batch: (B, n, T)
  -> TiViT / Activity Graph embedding: (B, D_v)
  -> Mantis embedding:              (B, D_m)   可选
  -> MOMENT embedding:              (B, D_t)   可选
  -> concat_embeddings(...)
  -> fused embedding:               (B, D_v + D_m + D_t)
  -> classifier
```

各分支的输入和处理方式：

| 分支 | 启用参数 | 输入方式 | 输出 |
| --- | --- | --- | --- |
| TiViT / Activity Graph | `--vit_1_name` 或 `--vit_2_name` | 完整 batch `(B, n, T)` 生成 Activity Graph | 视觉 embedding |
| Mantis | `--mantis` | 逐通道取 `(B, 1, T)`，resize 到 512 后提取表示 | 时间序列 embedding |
| MOMENT | `--moment small|base|large` | 逐通道取 `(B, 1, T)`，padding 或 downsample 到 512 后提取表示 | 时间序列 embedding |

`concat_embeddings()` 会跳过未启用的分支，只拼接实际存在的 embedding。因此项目可以运行纯 Activity Graph、Activity Graph + Mantis、Activity Graph + MOMENT，或多个视觉骨干加时间序列基础模型的组合。

融合后的特征送入 [src/classifier.py](src/classifier.py) 中的分类器，支持 `logistic_regression`、`nearest_centroid` 和 `random_forest`。当前融合方式是特征级拼接，不涉及端到端联合训练。

## 运行 Activity Graph + TiViT

下面示例使用 CLIP-ViT-H 作为视觉骨干，并拼接 Mantis embedding：

```bash
python main.py \
  --vit_1_name /path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --mantis \
  --mantis_name /path/to/Mantis-8M \
  --classifier_type logistic_regression \
  --datasets uci \
  --batch_size 32 \
  --data_dir "/path/to/UCI HAR Dataset" \
  --result_dir /path/to/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

只使用 Activity Graph 视觉分支时，去掉 `--mantis` 和 `--mantis_name` 即可：

```bash
python main.py \
  --vit_1_name /path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --classifier_type logistic_regression \
  --datasets uci \
  --batch_size 32 \
  --data_dir "/path/to/UCI HAR Dataset" \
  --result_dir /path/to/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

导出 Activity Graph 样例图：

```bash
python main.py \
  --vit_1_name /path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --classifier_type logistic_regression \
  --datasets uea \
  --dataset_names BasicMotions \
  --data_dir /path/to/dataset \
  --result_dir /path/to/results \
  --save_activity_graph_samples 5
```

也可以使用已有脚本：

```bash
bash scripts/run_lineplot_uea.sh
```

注意：该脚本文件名沿用了历史命名，但当前脚本实际使用 `--image_mode activity_graph`。

## 结果文件

每次运行会在 `--result_dir` 下创建一个带时间戳的实验目录：

```text
results/
`-- 20260525_182154_uci_CLIP_ViT_H_14_laion2B_s32B_b79K_mantis_logistic_regression/
    |-- args.json
    |-- train_val.csv
    `-- splits/
        `-- UCIHAR_seed2021_val0.2.npz
```

| 文件 | 内容 |
| --- | --- |
| `args.json` | 本次运行的完整参数 |
| `train_val.csv` | 验证集和测试集分类结果 |
| `splits/*.npz` | 从官方训练集划分出的 train / validation 索引 |
| `activity_graph_samples/*.png` | 可选导出的 Activity Graph 样例图 |

## 快速验证

在没有 PyTorch 环境时，可以先用 numpy / PIL 验证 Activity Graph 的形状变化：

```bash
python test_graph_shape.py
```

预期输出包含：

```text
sample 0 graph shape: (18, 384)
final batch shape: (4, 3, 224, 224)
shape verification passed
```

## 关键参数

| 参数 | 说明 |
| --- | --- |
| `--image_mode activity_graph` | 使用 Activity Graph 视觉输入 |
| `--vit_1_name` / `--vit_2_name` | 视觉骨干 checkpoint，可为 HuggingFace id 或本地路径 |
| `--vit_1_layer` / `--vit_2_layer` | 提取表示的 ViT 层 |
| `--aggregation` | `mean` 或 `cls_token` |
| `--mantis` / `--mantis_name` | 启用 Mantis 并指定 checkpoint |
| `--moment` | 启用 MOMENT，取值 `small`、`base` 或 `large` |
| `--classifier_type` | `logistic_regression`、`nearest_centroid` 或 `random_forest` |
| `--datasets` | `ucr`、`uea` 或 `uci` |
| `--dataset_names` | 限定运行的数据集名称 |
| `--data_dir` | 数据目录 |
| `--result_dir` | 结果目录 |
| `--save_activity_graph_samples` | 每个数据集保存的 Activity Graph 样例数量 |

完整参数见 [src/arguments.py](src/arguments.py)。

## 补充说明

### 原 TiViT 分段灰度图

原 TiViT 的分段灰度图路径仍保留，可通过下面参数启用：

```bash
--image_mode segment
--patch_size sqrt
--stride 0.1
```

该路径会把单通道时间序列切成二维 patch 图，再输入 ViT。它主要用于和 Activity Graph 做对照实验。

### line plot 历史路径

代码中仍保留了 `ts2line_plot_transformation()`，但当前 `BaseTiViT.forward()` 对 `line_plot` 和 `activity_graph` 都走 `preprocess_graph()` 的整批多通道图预处理。也就是说，当前主线已经以 Activity Graph 替代旧 line plot 视觉输入。

### 时间序列基础模型融合

除视觉分支外，项目还支持拼接时间序列基础模型表示：

| 模型 | 启用方式 |
| --- | --- |
| Mantis | `--mantis --mantis_name /path/or/repo-id` |
| MOMENT | `--moment small|base|large` |

融合方式在 [src/embedding.py](src/embedding.py) 中实现：不同模型的 train / test embedding 会在特征维度上拼接后送入分类器。

### 表示分析工具

分类之外，项目保留了 TiViT 的表示分析功能：

| 功能 | 参数 |
| --- | --- |
| intrinsic dimension | `--get_intrinsic_dimension` |
| principal components | `--get_principal_components` |
| representation alignment | `--measure_alignment` |

这些功能用于分析不同模型或不同层的表示结构。

## Acknowledgements

- TiViT: Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers
- OpenCLIP / CLIP / DINOv2 / SigLIP / MAE 视觉骨干
- Mantis 和 MOMENT 时间序列基础模型
- UCR / UEA time series classification benchmark
- aeon 时间序列数据集工具
