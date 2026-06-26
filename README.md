# TiViT Extension

本仓库实现了基于 Activity Graph 的多通道时间序列图像化分类流程。代码主入口是 `main.py`，核心逻辑位于 `src/`。

Activity Graph 的作用是把一个多通道时间序列样本 `(n, T)` 组织成一张图像，使 ViT / CLIP-ViT 可以作为视觉特征提取器使用。分类器可以使用传统机器学习模型，例如 logistic regression、nearest centroid 或 random forest，也可以使用端到端训练的 PyTorch MLP 分类头。

推荐阅读顺序：

1. 先看 `代码结构`，确认各模块位置。
2. 按 `安装` 和 `数据输入` 准备环境与数据。
3. 直接运行 `运行示例` 中的命令。
4. 通过 `输出文件` 检查结果与样例图。

## 数据与模型资源

数据以及 checkpoint 网盘链接：

```text
https://pan.baidu.com/s/1nTc5D7ROKLDA0YRyPHjzeA
提取码: 1234
```

## 图示

![Activity Graph sample](Extra/img/IMG.png)

上图展示的是 Activity Graph 的整体思路：先对多通道信号排序，再把通道信号组织成图像。实际送入模型的图像由 `src/tivit.py` 中的 `preprocess_graph()` 生成。

## 代码结构

```text
TiViT_Extension/
|-- main.py                  # 实验入口：加载数据、提取 embedding、训练分类器
|-- requirements.txt         # Python 依赖
|-- scripts/
|   |-- run_lineplot_ucr.sh  # UCR 运行脚本
|   |-- run_lineplot_uea.sh  # UEA / Activity Graph 运行脚本
|   |-- check_ts_file.py     # 检查 .ts 数据文件
|   `-- repair_ts_labels.py  # 修复 .ts 标签分隔符
`-- src/
    |-- arguments.py         # 命令行参数
    |-- datautils.py         # UCR / UEA / UCI HAR 数据加载
    |-- embedding.py         # TiViT / Mantis / MOMENT embedding 提取
    |-- tivit.py             # TiViT、Activity Graph、ViT 前向逻辑
    |-- classifier.py        # 分类器训练与评估
    |-- mlp_classifier.py    # 端到端 MLP 分类头与 PyTorch 训练逻辑
    |-- analysis.py          # 表示分析
    |-- mutual_knn.py        # mutual kNN alignment
    `-- utils.py             # 随机种子、结果写入、样例图导出
```

## 安装

建议使用 Python 3.11。

```bash
conda create -n tivit_env python=3.11
conda activate tivit_env
python -m pip install -r requirements.txt
```

如果使用本地模型 checkpoint，直接把 `--vit_1_name`、`--vit_2_name` 或 `--mantis_name` 指向对应目录即可。

## 数据输入

`main.py` 通过 `--datasets` 选择数据入口：

| 参数 | 数据源 | 加载位置 |
| --- | --- | --- |
| `--datasets ucr` | UCR 单变量数据集 | `src/datautils.py` |
| `--datasets uea` | UEA 多变量数据集 | `src/datautils.py` |
| `--datasets uci` | UCI HAR | `src/datautils.py` |

模型接收到的 batch 形状统一为：

```text
(B, n, T)
B = batch size
n = 通道数
T = 时间长度
```

`--data_dir` 的含义取决于数据集类型：UCR / UEA 通常指向包含对应数据集目录的根目录；UCI HAR 可以指向 `UCI HAR Dataset` 本身，也可以指向它的上级目录。

## Activity Graph 流程

`--image_mode activity_graph` 时，TiViT 会把完整多通道样本转换为 Activity Graph：

```text
batch (B, n, T)
  -> preprocess_graph(...)
  -> image tensor (B, 3, 224, 224)
  -> ViT / CLIP-ViT
  -> embedding
  -> classifier
```

相关函数：

| 函数 | 文件 | 作用 |
| --- | --- | --- |
| `get_optimal_order` | `src/tivit.py` | 生成通道访问顺序 |
| `generate_activity_graph` | `src/tivit.py` | 生成二维 Activity Graph 矩阵 |
| `render_activity_waveform_graph` | `src/tivit.py` | 生成波形样式 Activity Graph |
| `preprocess_graph` | `src/tivit.py` | 转为 ViT 输入张量 |
| `save_activity_graph_samples` | `src/utils.py` | 保存模型输入样例图 |
| `save_activity_lineplot_samples` | `src/utils.py` | 保存活动折线图样例图 |

## 图像模式

`--image_mode` 决定时间序列如何变成视觉输入：

| 模式 | 说明 |
| --- | --- |
| `activity_graph` | 当前主线模式，将完整多通道样本转换为波形样式 Activity Graph |
| `activity_matrix` | 使用 Activity Graph 矩阵形式，按数值归一化后输入 ViT |
| `line_plot` | 逐通道生成普通折线图输入 |
| `segment` | 原 TiViT 分段灰度图路径，需要配合 `--patch_size` 和 `--stride` |

如果只是复现当前 Activity Graph 实验，优先使用 `--image_mode activity_graph`。

## 端到端 MLP 分类头

默认的 `logistic_regression`、`nearest_centroid` 和 `random_forest` 仍然采用两阶段流程：先用 TiViT / Mantis / MOMENT 提取 embedding，并把多分支 embedding 拼接成 numpy 特征，再训练传统机器学习分类器。这一路径不会把分类损失反向传播到特征分支。

当 `--classifier_type mlp` 时，流程切换为 PyTorch 端到端训练：

```text
batch (B, n, T)
  -> enabled feature branches: TiViT / Mantis / MOMENT
  -> concatenate branch embeddings
  -> MLP classifier head
  -> cross-entropy loss
  -> backpropagate through MLP and enabled feature branches
```

MLP 模式下的 Mantis 不使用 `MantisTrainer.transform()`，因为该接口内部会设置 `torch.no_grad()`，只适合推理式 embedding 提取。代码会直接调用 `Mantis8M.forward()`，因此 `--mantis --classifier_type mlp` 可以参与反向传播训练。

MLP 训练相关参数：

| 参数 | 说明 |
| --- | --- |
| `--mlp_hidden_dim` | MLP 隐藏层维度，默认 `512` |
| `--mlp_num_layers` | MLP 线性层数，默认 `2` |
| `--mlp_dropout` | dropout 概率，默认 `0.1` |
| `--mlp_lr` | AdamW 学习率，默认 `1e-4` |
| `--mlp_weight_decay` | AdamW 权重衰减，默认 `1e-4` |
| `--mlp_epochs` | MLP 训练轮数，默认 `20` |

## 运行示例

只使用 Activity Graph 视觉分支：

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

导出 Activity Graph 和活动折线图样例：

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
  --save_activity_graph_samples 5 \
  --save_activity_lineplot_samples 5
```

`--save_activity_graph_samples` 保存的是模型实际输入样式；`--save_activity_lineplot_samples` 保存的是更适合展示和检查原始波形趋势的折线图，不参与训练。

使用 Activity Graph 视觉分支和 Mantis 时序分支，并通过 MLP 端到端训练：
```bash
python main.py \
  --vit_1_name /path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --mantis \
  --mantis_name /path/to/Mantis-8M \
  --classifier_type mlp \
  --mlp_hidden_dim 512 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-4 \
  --mlp_epochs 20 \
  --datasets uci \
  --batch_size 16 \
  --data_dir "/path/to/UCI HAR Dataset" \
  --result_dir /path/to/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

## 输出文件

每次运行会在 `--result_dir` 下创建一个带时间戳的实验目录。

| 输出 | 内容 |
| --- | --- |
| `args.json` | 本次运行参数 |
| `train_val.csv` | 验证集和测试集分类结果 |
| `splits/*.npz` | train / validation 划分索引 |
| `activity_graph_samples/*.png` | 可选导出的 Activity Graph 样例 |
| `activity_lineplot_samples/*.png` | 可选导出的活动折线图样例 |

`train_val.csv` 是主要结果文件；`args.json` 用于记录本次运行参数，方便之后复现实验。

## 关键参数

| 参数 | 说明 |
| --- | --- |
| `--image_mode` | 图像化方式：`line_plot`、`activity_graph`、`activity_matrix`、`segment` |
| `--vit_1_name` / `--vit_2_name` | 视觉骨干 checkpoint |
| `--vit_1_layer` / `--vit_2_layer` | 提取表示的 ViT 层 |
| `--aggregation` | `mean` 或 `cls_token` |
| `--classifier_type` | `logistic_regression`、`nearest_centroid`、`random_forest` 或 `mlp` |
| `--mlp_hidden_dim` / `--mlp_num_layers` | MLP 隐藏维度与线性层数 |
| `--mlp_dropout` / `--mlp_lr` / `--mlp_weight_decay` / `--mlp_epochs` | MLP 端到端训练参数 |
| `--mantis --classifier_type mlp` | 直接调用 `Mantis8M.forward()` 参与反向传播，不使用 `MantisTrainer.transform()` 推理接口 |
| `--datasets` | `ucr`、`uea` 或 `uci` |
| `--dataset_names` | 限定运行的数据集名称 |
| `--data_dir` | 数据目录 |
| `--result_dir` | 结果目录 |
| `--save_activity_graph_samples` | 每个数据集保存的 Activity Graph 样例数量 |
| `--save_activity_lineplot_samples` | 每个数据集保存的活动折线图样例数量 |

完整参数见 `src/arguments.py`。

## 常见注意事项

- 至少需要启用一个表示分支：设置 `--vit_1_name` / `--vit_2_name`，或启用 `--mantis` / `--moment`。
- `activity_graph` 和 `activity_matrix` 会把完整 `(B, n, T)` batch 送入 TiViT，不会逐通道拆开。
- `segment` 模式需要设置 `--patch_size` 和 `--stride`。
- 导出样例图只用于检查和展示，不改变分类训练逻辑。
- 结果目录使用时间戳创建，重复运行不会覆盖旧实验结果。

## 验证

检查 Python 文件语法：

```bash
python -m py_compile main.py src/arguments.py src/classifier.py src/mlp_classifier.py src/utils.py src/tivit.py src/embedding.py
```
