# Tivit_Extension

本项目是基于 TiViT 的时间序列图像化分类实验扩展，核心关注点是：**将一维或多维时间序列渲染成折线图图像，再输入预训练视觉 Transformer 提取表示，用于 UCR/UEA 时间序列分类对比**。

通过网盘分享的文件：dataset

链接: https://pan.baidu.com/s/1ydgwTojNom_kKigBZXd54w?pwd=1234 提取码: 1234

当前默认输入形式是折线图：

```bash
--image_mode line_plot
```

原始的分段灰度图表示仍然保留，可通过 `--image_mode segment` 启用，主要用于和原 TiViT 方案做对照。

## 项目亮点

- **折线图优先**：默认把每条时间序列转换为 224 x 224 的白底黑线图像，更贴近人类查看时间序列的方式。
- **支持多通道时间序列**：多变量数据会按通道分别绘制折线图，再分别送入视觉骨干网络提取特征。
- **复用大规模视觉模型**：支持 CLIP/OpenCLIP、DINOv2、SigLIP 2、MAE 等预训练 ViT。
- **可与 TSFM 融合**：支持拼接 Mantis、MOMENT 等时间序列基础模型表示，观察视觉表示与时间序列表示的互补性。
- **保留可复现实验设置**：使用官方 UCR/UEA train/test split，并将验证集划分索引保存到结果目录。

## 方法概览

![Plot](img/Plot.png)

折线图模式的流程如下：

1. 读取 UCR 或 UEA 时间序列数据。
2. 对每条序列做 robust scaling，减弱异常值和尺度差异的影响。
3. 将序列值归一化到图像坐标，并插值到固定宽度。
4. 渲染为白底黑线的 RGB 图像。
5. 输入冻结的 ViT/CLIP-ViT，在指定层提取 hidden representation。
6. 对 token 表示做 `mean` 或 `cls_token` 聚合。
7. 使用 logistic regression、nearest centroid 或 random forest 完成分类。

折线图转换实现位于 [src/tivit.py](src/tivit.py)，核心参数是 `--image_mode line_plot`。

## 环境安装

建议使用 Python 3.11。

```bash
conda create -n tivit_env python=3.11
conda activate tivit_env
python -m pip install -r requirements.txt
```

## 数据集

项目面向 [UCR/UEA time series classification benchmark](https://www.timeseriesclassification.com/)。

数据可通过 [aeon](https://www.aeon-toolkit.org) 加载。默认流程会：

- 对 UCR 数据中的缺失值做线性插值；
- 对不等长序列做 padding；
- 从官方训练集里按 `--val_ratio` 划分验证集；
- 将实际 train/validation 索引写入 `result_dir/splits/`，方便复现和跨模型比较。

如需使用 aeon 自带预处理，可添加：

```bash
--aeon
```

## 支持的视觉骨干

| 类型 | 示例 checkpoint |
| --- | --- |
| CLIP/OpenCLIP | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| DINOv2 | `facebook/dinov2-base` |
| SigLIP 2 | `google/siglip2-so400m-patch14-224` |
| MAE | `facebook/vit-mae-base` |

其中 CLIP-ViT-H 是当前折线图实验中推荐优先使用的骨干。

## 折线图分类实验

下面是在 UCR 数据集上使用 CLIP-ViT-H 折线图输入进行分类的示例：

```bash
python main.py \
  --vit_1_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --classifier_type logistic_regression \
  --datasets ucr \
  --data_dir /path/to/your/data \
  --result_dir /path/to/save/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

说明：

- `--image_mode line_plot` 是默认值，显式写出便于记录实验配置。
- 折线图模式不需要 `--patch_size`，代码会自动跳过 2D 分段 patch size 搜索。
- `--vit_1_layer 14` 表示截取 ViT 的第 14 层表示。
- `--aggregation mean` 表示对 token hidden states 做平均池化。

## 指定数据集运行

可以用 `--dataset_names` 只跑部分数据集。下面是和 DMMV 对比时常用的受控设置：

- random seed: `2021`
- validation ratio: `0.2`
- UCR: `ECG200`, `FordA`
- UEA: `BasicMotions`, `SelfRegulationSCP1`

运行 UCR 折线图实验：

```bash
python main.py \
  --vit_1_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --classifier_type logistic_regression \
  --datasets ucr \
  --dataset_names ECG200 FordA \
  --data_dir /path/to/your/data \
  --result_dir /path/to/save/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

运行 UEA 折线图实验：

```bash
python main.py \
  --vit_1_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --classifier_type logistic_regression \
  --datasets uea \
  --dataset_names BasicMotions SelfRegulationSCP1 \
  --data_dir /path/to/your/data \
  --result_dir /path/to/save/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

项目也提供了脚本入口：

```bash
bash scripts/run_lineplot_ucr.sh
bash scripts/run_lineplot_uea.sh
```

## 与时间序列基础模型融合

如果希望比较或融合传统时间序列基础模型，可以添加：

```bash
--mantis
```

或：

```bash
--moment base
```

例如，将折线图 ViT 表示和 Mantis 表示拼接后分类：

```bash
python main.py \
  --vit_1_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --mantis \
  --classifier_type logistic_regression \
  --datasets ucr \
  --data_dir /path/to/your/data \
  --result_dir /path/to/save/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

## 结果文件

每次运行会在 `result_dir` 下新建带时间戳的实验目录，通常包含：

- `args.json`：本次运行的完整参数；
- `train_val.csv`：各数据集验证集和测试集结果；
- `splits/*.npz`：从官方训练集划分出的 train/validation 索引。

结果表中会记录 `image_mode`，因此可以直接区分折线图输入和分段灰度图输入。

## 其他分析功能

除了分类，本项目还保留 TiViT 的表示分析功能：

```bash
--get_intrinsic_dimension
--get_principal_components
--measure_alignment
```

这些功能可用于分析不同 ViT 层的表示结构，以及折线图视觉表示和 Mantis/MOMENT 表示之间的 alignment。

## 主要参数

| 参数 | 作用 |
| --- | --- |
| `--image_mode line_plot` | 使用折线图作为视觉输入，当前默认模式 |
| `--image_mode segment` | 使用原 TiViT 分段灰度图输入 |
| `--vit_1_name` | 第一个视觉骨干 checkpoint |
| `--vit_1_layer` | 提取表示的 ViT 层 |
| `--aggregation` | hidden states 聚合方式，支持 `mean` 和 `cls_token` |
| `--classifier_type` | 分类器，支持 `logistic_regression`、`nearest_centroid`、`random_forest` |
| `--datasets` | 选择 `ucr` 或 `uea` |
| `--dataset_names` | 限定要运行的数据集名称 |
| `--mantis` | 拼接 Mantis 时间序列表示 |
| `--moment` | 拼接 MOMENT 表示，支持 `small`、`base`、`large` |

完整参数定义见 [src/arguments.py](src/arguments.py)。

## 致谢

本项目基于 TiViT 思路扩展折线图输入实验，并参考了以下工作：

- TiViT: Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers
- OpenCLIP / CLIP 视觉骨干
- Mantis 和 MOMENT 时间序列基础模型
- aeon 时间序列数据集工具
