# TiViT Extension

一句话简介：本项目将时间序列传感器数据转换为视觉图像表示，并结合预训练 ViT 与时间序列基础模型完成 UCR、UEA、UCI HAR 分类实验。

本项目是基于 TiViT 的时间序列图像化分类实验扩展，核心关注点是：**将一维或多维时间序列渲染为视觉输入，再输入预训练视觉 Transformer 提取表示，用于时间序列分类对比**。项目保留原有折线图、分段灰度图实验路径，并新增 **Activity Graph** 生成方式，用于替换视觉分支输入。

通过网盘分享的文件：dataset

链接: https://pan.baidu.com/s/1ydgwTojNom_kKigBZXd54w?pwd=1234 提取码: 1234

## 功能概览

| 模块 | 能力 |
| --- | --- |
| 视觉输入 | `line_plot`、`activity_graph`、`segment` |
| 数据集入口 | `ucr`、`uea`、`uci` |
| 视觉骨干 | CLIP/OpenCLIP、DINOv2、SigLIP 2、MAE |
| 时间序列基础模型 | Mantis、MOMENT |
| 融合方式 | 视觉 embedding 与时间序列 embedding 拼接 |
| 分类器 | `logistic_regression`、`nearest_centroid`、`random_forest` |
| 复现实验 | 保存 `args.json`、`train_val.csv`、`splits/*.npz` |

## 方法概览

![Plot](img/Plot.png)

### 总体数据流

```text
原始时间序列 (B, n, T)
  → 视觉输入生成 line_plot / activity_graph / segment
  → RGB 图像 (B, 3, 224, 224)
  → 冻结 ViT / CLIP-ViT
  → hidden states
  → mean 或 cls_token 聚合
  → 视觉 embedding
  → 可选拼接 Mantis / MOMENT embedding
  → logistic regression / nearest centroid / random forest
  → 分类结果
```

### 原始 line_plot 流程

| 步骤 | 说明 |
| --- | --- |
| 1 | 读取 UCR 或 UEA 时间序列数据 |
| 2 | 对每条序列做 robust scaling，减弱异常值和尺度差异影响 |
| 3 | 将序列值归一化到图像坐标，并插值到固定宽度 |
| 4 | 渲染为白底黑线 RGB 图像 |
| 5 | 输入冻结 ViT/CLIP-ViT，在指定层提取 hidden representation |
| 6 | 对 token 表示做 `mean` 或 `cls_token` 聚合 |
| 7 | 使用分类器完成分类 |

折线图转换实现位于 [src/tivit.py](src/tivit.py)，历史 baseline 的核心参数是：

```bash
--image_mode line_plot
```

原始的分段灰度图表示仍然保留，可通过以下参数启用，主要用于和原 TiViT 方案做对照：

```bash
--image_mode segment
```

## Activity Graph

当前新增的 Activity Graph 方法位于 [src/tivit.py](src/tivit.py)。

| 函数 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| `get_optimal_order(n)` | 通道数 `n` | `id_list` | 生成通道访问顺序，使任意两个通道至少相邻一次 |
| `build_single_column_graph(signals, id_list)` | `(n, T)` | `(len(id_list), T)` | 按 `id_list` 逐行堆叠 |
| `build_multicolumn_graph(signals, id_list)` | `(n, T)` | `(len(id_list), 3*T)` | 每行拼接左、中、右三个相邻通道 |
| `generate_activity_graph(signals, mode)` | `(n, T)` | 二维 Activity Graph | 统一生成入口 |
| `preprocess_graph(signals, mode, img_size)` | `(n, T)` 或 `(B, n, T)` | `(B, 3, 224, 224)` | 归一化、resize、复制 3 通道 |

### Activity Graph 形状流

```text
signals: (B, n, T)
  → per sample: (n, T)
  → generate_activity_graph: (len(id_list), 3*T)
  → normalize per sample: [0, 1]
  → resize: (224, 224)
  → repeat channel: (3, 224, 224)
  → batch stack: (B, 3, 224, 224)
```

`multicolumn` 模式下，每一行的构造为：

```text
left  = signals[id_list[k - 1]]
mid   = signals[id_list[k]]
right = signals[id_list[k + 1]]
row   = concat(left, mid, right)
```

形状验证示例：

| 设置 | Activity Graph 原始形状 | ViT 输入形状 |
| --- | --- | --- |
| `B=4, n=6, T=128` | `(18, 384)` | `(4, 3, 224, 224)` |

项目根目录提供隔离验证脚本 [test_graph_shape.py](test_graph_shape.py)，用于在无法导入 PyTorch 的环境中用 numpy/PIL 验证形状：

```bash
python test_graph_shape.py
```

预期输出：

```text
sample 0 graph shape: (18, 384)
sample 1 graph shape: (18, 384)
sample 2 graph shape: (18, 384)
sample 3 graph shape: (18, 384)
final batch shape: (4, 3, 224, 224)
shape verification passed
```

## 视觉分支替换

模型入口位于 [src/tivit.py](src/tivit.py) 的 `BaseTiViT.forward()`。

```python
if self.image_mode in {"line_plot", "activity_graph"}:
    inputs = preprocess_graph(inputs, mode="multicolumn")
elif self.image_mode == "segment":
    inputs = self.ts2image_transformation(...)
```

| 模式 | 当前行为 |
| --- | --- |
| `activity_graph` | 使用 Activity Graph 作为视觉输入 |
| `line_plot` | 当前已切换到 Activity Graph 预处理路径，用于替换旧视觉输入 |
| `segment` | 保留原 TiViT 分段灰度图路径 |

TiViT embedding 入口已调整为：视觉分支接收完整多通道 batch `(B, n, T)`，不再逐通道拆开调用，否则 Activity Graph 无法覆盖通道组合。

## 环境安装

建议使用 Python 3.11。

```bash
conda create -n tivit_env python=3.11
conda activate tivit_env
python -m pip install -r requirements.txt
```

服务器无法联网时，可以先在可联网机器下载并上传 HuggingFace cache，然后启用离线模式：

```bash
export HF_HOME=/home/guoyin/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 数据集

项目面向 [UCR/UEA time series classification benchmark](https://www.timeseriesclassification.com/)，UCR/UEA 数据可通过 [aeon](https://www.aeon-toolkit.org) 加载。

| 数据入口 | 说明 | 典型数据集 |
| --- | --- | --- |
| `--datasets ucr` | UCR 单变量时间序列分类数据集 | `ECG200`、`FordA` |
| `--datasets uea` | UEA 多变量时间序列分类数据集 | `BasicMotions`、`SelfRegulationSCP1` |
| `--datasets uci` | UCI HAR 人体活动识别数据集 | `UCIHAR` |

### UCR/UEA 目录结构

```text
dataset/
├── UCR/
│   ├── ECG200/
│   └── FordA/
└── UEA/
    ├── BasicMotions/
    └── SelfRegulationSCP1/
```

UCR/UEA 默认流程：

| 处理 | 说明 |
| --- | --- |
| 缺失值 | 对 UCR 数据中的缺失值做线性插值 |
| 不等长序列 | 对不等长序列做 padding |
| 验证集 | 从官方训练集按 `--val_ratio` 划分 |
| 复现索引 | 写入 `result_dir/splits/` |

如需使用 aeon 自带预处理，可添加：

```bash
--aeon
```

### UCI HAR 目录结构

```text
data/
└── UCI HAR Dataset/
    ├── activity_labels.txt
    ├── features.txt
    ├── features_info.txt
    ├── README.txt
    ├── train/
    │   ├── X_train.txt
    │   ├── y_train.txt
    │   ├── subject_train.txt
    │   └── Inertial Signals/
    │       ├── body_acc_x_train.txt
    │       ├── body_acc_y_train.txt
    │       ├── body_acc_z_train.txt
    │       ├── body_gyro_x_train.txt
    │       ├── body_gyro_y_train.txt
    │       ├── body_gyro_z_train.txt
    │       ├── total_acc_x_train.txt
    │       ├── total_acc_y_train.txt
    │       └── total_acc_z_train.txt
    └── test/
        ├── X_test.txt
        ├── y_test.txt
        ├── subject_test.txt
        └── Inertial Signals/
            ├── body_acc_x_test.txt
            ├── body_acc_y_test.txt
            ├── body_acc_z_test.txt
            ├── body_gyro_x_test.txt
            ├── body_gyro_y_test.txt
            ├── body_gyro_z_test.txt
            ├── total_acc_x_test.txt
            ├── total_acc_y_test.txt
            └── total_acc_z_test.txt
```

UCI HAR 被视为一个数据集 `UCIHAR`，包含 6 个活动类别，不是 6 个独立数据集。

| Split | Shape | 说明 |
| --- | --- | --- |
| train | `(7352, 9, 128)` | 7352 个样本，9 个 inertial-signal 通道，长度 128 |
| test | `(2947, 9, 128)` | 2947 个样本，9 个 inertial-signal 通道，长度 128 |

9 个通道为：

| 类型 | 通道 |
| --- | --- |
| body acceleration | `body_acc_x`、`body_acc_y`、`body_acc_z` |
| body gyroscope | `body_gyro_x`、`body_gyro_y`、`body_gyro_z` |
| total acceleration | `total_acc_x`、`total_acc_y`、`total_acc_z` |

6 个活动类别为：

| Label | Activity |
| ---: | --- |
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

## 支持的视觉骨干

| 类型 | 示例 checkpoint |
| --- | --- |
| CLIP/OpenCLIP | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| DINOv2 | `facebook/dinov2-base` |
| SigLIP 2 | `google/siglip2-so400m-patch14-224` |
| MAE | `facebook/vit-mae-base` |

其中 CLIP-ViT-H 是当前实验中推荐优先使用的骨干。

## 主要参数

完整参数定义见 [src/arguments.py](src/arguments.py)。

| 参数 | 作用 | 示例 |
| --- | --- | --- |
| `--vit_1_name` | 第一个视觉骨干 checkpoint | `/home/guoyin/hf_models/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| `--vit_2_name` | 第二个视觉骨干 checkpoint | 可选 |
| `--vit_1_layer` | 提取表示的 ViT 层 | `14` |
| `--vit_2_layer` | 第二个 ViT 的层 | 可选 |
| `--aggregation` | hidden states 聚合方式 | `mean`、`cls_token` |
| `--image_mode` | 视觉输入模式 | `line_plot`、`activity_graph`、`segment` |
| `--patch_size` | 分段灰度图 patch size 策略 | `sqrt`、`linspace` |
| `--stride` | patch stride 比例 | `0.1` |
| `--mantis` | 拼接 Mantis 时间序列表示 | flag |
| `--mantis_name` | Mantis checkpoint | `paris-noah/Mantis-8M` |
| `--moment` | 拼接 MOMENT 表示 | `small`、`base`、`large` |
| `--classifier_type` | 分类器 | `logistic_regression`、`nearest_centroid`、`random_forest` |
| `--datasets` | 数据集入口 | `ucr`、`uea`、`uci` |
| `--dataset_names` | 限定数据集名称 | `UCIHAR`、`BasicMotions` |
| `--batch_size` | batch size | `32` |
| `--aeon` | 启用 aeon 预处理 | flag |
| `--data_dir` | 数据目录 | `/path/to/data` |
| `--result_dir` | 结果目录 | `/path/to/results` |
| `--random_seed` | 随机种子 | `2021` |
| `--val_ratio` | 验证集比例 | `0.2` |
| `--measure_alignment` | 计算表示空间 alignment | flag |
| `--get_intrinsic_dimension` | 计算 intrinsic dimension | flag |
| `--get_principal_components` | 计算主成分数量 | flag |

## 运行示例

### UCI HAR: Activity Graph + Mantis

```bash
cd /home/guoyin/TiViT-main

export HF_HOME=/home/guoyin/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python main.py \
  --vit_1_name /home/guoyin/hf_models/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --mantis \
  --mantis_name paris-noah/Mantis-8M \
  --classifier_type logistic_regression \
  --datasets uci \
  --batch_size 32 \
  --data_dir "/home/guoyin/TiViT-main/data/UCI HAR Dataset" \
  --result_dir /home/guoyin/TiViT-main/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

### UCI HAR: pure Activity Graph

```bash
cd /home/guoyin/TiViT-main

python main.py \
  --vit_1_name /home/guoyin/hf_models/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --classifier_type logistic_regression \
  --datasets uci \
  --batch_size 32 \
  --data_dir "/home/guoyin/TiViT-main/data/UCI HAR Dataset" \
  --result_dir /home/guoyin/TiViT-main/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

### UCR line_plot baseline

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

### UEA line_plot + Mantis baseline

```bash
cd /home/guoyin/TiViT-main

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python main.py \
  --vit_1_name /home/guoyin/hf_models/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --mantis \
  --mantis_name paris-noah/Mantis-8M \
  --classifier_type logistic_regression \
  --datasets uea \
  --dataset_names BasicMotions SelfRegulationSCP1 \
  --data_dir /home/guoyin/dmmv_extension/dmmv/dmmv/dataset \
  --result_dir /home/guoyin/TiViT-main/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

### UEA Activity Graph + Mantis

```bash
cd /home/guoyin/TiViT-main

export HF_HOME=/home/guoyin/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python main.py \
  --vit_1_name /home/guoyin/hf_models/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --mantis \
  --mantis_name paris-noah/Mantis-8M \
  --classifier_type logistic_regression \
  --datasets uea \
  --dataset_names BasicMotions SelfRegulationSCP1 \
  --batch_size 32 \
  --data_dir /home/guoyin/dmmv_extension/dmmv/dmmv/dataset \
  --result_dir /home/guoyin/TiViT-main/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

项目也提供了脚本入口：

```bash
bash scripts/run_lineplot_ucr.sh
bash scripts/run_lineplot_uea.sh
```

## 与时间序列基础模型融合

| 模型 | 启用方式 | 说明 |
| --- | --- | --- |
| Mantis | `--mantis` | 通过 `--mantis_name` 指定 checkpoint |
| MOMENT | `--moment base` | 支持 `small`、`base`、`large` |

Mantis checkpoint 可以使用 HuggingFace repo id：

```bash
--mantis_name paris-noah/Mantis-8M
```

也可以在已正确保存的本地 HuggingFace cache 中离线加载。

## 结果文件

每次运行会在 `result_dir` 下新建带时间戳的实验目录：

```text
results/
└── 20260525_182154_uci_CLIP_ViT_H_14_laion2B_s32B_b79K_mantis_logistic_regression/
    ├── args.json
    ├── train_val.csv
    └── splits/
        └── UCIHAR_seed2021_val0.2.npz
```

| 文件 | 内容 |
| --- | --- |
| `args.json` | 本次运行的完整参数 |
| `train_val.csv` | 各数据集验证集和测试集结果 |
| `splits/*.npz` | 从官方训练集划分出的 train/validation 索引 |

`train_val.csv` 当前字段为：

```text
dataset,image_mode,patch_size,val_accuracy,test_accuracy
```

是否启用 Mantis、模型路径、seed、data_dir 等信息请查看同目录下的 `args.json`。

## 实验结果

### Activity Graph + Mantis

| Benchmark | Dataset | Classes | Validation accuracy | Test accuracy |
| --- | --- | ---: | ---: | ---: |
| UCI | UCIHAR | 6 | **0.9905** | **0.9800** |

实验设置：

| 项目 | 配置 |
| --- | --- |
| Vision modality | `--image_mode activity_graph` |
| Vision backbone | `CLIP-ViT-H-14-laion2B-s32B-b79K` |
| Time-series modality | `--mantis` with `paris-noah/Mantis-8M` |
| Fusion | concatenate embeddings |
| Classifier | `logistic_regression` |
| Random seed | `2021` |
| Validation ratio | `0.2` |

### Previous line_plot + Mantis baseline

| Benchmark | Dataset | Test accuracy |
| --- | --- | ---: |
| UCR | ECG200 | 0.84 |
| UCR | FordA | **0.89** |
| UEA | BasicMotions | **1.00** |
| UEA | SelfRegulationSCP1 | 0.82 |

### Activity Graph sanity runs

以下结果来自 `~/TiViT-main/data/UEA` 中的本地 UEA 数据，用于确认流程可运行。

| Benchmark | Dataset | Test accuracy |
| --- | --- | ---: |
| UEA | BasicMotions | **1.00** |
| UEA | Epilepsy | **1.00** |
| UEA | FingerMovements | 0.43 |
| UEA | NATOPS | 0.85 |

## 其他分析功能

除了分类，本项目还保留 TiViT 的表示分析功能：

| 功能 | 参数 |
| --- | --- |
| intrinsic dimension | `--get_intrinsic_dimension` |
| principal components | `--get_principal_components` |
| representation alignment | `--measure_alignment` |

这些功能可用于分析不同 ViT 层的表示结构，以及视觉表示与 Mantis/MOMENT 表示之间的 alignment。

## 代码结构

```text
TiViT_Extension/
├── main.py                    # 实验主入口，调度数据、模型、embedding、分类器
├── requirements.txt           # Python 依赖
├── README.md                  # 项目说明与复现实验命令
├── test_graph_shape.py        # numpy/PIL 形状验证脚本，不依赖 torch
├── scripts/
│   ├── run_lineplot_ucr.sh    # UCR line_plot baseline 脚本
│   └── run_lineplot_uea.sh    # UEA line_plot baseline 脚本
├── src/
│   ├── arguments.py           # 命令行参数定义
│   ├── datautils.py           # UCR/UEA/UCI HAR 数据读取与预处理
│   ├── embedding.py           # 模型 embedding 提取与多模态拼接前准备
│   ├── tivit.py               # TiViT、Activity Graph、ViT 前向
│   ├── classifier.py          # 分类器训练与测试
│   ├── analysis.py            # 表示分析功能
│   ├── mutual_knn.py          # mutual kNN alignment 工具
│   └── utils.py               # 随机种子、尺寸变换、结果写入等工具
├── img/
│   └── Plot.png               # README 方法示意图
├── assets/
│   └── methodology.svg        # 项目图示资源
└── data/
    └── UCI HAR Dataset/       # UCI HAR 数据目录示例
```

## Acknowledgements

- TiViT: Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers
- OpenCLIP / CLIP 视觉骨干
- Mantis 和 MOMENT 时间序列基础模型
- [UCR/UEA time series classification benchmark](https://www.timeseriesclassification.com/)
- [aeon 时间序列数据集工具](https://www.aeon-toolkit.org)
