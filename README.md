# Intelligent Meteorology

面向天池 CIKM2017 雷达降水数据的统一训练、评估与模型融合框架。

项目将 ConvLSTM、PredRNN、Earthformer 和 SimVP 适配到同一个单值降水回归任务：

```text
输入：[B, 15, 4, H, W] 雷达序列
输出：[B] 非负降水量
```

这里预测的是一个降水标量，不是未来雷达图。各模型共享相同的数据读取、划分、损失函数、评估指标和输出格式，便于公平比较与融合。

## 主要功能

- 支持 `convlstm`、`predrnn`、`earthformer`、`simvp`、`simvp_scalar`。
- 支持原始 TXT 数据和内存映射 NPY 数据。
- 自动划分训练集、验证集和独立留出集，避免用留出集选择模型或融合权重。
- 支持 MSE、SmoothL1 和强降水加权损失。
- 支持学习率预热、验证集驱动的降学习率、早停和 CUDA 混合精度。
- 输出逐轮指标、最佳权重、数据划分清单、验证/留出/测试预测和基线结果。
- 支持多模型权重搜索、实验比较、运行诊断、图表报告和提交格式校验。

## 项目结构

### 核心入口

| 文件 | 作用 |
|---|---|
| `train.py` | 训练模型，完成数据划分、验证、独立留出评估、保存最佳权重，并可直接预测 TestA/TestB。 |
| `predict.py` | 加载 `best.pt`，对指定 TXT 或 NPY 数据独立预测。 |
| `ensemble.py` | 在共同验证集上搜索多个模型的融合权重，并输出验证、留出和测试集融合结果。 |

### 数据与检查工具

| 文件 | 作用 |
|---|---|
| `convert_three_files_to_npy.py` | 将训练集、验证集、TestA、TestB 的 TXT 文件转换为可内存映射的 NPY bundle。 |
| `split_train_to_npy.py` | 将训练 TXT 按固定比例直接转换并拆分为开发集和最终评估集。 |
| `verify_npy.py` | 检查 NPY 的形状、数值范围、样本 ID，并可抽样对比原始 TXT。 |
| `data_audit.py` | 检查样本数量、重复 ID、标签分布、异常雷达值和答案样例格式。 |
| `validate_submission.py` | 检查提交预测的行数、有限性和非负性。 |

### 实验分析工具

| 文件 | 作用 |
|---|---|
| `compare_runs.py` | 汇总多个训练目录的最佳验证指标和留出集指标。 |
| `diagnose_runs.py` | 只读复查已有运行，重算指标并检查样本一致性、近常数输出和日志异常。 |
| `report_experiments.py` | 生成实验汇总 CSV、学习曲线、模型对比图和预测散点图。 |
| `visualize_prediction.py` | 展示单个样本的雷达回波、历史最大回波和模型注意力。 |
| `make_risk_report.py` | 按训练集降水分位数把预测转换为相对风险等级。 |
| `smoke_test.py` | 用随机小输入检查模型输出形状、注意力归一化、非负输出和反向梯度。 |

### 内部模块

| 文件 | 作用 |
|---|---|
| `src/tianchi_rainfall/data.py` | TXT/NPY 数据读取、行索引、裁剪、异常值处理和归一化。 |
| `src/tianchi_rainfall/models.py` | 四类模型适配器、多尺度注意力回归头和统一模型构建接口。 |
| `src/tianchi_rainfall/losses.py` | MSE、SmoothL1 和强降水加权损失。 |
| `src/tianchi_rainfall/metrics.py` | MAE、RMSE、R²、Pearson，以及 q50/q75/q90/q95 事件指标。 |
| `src/tianchi_rainfall/splits.py` | 可复现的分位数分层划分和顺序划分。 |
| `src/tianchi_rainfall/training_data.py` | 构建训练、验证、留出数据集并检查样本交叉。 |
| `src/tianchi_rainfall/optimization.py` | 学习率预热、自动降学习率和早停控制。 |

### 自动化脚本

| 文件 | 作用 |
|---|---|
| `scripts/bootstrap_vendor.sh` | 下载并固定第三方模型源码版本。 |
| `scripts/train_server.sh` | 在 CUDA 服务器上训练一个指定模型。 |
| `scripts/run_full_experiment.sh` | 从 TXT 数据执行审计、训练、比较、融合、报告和提交校验的完整流程。 |
| `scripts/run_npy_experiment.sh` | 从 NPY 数据训练一个或多个模型，并生成比较、融合和诊断结果。 |
| `scripts/train_on_two_parts.sh` | 使用 `split_train_to_npy.py` 生成的开发集和最终评估集启动训练。 |
| `scripts/compare_simvp_scalar.sh` | 对比 `simvp`/`simvp_scalar` 与 `heavy_rain`/`mse` 四种组合。 |

`tests/` 是核心单元测试，`.github/workflows/ci.yml` 会在推送和拉取请求时执行语法检查、Shell 检查和测试。

## 环境准备

建议使用 Python 3.11。服务器上应先安装与 CUDA 版本匹配的 PyTorch，再安装项目依赖：

```bash
python3 -m pip install -r requirements.txt
```

下载固定版本的第三方模型源码：

```bash
bash scripts/bootstrap_vendor.sh
```

默认下载到 `vendor/`。如果源码位于其他目录，在训练和预测时使用 `--vendor-root` 指定。Earthformer 可能还需要按照其上游项目说明补充依赖。

先做模型通路检查：

```bash
python3 smoke_test.py --vendor-root ./vendor
```

## 数据格式

原始 TXT 每行包含：

```text
sample_id,label,flattened_radar_values
```

每个样本包含 `15 × 4 × 101 × 101` 个雷达值。TestA/TestB 的标签通常为 `-1`，只能用于预测，不能用于本地精度评估。

推荐目录结构：

```text
/data/CIKM2017/
├── train/train.txt
├── testA/testA.txt
└── testB/testB.txt
```

代码不会修改原始 TXT。运行时生成的索引默认写入 `cache/indexes/`。

## 数据审计

正式训练前建议检查数据：

```bash
python3 data_audit.py \
  --train-file /data/CIKM2017/train/train.txt \
  --test-a-file /data/CIKM2017/testA/testA.txt \
  --test-b-file /data/CIKM2017/testB/testB.txt \
  --answer-a /data/CIKM2017/testA/answer_sample.csv \
  --answer-b /data/CIKM2017/testB/answer_sample.csv \
  --output-dir /data/experiments/data-audit
```

## 小样本验证

先用少量样本和一个 epoch 验证环境、数据和模型路径：

```bash
python3 train.py \
  --model convlstm \
  --train-file /data/CIKM2017/train/train.txt \
  --vendor-root ./vendor \
  --output-dir /data/experiments/smoke-convlstm \
  --max-samples 100 \
  --epochs 1 \
  --skip-test-prediction
```

## 正式训练

```bash
python3 train.py \
  --model convlstm \
  --train-file /data/CIKM2017/train/train.txt \
  --test-file /data/CIKM2017/testB/testB.txt \
  --test-a-file /data/CIKM2017/testA/testA.txt \
  --vendor-root ./vendor \
  --output-dir /data/experiments/convlstm \
  --crop-size 41 \
  --epochs 100 \
  --batch-size 8 \
  --device cuda \
  --amp
```

`--model` 可选：

- `convlstm`
- `predrnn`
- `earthformer`
- `simvp`
- `simvp_scalar`

训练数据默认被划分为训练集、验证集和独立留出集。验证集用于早停和选择最佳 checkpoint；留出集只用于最终评估。输出目录必须为空，避免覆盖已有实验。

常用参数：

- `--crop-size 41`：中心裁剪尺寸，也可使用完整的 `101`。
- `--loss heavy_rain`：默认强降水损失，也可选 `mse`、`smooth_l1`。
- `--split-strategy stratified`：默认按标签分位数分层划分。
- `--split-seed 42`：控制数据划分。
- `--seed 42`：控制模型初始化与训练随机性。
- `--head-variant full`：也可选 `no_attention`、`no_multiscale`、`avg_only` 做消融。
- `--amp`：在 CUDA 上启用混合精度。

## NPY 工作流

TXT 可以直接训练。数据较大或反复实验时，可先转换为 NPY：

```bash
python3 convert_three_files_to_npy.py \
  --train-file /data/CIKM2017/train/train.txt \
  --test-a-file /data/CIKM2017/testA/testA.txt \
  --test-b-file /data/CIKM2017/testB/testB.txt \
  --output-dir /data/CIKM2017/npy \
  --crop-size 41
```

每个数据集会生成四个相互配套的文件：

```text
*_inputs.npy
*_targets.npy
*_ids.npy
*_meta.json
```

训练时将 `*_inputs.npy` 传给 `--train-file`，其余三个文件必须保留在同一目录。转换后可验证：

```bash
python3 verify_npy.py \
  --inputs /data/CIKM2017/npy/train_crop41_inputs.npy \
  --source-txt /data/CIKM2017/train/train.txt \
  --output /data/CIKM2017/npy/verification.json
```

如果需要预先固定 80%/20% 开发与最终评估数据：

```bash
python3 split_train_to_npy.py \
  --train-file /data/CIKM2017/train/train.txt \
  --output-dir /data/CIKM2017/split \
  --eval-ratio 0.2 \
  --crop-size 41
```

## 训练输出

每个训练目录主要包含：

| 文件 | 内容 |
|---|---|
| `config.json` | 完整训练配置和降水分位数阈值。 |
| `metrics.jsonl` | 每个 epoch 的训练与验证指标。 |
| `best.pt` | 验证 RMSE 最优的权重和模型配置。 |
| `training_summary.json` | 完成 epoch、最佳 epoch 和早停信息。 |
| `split_manifest.csv` | 每个样本所属的数据划分。 |
| `baselines.json` | 验证集零值、训练均值和训练中位数基线。 |
| `validation_predictions.csv` | 验证集目标与预测，用于模型融合。 |
| `holdout_metrics.json` | 独立留出集最终指标。 |
| `holdout_predictions.csv` | 独立留出集目标与预测。 |
| `test_predictions.csv` | `--test-file` 对应的测试预测。 |
| `testA_predictions.csv` | `--test-a-file` 对应的 TestA 预测。 |

## 独立预测

```bash
python3 predict.py \
  --checkpoint /data/experiments/convlstm/best.pt \
  --data-file /data/CIKM2017/testB/testB.txt \
  --vendor-root ./vendor \
  --output-file /data/predictions/testB.csv
```

## 模型比较与融合

比较多个训练目录：

```bash
python3 compare_runs.py \
  /data/experiments/convlstm \
  /data/experiments/predrnn \
  /data/experiments/earthformer \
  --output /data/experiments/model_comparison.csv
```

融合模型：

```bash
python3 ensemble.py \
  /data/experiments/convlstm \
  /data/experiments/predrnn \
  /data/experiments/earthformer \
  --output-dir /data/experiments/ensemble
```

所有成员必须使用完全相同的验证集和留出集样本。融合权重只在验证集上选择，留出集只报告最终效果。

复查训练和融合结果：

```bash
python3 diagnose_runs.py \
  /data/experiments/convlstm \
  /data/experiments/predrnn \
  /data/experiments/earthformer \
  --ensemble-dir /data/experiments/ensemble \
  --output /data/experiments/diagnosis.json
```

## 一键实验

设置数据与输出路径：

```bash
export TRAIN_FILE=/data/CIKM2017/train/train.txt
export TEST_A_FILE=/data/CIKM2017/testA/testA.txt
export TEST_B_FILE=/data/CIKM2017/testB/testB.txt
export ANSWER_A=/data/CIKM2017/testA/answer_sample.csv
export ANSWER_B=/data/CIKM2017/testB/answer_sample.csv
export VENDOR_ROOT=$PWD/vendor
export OUTPUT_ROOT=/data/experiments
```

运行三个主模型、融合和报告：

```bash
bash scripts/run_full_experiment.sh core cuda
```

运行包含多随机种子、损失函数和结构消融的扩展实验：

```bash
bash scripts/run_full_experiment.sh paper cuda
```

## 测试

```bash
python3 -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
```

测试代码不会参与正式训练，但用于防止数据解析、划分、模型融合和报告逻辑在修改后出现回归。

## 注意事项

- 仓库不包含原始数据、训练权重和第三方模型源码。
- `vendor/`、`cache/`、`runs/`、`outputs/` 和模型权重已在 `.gitignore` 中排除。
- 默认 `crop-size=41` 只使用雷达图中心区域；正式实验前应与 `101` 做显存、耗时和效果对比。
- TestA/TestB 的 `-1` 标签不是实际降水量，不能用于计算模型精度。
- 第三方源码继续遵循各自仓库的许可证。
