# 天池 CIKM2017 降水预测核心代码

这是一个面向 CIKM2017 雷达降水数据的单值回归项目。输入为 15 个时间步、4 个高度层的雷达序列，输出为一个非负降水量：

```text
输入：[B, 15, 4, H, W]
输出：[B]
```

项目支持 ConvLSTM、PredRNN、Earthformer、SimVP 和 `simvp_scalar`，并统一使用多尺度注意力回归头。仓库仅保留数据加载、训练、预测和模型融合所需的核心实现。

## 文件说明

| 文件 | 作用 |
|---|---|
| `train.py` | 主训练入口。完成数据划分、模型训练、验证、留出集评估、早停、保存最佳权重，并可生成 TestA/TestB 预测。 |
| `predict.py` | 加载 `best.pt` 权重，对指定 TXT 或 NPY 数据独立预测。 |
| `ensemble.py` | 读取多个训练目录的预测结果，在验证集上搜索融合权重，并输出融合预测。 |
| `src/tianchi_rainfall/__init__.py` | Python 包入口，导出常用的数据集与模型构建接口。 |
| `src/tianchi_rainfall/data.py` | 读取 CIKM2017 TXT/NPY 数据，建立行索引，解析雷达序列并完成裁剪、清洗和归一化。 |
| `src/tianchi_rainfall/models.py` | 定义统一回归头，以及 ConvLSTM、PredRNN、Earthformer、SimVP 的适配器。 |
| `src/tianchi_rainfall/losses.py` | 定义 MSE、SmoothL1 和强降水加权损失。 |
| `src/tianchi_rainfall/metrics.py` | 计算 MAE、RMSE、R²、Pearson 相关系数和强降水分类指标。 |
| `src/tianchi_rainfall/splits.py` | 按降水量分位数生成可复现的训练集、验证集划分。 |
| `src/tianchi_rainfall/training_data.py` | 组织训练、验证和独立留出数据集，检查样本是否重复。 |
| `src/tianchi_rainfall/optimization.py` | 管理学习率预热、自动降学习率和早停。 |
| `scripts/bootstrap_vendor.sh` | 下载并固定四类第三方模型源码的版本。 |
| `requirements.txt` | Python 运行依赖。 |
| `.gitignore` | 排除数据、权重、缓存和运行结果。 |
| `LICENSE` | 项目许可证。 |

## 安装

先安装与你的 CUDA 环境匹配的 PyTorch，再安装其余依赖并下载第三方模型源码：

```bash
python3 -m pip install -r requirements.txt
bash scripts/bootstrap_vendor.sh
```

如第三方源码已在其他位置，可通过 `--vendor-root` 指定。Earthformer 可能还需要按其上游项目说明安装额外依赖。

## 训练

```bash
python3 train.py \
  --model convlstm \
  --train-file /data/CIKM2017_train/train.txt \
  --test-file /data/CIKM2017_testB/testB.txt \
  --test-a-file /data/CIKM2017_testA/testA.txt \
  --vendor-root ./vendor \
  --output-dir /data/experiments/convlstm \
  --amp
```

`--model` 可选：`convlstm`、`predrnn`、`earthformer`、`simvp`、`simvp_scalar`。

训练数据默认划分为训练集、验证集和独立留出集。模型按验证 RMSE 保存最佳权重，留出集不参与早停或融合权重选择。

常用参数：

- `--crop-size 41`：中心裁剪尺寸，也可使用 101。
- `--loss heavy_rain`：强降水加权损失，也可选 `mse` 或 `smooth_l1`。
- `--max-samples 100 --epochs 1`：正式训练前的小样本通路检查。
- `--device cuda`：指定设备；省略时自动选择 CUDA、MPS 或 CPU。
- `--amp`：在 CUDA 上启用混合精度。

主要输出：

- `best.pt`：最佳模型权重与训练配置。
- `metrics.jsonl`：每轮训练和验证指标。
- `validation_predictions.csv`：验证集预测，用于融合权重搜索。
- `holdout_metrics.json`、`holdout_predictions.csv`：独立留出评估。
- `test_predictions.csv`、`testA_predictions.csv`：测试集预测。
- `split_manifest.csv`：数据划分明细。

## 独立预测

```bash
python3 predict.py \
  --checkpoint /data/experiments/convlstm/best.pt \
  --data-file /data/CIKM2017_testB/testB.txt \
  --vendor-root ./vendor \
  --output-file /data/predictions/testB.csv
```

## 模型融合

```bash
python3 ensemble.py \
  /data/experiments/convlstm \
  /data/experiments/predrnn \
  /data/experiments/earthformer \
  --output-dir /data/experiments/ensemble
```

参与融合的训练目录必须使用相同的数据划分和样本顺序。
