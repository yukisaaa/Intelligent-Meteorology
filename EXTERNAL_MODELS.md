# 外部模型与比赛适配说明

当前 `main` 使用用户提供的本机统一版代码。它使用过去 15 个时刻、4 个高度的雷达回波预测单个非负降水量，输入 `[B,15,4,H,W]`，输出 `[B]`。

## 当前支持的模型

| 模型参数 | 第三方源码位置 |
|---|---|
| `convlstm` | `vendor/ConvLSTM_pytorch/` |
| `predrnn` | `vendor/predrnn-pytorch/` |
| `earthformer` | `vendor/earth-forecasting-transformer/` |
| `simvp`、`simvp_scalar` | `vendor/external_models/SimVP/`，来自 OpenSTL |

训练实际实现直接位于 `train.py`，模型适配器位于 `src/tianchi_rainfall/models.py`。`ensemble.py` 支持三个、四个及更多模型的非负加权融合，权重之和为 1；权重仅在共同验证集上选择，独立留出集用于最终评估。

## 模型范围与历史版本

当前统一版只提供上表中的模型参数。TAU、FourCastLSTM、CasCast 和 PostRainBench 等额外适配可在历史提交 `f2c19c1` 中查阅，未包含在当前实现中。

## 第三方源码与依赖

第三方源码、数据集和模型权重不随本次提交上传。已有 vendor 可通过 `--vendor-root` 指定；固定版本记录见 `VENDOR_COMMITS.md`。先在服务器配置好 PyTorch/CUDA，再按 `requirements.txt` 安装其余依赖。各第三方实现继续遵循自己的许可证。
