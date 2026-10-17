# SimVP 使用说明

本项目直接通过 `train.py` 训练，预测使用 `predict.py`，融合使用 `ensemble.py`。这些文件都包含实际实现。

## 两种 SimVP 适配方式

- `--model simvp`：使用 OpenSTL 的 SimVP/gSTA 主干，再接单值降水回归头。
- `--model simvp_scalar`：使用全部时序特征直接回归降水量，详见 [SimVP单值回归适配说明.md](SimVP单值回归适配说明.md)。

第三方源码位置为 `vendor/external_models/SimVP`，模型适配器位于 `src/tianchi_rainfall/models.py`。输入为 `[B,15,4,H,W]`，输出为 `[B]` 的非负降水量。

## 训练控制

SimVP 默认学习率为 1e-4，最大轮数 100，预热 5 轮，允许早停的最少轮数 30，早停等待 20 轮。学习率停滞时减半，最低为 1e-6；降速后早停计数重新开始。

前 6 轮学习率为 `1e-5 → 3.25e-5 → 5.5e-5 → 7.75e-5 → 1e-4 → 1e-4`。预热后根据 validation RMSE 调整；holdout 不参与调度。显式指定 `--epochs 6` 时仍在第 6 轮结束。

这些是起始配置，完整训练后的 RMSE/CSI 需要通过实际评估确认。

## 服务器训练示例

在项目根目录执行，替换实际数据路径：

```bash
python train.py --model simvp \
  --train-file '/你的数据路径/train_crop41_inputs.npy' \
  --holdout-file '/你的数据路径/fusion_eval_crop41_inputs.npy' \
  --vendor-root ./vendor \
  --output-dir ./experiments/simvp-run01 \
  --epochs 100 --batch-size 8 --crop-size 41 \
  --learning-rate 1e-4 --warmup-epochs 5 --min-epochs 30 \
  --early-stopping-patience 20 --device cuda --amp
```

输出目录必须新建。TestA/TestB 未提供时可以仅训练并评估本地 holdout。

也可以使用两部分数据脚本：

```bash
SPLIT_DIR='/你的数据路径/npy_split_80_20' DEVICE=cuda \
  bash scripts/train_on_two_parts.sh simvp
```

`all` 仍执行原来的 ConvLSTM、PredRNN、Earthformer 三个主模型。第四个模型可单独训练，随后将四个运行目录传给 `ensemble.py`。融合权重只在共同验证集上选择，最终成绩查看独立留出集。

## 已有权重

本次文件名整理没有改变网络结构、参数名称或 checkpoint 格式。已有权重可继续通过 `predict.py` 使用；已有验证及留出预测文件可直接用于融合，无需因整理入口重训。
