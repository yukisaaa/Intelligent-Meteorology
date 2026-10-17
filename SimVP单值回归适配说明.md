# SimVP 时序特征直接回归降水量

新增模型名称为 **`simvp_scalar`**，原 `simvp` 保留作基线和加载旧权重。两者都用于 CIKM2017 的单值降水量任务；新增版本是本项目适配，不是原始 SimVP 视频预测模型的原样评测。

## 结构

```text
原 simvp：
15 帧 × 4 高度雷达 → SimVP 编码器 → gSTA 时序模块
→ 图像解码器 → 最后一个输出时刻的 4 通道图 → 回归头 → 降水量

新 simvp_scalar：
15 帧 × 4 高度雷达 → 同规格 SimVP 编码器 → 同规格 gSTA 时序模块
→ 15 个时刻的 16 通道特征 → 可学习的时间权重聚合
→ 多尺度空间回归头 → Softplus → 非负降水量
```

旧模型也会通过 gSTA 混合所有输入时刻，不能理解为它只用了最后一帧输入。新版本的变化是：不再经过图像解码器，也不只选取一个输出时刻，而是让回归头直接读取聚合后的时序特征。

- 输入保持 `[B,15,4,H,W]`，输出保持 `[B]`；标签单位与原数据一致。
- 41×41 输入补齐到 48×48，特征分辨率为 24×24。按原始有效区域计算覆盖比例掩码，时间描述符及空间平均/最大/注意力池化排除完全补齐的格点，并按覆盖比例处理平均和注意力池化。
- 时间权重按样本学习，15 个权重经 softmax 归一化。它们描述内部特征的聚合，不是输入时刻的因果贡献证明。
- 新模型不注册、训练或保存图像解码器参数。官方 vendor 源码和固定提交保持不变，改动位于项目适配层。
- 可视化返回 H×W 的归一化注意力图，由低分辨率权重映射而来，不能当作未来降雨空间预测图。

## 训练新版本

在项目目录运行，替换数据路径。默认 vendor 为本项目的 vendor：

```bash
python3 train.py --model simvp_scalar \
  --train-file '/你的数据路径/train_crop41_inputs.npy' \
  --output-dir ./experiments/simvp-scalar-mse-v1 \
  --loss mse --epochs 100 --batch-size 8 --crop-size 41 \
  --learning-rate 1e-4 --warmup-epochs 5 --min-epochs 30 \
  --early-stopping-patience 20 --device cuda --amp
```

新版本沿用 SimVP 的学习率 1e-4、5 轮预热、最多 100 轮、至少 30 轮后才允许早停、patience=20。上面显式选择 MSE，使训练目标与 RMSE 一致；训练入口的默认损失仍为 heavy_rain，避免自动同时改变结构与损失。

已有单模型脚本也支持：

```bash
TRAIN_FILE='/你的数据路径/train_crop41_inputs.npy' DEVICE=cuda \
  bash scripts/run_npy_experiment.sh simvp_scalar
```

该单模型脚本使用默认 heavy_rain 损失。每次使用新的输出目录，避免混入旧实验。

## 四组对照

新增脚本同时运行以下组合：

| 结构 | 损失 |
|---|---|
| simvp，原解码后回归 | heavy_rain |
| simvp_scalar，直接特征回归 | heavy_rain |
| simvp，原解码后回归 | mse |
| simvp_scalar，直接特征回归 | mse |

```bash
TRAIN_FILE='/你的数据路径/train_crop41_inputs.npy' \
OUTPUT_ROOT='./experiments/simvp-scalar-comparison-v1' \
DEVICE=cuda bash scripts/compare_simvp_scalar.sh
```

四组均使用相同输入文件、seed=42、split_seed=42、学习率、batch size 和最大轮数；同损失下比较结构，同结构下比较损失。生成 comparison.csv、diagnosis.json 和逐样本划分清单。可通过 EPOCHS、BATCH_SIZE、MAX_SAMPLES、CROP_SIZE、VALIDATION_FILE、HOLDOUT_FILE、VENDOR_ROOT、PYTHON_BIN 环境变量覆盖相应配置。

例如先检查流程：

```bash
TRAIN_FILE='/你的数据路径/train_crop41_inputs.npy' \
OUTPUT_ROOT='./experiments/simvp-scalar-smoke' \
EPOCHS=2 MAX_SAMPLES=24 BATCH_SIZE=2 DEVICE=cuda \
  bash scripts/compare_simvp_scalar.sh
```

学习率由 validation RMSE 调度，最佳权重也由 validation RMSE 选择；holdout 不参与调度、早停或结构参数选择。完整训练后再评估泛化和强降水 CSI，短程测试不能用于宣布哪一结构更好。

## 权重与预测

- 原 `simvp` 的权重仍用 `model_name=simvp` 重建；已验证此前真实保存的旧权重可严格加载，预测与原记录在小数保存误差内一致。
- 新权重记录 `model_name=simvp_scalar`、`adapter_version=temporal_features_v1`，predict.py 根据权重中的模型名称自动选择结构；visualize_prediction.py 使用同一模型工厂。
- 新旧结构的参数不同，新结构需要重新训练，不能直接把旧 SimVP 权重当成完整的新结构权重加载。

```bash
python3 predict.py \
  --checkpoint ./experiments/simvp-scalar-mse-v1/best.pt \
  --data-file '/你的数据路径/testB_crop41_inputs.npy' \
  --output-file ./experiments/simvp-scalar-mse-v1/testB_predictions.csv \
  --device cuda
```

## 验证范围

- 22 项单元测试通过，其中新增测试核查全部 15 个输入时刻都有梯度、编码器/gSTA/时间聚合/回归头有有效梯度、模型没有解码器参数、时间权重与空间注意力归一化、有效区域掩码，以及新旧结构的权重重载。
- 41×41 输入下，full、no_attention、no_multiscale、avg_only 四种回归头均完成前向/反向检查。
- 四组配置分别用真实数据前 24 条、18/4/2 划分、各训练 2 轮，完成 holdout、对比表和诊断；新模型权重已通过独立预测验证。
- 小样本可学习性检查：对 4 条真实样本关闭 dropout，使用 MSE、诊断用学习率 1e-3，训练 200 步；训练 RMSE 从约 2.498 降至 0.131，低于该 4 条样本的常数均值基线 0.944。这只证明可以拟合这些样本，**不是 validation/holdout 精度，也不是正式学习率选择的依据**。
- 尚未执行完整数据 100 轮训练，不据此保证最终 RMSE 或强降水 CSI 提升。
