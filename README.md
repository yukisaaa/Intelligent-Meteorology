# 天池 CIKM2017 降水预测统一框架

本项目保持原始天池数据文件不变，只在项目目录中生成索引、日志、模型权重和预测文件。

当前接入三个公开实现：

- ConvLSTM: `ndrplz/ConvLSTM_pytorch`
- PredRNN: `thuml/predrnn-pytorch`
- Earthformer: `amazon-science/earth-forecasting-transformer`

三种模型共享同一任务定义：

```text
输入:  [B, 15, 4, H, W]
输出:  [B]，非负降水值
```

这里是针对 CIKM2017 单值标签的模型适配，不等同于原论文的未来雷达帧生成任务。

## 1. 下载模型源码

```bash
bash scripts/bootstrap_vendor.sh
python -m pip install -r requirements.txt
```

如源码已下载到其他目录，可以在训练时使用 `--vendor-root` 指定。

## 2. 小样本验证

```bash
python smoke_test.py --vendor-root vendor
```

## 3. 训练

ConvLSTM：

```bash
python train.py \
  --model convlstm \
  --train-file "/path/to/CIKM2017_train/train.txt" \
  --test-file "/path/to/CIKM2017_testA/testA.txt" \
  --vendor-root vendor \
  --output-dir runs/convlstm
```

PredRNN：

```bash
python train.py \
  --model predrnn \
  --train-file "/path/to/CIKM2017_train/train.txt" \
  --test-file "/path/to/CIKM2017_testA/testA.txt" \
  --vendor-root vendor \
  --output-dir runs/predrnn
```

Earthformer：

```bash
python train.py \
  --model earthformer \
  --train-file "/path/to/CIKM2017_train/train.txt" \
  --test-file "/path/to/CIKM2017_testA/testA.txt" \
  --vendor-root vendor \
  --output-dir runs/earthformer \
  --batch-size 1
```

正式服务器训练建议先使用 `--max-samples 100 --epochs 1` 验证，再去掉 `--max-samples`。

## 4. 输出

每个运行目录包含：

- `config.json`: 完整运行配置
- `metrics.jsonl`: 每轮训练与验证指标
- `best.pt`: 验证集 RMSE 最优权重
- `test_predictions.csv`: 与测试集行顺序一致的单列预测

默认中心裁剪为 `41×41`，与已有 ConvLSTM 实验保持一致。若服务器显存允许，可设 `--crop-size 101` 使用完整雷达区域。

## 5. 独立预测与比较

```bash
python predict.py \
  --checkpoint runs/convlstm/best.pt \
  --data-file "/path/to/CIKM2017_testA/testA.txt" \
  --vendor-root vendor \
  --output-file runs/convlstm/test_predictions.csv

python compare_runs.py runs/convlstm runs/predrnn runs/earthformer
```

公开仓库及本次验证所用提交记录在 `VENDOR_COMMITS.md` 中。第三方代码继续遵循各自仓库许可证，本项目的适配代码不会删除或改写其许可证。

