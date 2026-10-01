# 天池 CIKM2017 降水预测统一框架
接入三个公开实现：

- ConvLSTM: `ndrplz/ConvLSTM_pytorch`
- PredRNN: `thuml/predrnn-pytorch`
- Earthformer: `amazon-science/earth-forecasting-transformer`

三种模型共享同一任务定义：

```text
输入:  [B, 15, 4, H, W]
输出:  [B]，非负降水值
```

这里是针对 CIKM2017 单值标签的模型适配，不等同于原论文的未来雷达帧生成任务。

论文依据、适配边界以及哪些方法没有照搬，见 `PAPER_BASIS.md`。

当前版本加入了以下论文启发的改进：

- 基于训练集 q90/q95 的强降水加权 SmoothL1；
- 对降水欠预测增加非对称惩罚；
- `1×3`、`3×1`、`3×3` 多尺度卷积；
- 全局平均、全局最大和空间注意力池化；
- q50/q75/q90/q95 下的 CSI、POD、FAR、强降水 MAE；
- 根据共同验证集自动搜索 ConvLSTM、PredRNN、Earthformer 融合权重；
- 标签分位数分层划分、逐样本划分清单和固定随机种子；
- 零值、训练均值、训练中位数三种基线；
- 早停、学习率自动衰减和可选 CUDA 混合精度；
- 一键数据审计、消融实验、图表报告及提交文件校验。

## 1. 下载模型源码

```bash
bash scripts/bootstrap_vendor.sh
python -m pip install -r requirements.txt
```

如源码已下载到其他目录，可以在训练时使用 `--vendor-root` 指定。

## 2. 小样本验证

```bash
python smoke_test.py --vendor-root "#TODO"
```

完整数据审计：

```bash
python data_audit.py \
  --train-file "#TODO" \
  --test-a-file "#TODO" \
  --test-b-file "#TODO" \
  --answer-a "#TODO" \
  --answer-b "#TODO" \
  --output-dir "#TODO"
```

审计不会改写数据；检测到的负雷达哨兵和非有限值只会在加载到内存后被安全处理。

## 服务器数据路径

代码不写死本机路径。请将下面所有 `#TODO` 替换为服务器实际路径：

```text
#TODO/
├── CIKM2017_train/train.txt
├── CIKM2017_testA/testA.txt
└── CIKM2017_testB/testB.txt
```

## 3. 训练

ConvLSTM：

```bash
python train.py \
  --model convlstm \
  --train-file "#TODO" \
  --test-file "#TODO" \
  --vendor-root "#TODO" \
  --output-dir "#TODO"
```

默认采用标签分位数分层划分。`--split-seed` 默认固定为 42，与模型初始化用的 `--seed` 分离，因此多随机种子实验仍使用完全相同的验证样本。训练目录会额外生成 `split_manifest.csv` 和 `baselines.json`，便于证明划分可复现并与简单基线比较。CUDA服务器建议加 `--amp`。

PredRNN：

```bash
python train.py \
  --model predrnn \
  --train-file "#TODO" \
  --test-file "#TODO" \
  --vendor-root "#TODO" \
  --output-dir "#TODO"
```

Earthformer：

```bash
python train.py \
  --model earthformer \
  --train-file "#TODO" \
  --test-file "#TODO" \
  --vendor-root "#TODO" \
  --output-dir "#TODO" \
  --batch-size 1
```

正式服务器训练建议先使用 `--max-samples 100 --epochs 1` 验证，再去掉 `--max-samples`。

## 4. 输出

每个运行目录包含：

- `config.json`: 完整运行配置
- `metrics.jsonl`: 每轮训练与验证指标
- `best.pt`: 验证集 RMSE 最优权重
- `validation_predictions.csv`: 最优轮次的验证集预测，用于模型融合
- `test_predictions.csv`: 与测试集行顺序一致的单列预测
- `split_manifest.csv`: 每条训练样本对应的训练/验证划分
- `baselines.json`: 零值、训练均值和训练中位数基线

升级后的多尺度注意力回归头改变了参数名称和维度，旧版 `best.pt` 不能直接加载，需重新训练。

默认中心裁剪为 `41×41`，与已有 ConvLSTM 实验保持一致。若服务器显存允许，可设 `--crop-size 101` 使用完整雷达区域。

## 5. 独立预测与比较

```bash
python predict.py \
  --checkpoint "#TODO" \
  --data-file "#TODO" \
  --vendor-root "#TODO" \
  --output-file "#TODO"

python compare_runs.py "#TODO" "#TODO" "#TODO" --output "#TODO"
```

融合三个模型：

```bash
python ensemble.py "#TODO" "#TODO" "#TODO" --output-dir "#TODO"
```

前三个 `#TODO` 分别填写 ConvLSTM、PredRNN、Earthformer 的运行目录。输出包含：

- `ensemble.json`: 验证集选择的三个权重和融合指标；
- `test_predictions.csv`: 融合后的测试集降水预测。

生成样本解释图：

```bash
python visualize_prediction.py \
  --checkpoint "#TODO" \
  --data-file "#TODO" \
  --vendor-root "#TODO" \
  --output-file "#TODO"
```

该图片展示历史雷达、历史最大回波和模型注意力区域。由于本数据集只有标量标签，注意力热力图不是未来降水空间预测图。

`examples/attention-smoke.png` 是用 6 个样本、1 个 epoch 做通路测试得到的格式示例，只证明可视化程序可运行，不代表模型精度。

预测 Test B 时显式指定：

```bash
python predict.py \
  --checkpoint "#TODO" \
  --data-file "#TODO" \
  --vendor-root "#TODO" \
  --output-file "#TODO"
```

也可以直接使用服务器脚本：

```bash
bash scripts/train_server.sh convlstm
bash scripts/train_server.sh predrnn
bash scripts/train_server.sh earthformer
```

## 6. 一键正式实验

先填写 `scripts/run_full_experiment.sh` 顶部全部 `#TODO`，然后执行：

```bash
# 三个主模型、融合、图表、风险分级和提交校验
bash scripts/run_full_experiment.sh core cuda

# 在 core 基础上增加三随机种子与损失/注意力/多尺度消融
bash scripts/run_full_experiment.sh paper cuda
```

`paper` 模式会自动运行：

- ConvLSTM、PredRNN、Earthformer 主实验；
- ConvLSTM 三个随机种子；
- SmoothL1 与强降水损失对比；
- 去注意力、去多尺度、仅平均池化消融；
- 自动生成学习曲线、误差柱状图、真实值—预测值散点图；
- 三模型融合、2000行提交格式校验和数据集相对风险分级。

实验报告也可以单独生成：

```bash
python report_experiments.py "#TODO" "#TODO" "#TODO" --output-dir "#TODO"
python validate_submission.py --prediction-file "#TODO" --answer-sample "#TODO"
```

数据审计结果保存在 `examples/data-audit-current/`，可执行 Notebook 位于 `notebooks/data_quality_audit.ipynb`。外部小型数据集的取舍见 `DATASET_DECISION.md`。

公开仓库及本次验证所用提交记录在 `VENDOR_COMMITS.md` 中。第三方代码继续遵循各自仓库许可证，本项目的适配代码不会删除或改写其许可证。

## 7. GitHub 发布说明

- 数据集、训练权重、缓存和第三方源码均被 `.gitignore` 排除，不要上传 CIKM2017 原始文件；
- `scripts/bootstrap_vendor.sh` 会检出并核验 `VENDOR_COMMITS.md` 中记录的固定提交；
- GitHub Actions 会执行编译检查、Shell语法检查和核心单元测试；
- `examples/` 中的结果均为 smoke test，只证明代码通路可运行；
- 当前仓库未附带项目级开源许可证；公开发布前是否添加许可证由仓库所有者决定。
