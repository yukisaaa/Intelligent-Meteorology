> 新增物理 80%/20% 数据划分脚本：见 [两部分数据划分与融合评估.md](两部分数据划分与融合评估.md)。最终评估文件通过 `--holdout-file` 使用。

> 新增 `simvp_scalar`：从全部时序特征直接回归降水量，保留 `simvp` 作基线。训练与四组对照见 [SimVP单值回归适配说明.md](SimVP单值回归适配说明.md)。

> **2026-10-17 统一版**：所有程序直接使用标准文件名，例如 `train.py`、`ensemble.py`；这些文件包含实际实现。项目只保留一套代码，融合支持三个或四个模型。文件职责见 [PROJECT_FILES.md](PROJECT_FILES.md)，服务器迁移见 [统一部署说明.md](统一部署说明.md)。

> 修改版：本目录是在 Intelligent-Meteorology 原项目基础上的本地适配与改进版本，不是原始仓库的原样副本。

# 天池 CIKM2017 降水预测统一框架
接入四类公开模型实现：

- ConvLSTM: `ndrplz/ConvLSTM_pytorch`
- PredRNN: `thuml/predrnn-pytorch`
- Earthformer: `amazon-science/earth-forecasting-transformer`
- SimVP: OpenSTL，支持 `simvp` 和直接回归降水量的 `simvp_scalar`

这些模型共享同一任务定义：

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
- 根据共同验证集自动搜索多个模型的融合权重，支持四模型融合；
- 标签分位数分层划分、逐样本划分清单和固定随机种子；
- 零值、训练均值、训练中位数三种基线；
- 早停、学习率自动衰减和可选 CUDA 混合精度；
- 一键数据审计、消融实验、图表报告及提交文件校验。

## 1. 服务器环境与模型源码

```bash
bash scripts/bootstrap_vendor.sh
python -m pip install -r requirements.txt
```

如源码已下载到其他目录，可以在训练时使用 `--vendor-root` 指定。

服务器上先安装与你的 CUDA 版本匹配的 PyTorch，再安装本项目依赖。`bootstrap_vendor.sh` 只检出固定版本的第三方源码，不负责安装 Earthformer 的额外运行依赖；若 Earthformer 导入失败，按该 vendor 提交中的依赖说明补齐环境，不要让 pip 擅自替换服务器上的 CUDA PyTorch。

先验证前向和反向传播：

```bash
python smoke_test.py --vendor-root /path/to/vendor
```

这个 smoke test 会对每个模型检查输出形状、注意力归一化、非负输出和梯度有限性。若某个 vendor 依赖没装好，会在这里尽早暴露。

## 2. 小样本训练验证

```bash
python train.py \
  --model convlstm \
  --train-file /data/CIKM2017_train/train.txt \
  --test-file /data/CIKM2017_testB/testB.txt \
  --vendor-root /path/to/vendor \
  --output-dir /data/experiments/smoke-convlstm \
  --max-samples 100 --epochs 1 --skip-test-prediction
```

先完成前一节的 `smoke_test.py`，再执行此小样本训练。

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

每次训练会将样本划分为训练集、早停/融合权重选择验证集，以及独立留出集（默认比例约 72%/18%/10%）。模型按验证 RMSE 选 checkpoint；融合权重在验证集选择，最终融合效果另报独立留出集指标。留出集不会用于早停、选权重或训练。

若主要目标是比赛 RMSE，建议把默认强降水加权损失和普通 MSE 对照：

```bash
python train.py \
  --model convlstm \
  --train-file /data/CIKM2017_train/train.txt \
  --test-file /data/CIKM2017_testB/testB.txt \
  --test-a-file /data/CIKM2017_testA/testA.txt \
  --vendor-root /path/to/vendor \
  --output-dir /data/experiments/convlstm-mse \
  --loss mse
```

TestA 和 TestB 都可以预测：训练时用 `--test-file` 传 TestB，并用 `--test-a-file` 传 TestA；正式实验脚本会输出两个测试集的预测。测试标签为 -1，仍不可用于本地精度评估。

默认采用标签分位数分层。`--split-seed` 默认固定为 42，与模型初始化用的 `--seed` 分离，因此多随机种子实验仍使用完全相同的训练/验证/留出样本。训练目录会生成 `split_manifest.csv`、`baselines.json` 和 `holdout_baselines.json`。输出目录非空时训练会停止，避免覆盖旧结果；请每次使用新目录。CUDA服务器建议加 `--amp`。

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
- `holdout_metrics.json` / `holdout_predictions.csv`: 独立留出集最终评估，不参与早停和模型融合权重选择
- `test_predictions.csv`: 与 `--test-file`（通常 TestB）行顺序一致的单列预测
- `testA_predictions.csv`: 可选的 TestA 单列预测
- `split_manifest.csv`: 每条训练样本对应的训练/验证划分
- `baselines.json` / `holdout_baselines.json`: 零值、训练均值和训练中位数基线

升级后的多尺度注意力回归头改变了参数名称和维度，旧版 `best.pt` 不能直接加载，需重新训练。

默认中心裁剪为 `41×41`，只保留 101×101 雷达图约 16.5% 的像素区域。应先以 `--crop-size 41` 和 `--crop-size 101` 各跑小样本对照，并比较验证/留出 RMSE、显存和耗时，再决定正式配置。完整实验脚本和服务器脚本均可通过 `CROP_SIZE=101` 设置尺寸；Earthformer 的输入会补齐到其网络要求的空间大小。

## 5. 独立预测与比较

```bash
python predict.py \
  --checkpoint "#TODO" \
  --data-file "#TODO" \
  --vendor-root "#TODO" \
  --output-file "#TODO"

python compare_runs.py "#TODO" "#TODO" "#TODO" --output "#TODO"
```

融合四个模型（也可以只传三个目录）：

```bash
python ensemble.py "#TODO" "#TODO" "#TODO" "#TODO" --output-dir "#TODO"
```

前四个 `#TODO` 分别填写 ConvLSTM、PredRNN、Earthformer、SimVP（或 simvp_scalar）的运行目录。最后一个为新的融合输出目录。所有运行必须使用相同的验证和留出样本及顺序。输出包含：

- `ensemble.json`: 验证集选择的各模型权重、验证集选择指标和独立留出集最终指标；
- `validation_predictions.csv` / `holdout_predictions.csv`: 融合后的验证集及留出集逐样本预测；
- `test_predictions.csv`: 融合后的测试集降水预测。
- `testA_predictions.csv`: 所有模型都提供 TestA 预测时生成的 TestA 融合结果。

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

也可以直接使用单模型服务器脚本。路径可以通过环境变量指定，输出会自动放入带运行时间戳的新目录，避免覆盖旧实验：

```bash
TRAIN_FILE=/data/CIKM2017_train/train.txt \
TEST_A_FILE=/data/CIKM2017_testA/testA.txt \
TEST_B_FILE=/data/CIKM2017_testB/testB.txt \
VENDOR_ROOT=/path/to/vendor OUTPUT_ROOT=/data/experiments \
CROP_SIZE=41 bash scripts/train_server.sh predrnn
```

## 6. 一键正式实验

完整实验脚本也接受环境变量路径，并按时间戳创建独立输出目录：

```bash
export TRAIN_FILE=/data/CIKM2017_train/train.txt
export TEST_A_FILE=/data/CIKM2017_testA/testA.txt
export TEST_B_FILE=/data/CIKM2017_testB/testB.txt
export ANSWER_A=/data/CIKM2017_testA/answer_sample.csv
export ANSWER_B=/data/CIKM2017_testB/answer_sample.csv
export VENDOR_ROOT=/path/to/vendor
export OUTPUT_ROOT=/data/experiments
export CROP_SIZE=41

# 三个主模型、融合、图表、风险分级和 A/B 提交格式校验
bash scripts/run_full_experiment.sh core cuda

# 可选：增加多随机种子及 MSE/SmoothL1/强降水损失和结构消融
bash scripts/run_full_experiment.sh paper cuda
```

`paper` 模式会自动运行：

- ConvLSTM、PredRNN、Earthformer 主实验；
- ConvLSTM 三个随机种子；
- MSE、SmoothL1 与强降水损失对比；
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
- 根目录已有项目级 MIT License；第三方 vendor 代码仍按各自许可证管理。
