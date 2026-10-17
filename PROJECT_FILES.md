# 项目文件说明

项目只保留一套标准文件名。下表中的程序文件包含实际实现，内部模块导入、训练脚本、测试和文档已统一使用这些名称。

| 文件 | 内容 |
|---|---|
| `train.py` | 三段式数据划分、独立留出评估、TestA/TestB 预测、输出目录保护与 MSE 选项接入 |
| `src/tianchi_rainfall/losses.py` | 增加普通 MSE 损失选项 |
| `ensemble.py` | 支持三模型、四模型等多个模型融合；在验证集选权重，在留出集评估；可输出 TestA/TestB 融合预测 |
| `compare_runs.py` | 比较表增加独立留出集指标 |
| `report_experiments.py` | 实验报告加入留出集指标 |
| `smoke_test.py` | 检查前向输出与反向梯度 |
| `scripts/run_full_experiment.sh` | 环境变量路径配置、独立输出目录、MSE 对照和 TestA/TestB 全流程 |
| `scripts/train_server.sh` | 单模型服务器运行配置与独立输出目录 |
| `.github/workflows/ci.yml` | Python、Shell 语法检查和单元测试 |
| `README.md` | 使用说明与服务器流程 |
| `DATASET_DECISION.md` | 补充独立留出集评估说明 |
| `PAPER_BASIS.md` | 补充 MSE 对照和独立留出集评估边界 |

`vendor/` 保存第三方模型源码，不属于本次文件名整理范围。训练结果与模型权重的格式保持不变。`scripts/run_npy_experiment.sh all` 与完整实验脚本仍执行原来的三个主模型；已有四个模型的结果可直接传给 `ensemble.py` 融合。
