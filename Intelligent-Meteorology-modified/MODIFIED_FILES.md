# 修改文件清单

本目录是基于 Intelligent-Meteorology 的修改版。核心改动文件名带有 `_modified` 后缀；原文件名入口保留为轻量兼容包装。

| 修改版文件 | 内容 |
|---|---|
| `train_modified.py` | 三段式数据划分、独立留出评估、TestA/TestB 预测、输出目录保护与 MSE 选项接入 |
| `src/tianchi_rainfall/losses_modified.py` | 增加普通 MSE 损失选项 |
| `ensemble_modified.py` | 在验证集选择融合权重，并在独立留出集报告融合指标；输出 TestA/TestB 融合预测 |
| `compare_runs_modified.py` | 比较表增加独立留出集指标 |
| `report_experiments_modified.py` | 实验报告加入留出集指标 |
| `smoke_test_modified.py` | 检查前向输出与反向梯度 |
| `scripts/run_full_experiment_modified.sh` | 环境变量路径配置、独立输出目录、MSE 对照和 TestA/TestB 全流程 |
| `scripts/train_server_modified.sh` | 单模型服务器运行配置与独立输出目录 |
| `.github/workflows/ci.yml` | CI 语法检查覆盖修改版服务器与完整实验脚本 |
| `README.md` | 修改版使用说明与服务器流程 |
| `DATASET_DECISION_modified.md` | 补充独立留出集评估说明 |
| `PAPER_BASIS_modified.md` | 补充 MSE 对照和独立留出集评估边界 |

原名入口（例如 `train.py`、`ensemble.py` 和 `scripts/train_server.sh`）会转到对应的修改版实现，方便兼容旧命令；推荐直接使用带 `_modified` 后缀的文件。
