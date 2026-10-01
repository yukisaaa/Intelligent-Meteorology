# 自动实验报告

共汇总 3 个运行。按验证集 RMSE，当前最佳运行是 `predrnn-full-heavy_rain-s42`，RMSE=3.7582。
最佳简单基线是 `train_median`，RMSE=2.4418。

## 输出图表

- `learning_curves.png`：验证集学习曲线
- `model_comparison.png`：模型 MAE/RMSE 对比
- `observed_vs_predicted.png`：真实值与预测值散点图
- `experiment_summary.csv`：完整可复查指标表
- `seed_summary.csv`：相同配置在不同模型随机种子下的均值与标准差

## 使用说明

正式报告应只引用全量训练结果。小样本 smoke test 只能证明代码通路可运行。
