# 论文依据与代码对应关系

## 本次实现

| 文献中的问题或方向 | 本项目实现 | 代码位置 |
|---|---|---|
| 普通 MSE/MAE 容易平滑强回波并低估极端降水 | 用训练集 q90/q95 提高强降水样本权重，并加入欠预测惩罚 | `src/tianchi_rainfall/losses_modified.py` |
| 局部强回波需要多尺度、空间注意力特征 | `1×3`、`3×1`、`3×3` 分支，加平均池化、最大池化、注意力池化 | `src/tianchi_rainfall/models.py` |
| 单一模型在不同天气过程中的误差不同 | 在共同验证集搜索非负、和为 1 的三模型权重 | `ensemble_modified.py` |
| 只用全局 MAE/RMSE 会掩盖强降水表现 | 在训练集 q50/q75/q90/q95 阈值下报告 CSI、POD、FAR、强降水 MAE、欠预测率 | `src/tianchi_rainfall/metrics.py` |

## 适配边界

- 综述中的 DQWL、BDWLoss 等原本主要面向雷达帧预测。本项目把其“强降水加权、抑制欠预测”的思想改写为标量回归损失，并非逐公式复现。
- 当前融合是验证集加权平均，是稳定、可解释的第一阶段集成，不是需要额外训练的 MoE 门控网络。
- 融合权重只在验证集选择；独立留出集用于报告最终泛化指标，不参与选权重。标签分层随机留出仍不能代替按天气过程/时间分组验证。
- 普通 MSE 作为比赛 RMSE 目标的直接训练对照；强降水加权损失的结果应同时报告 RMSE 和阈值化强降水指标。
- CIKM2017 当前文件提供历史雷达序列和一个降雨标量标签，因此 SSIM、LPIPS、PSD、未来雷达图像等帧预测指标或结果不适用。
- PredRNN 的反向计划采样和逐帧损失需要未来帧监督；在本任务中只保留时空记忆编码器，后接统一的降水回归头。

## 主要参考

1. *Review on deep learning quantitative precipitation nowcasting: Advances, challenges, and future directions*：综述中的损失函数、注意力、多尺度、模型集成与阈值化评估建议。
2. Shi et al., *Deep Learning for Precipitation Nowcasting: A Benchmark and a New Model*：TrajGRU、HKO-7 基准以及按降水强度加权的评估思想。
3. Zhao et al., *Mamba-UNet: An Efficient Mamba-UNet for Precipitation Nowcasting*：动态分位数加权损失与时空注意思想。
4. Liu et al., *MMF-RNN: A Multiscale Multi-Level Feature Fusion Recurrent Neural Network for Precipitation Nowcasting*：多尺度特征融合与强降水加权损失思想。

代码中的注释和本文档使用“inspired by/启发”表述，避免把任务适配误写成对上述论文完整算法的复现。
