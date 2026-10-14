# 外部模型与比赛适配说明

本项目的任务不是预测未来雷达图，而是使用过去 15 个时刻、4 个高度的雷达回波，预测未来 1～2 小时累计降雨量。因此，外部仓库只作为时空特征编码器使用，统一接入本项目的单值非负回归头。

## 已下载仓库

代码位于 `vendor/external_models/`，下载时的浅克隆版本如下：

| 仓库 | 用途 | 当前状态 |
|---|---|---|
| `Fourier-NowCasting-LSTM` | FourCastLSTM | 已适配为 `fourcastlstm` |
| `OpenSTL` | SimVP-V2/gSTA、TAU | 已适配为 `simvp`、`tau` |
| `PostRainBench` | Swin-Unet 主干 + 标量回归头 | 已适配为 `postrainbench`（不是原始 NWP 多任务目标） |
| `BlockGPT` | 生成式降雨图预测 | 仍为参考代码，暂不接入单值任务 |
| `CasCast` | 确定性 EarthFormer 主干 + 标量回归头 | 已适配为 `cascast`；不包含原始 SEVIR 扩散三阶段 |
| `PFST-LSTM` | CIKM 降雨预测参考实现 | 已下载，保留作对照参考 |

## 为什么不是所有仓库都直接接入

`BlockGPT` 的训练目标是未来多帧 token/雷达图，需要 VQGAN tokenizer 和预测器 checkpoint；当前比赛标签只有一个累计降雨值，因此仍不作为主训练模型。

`CasCast` 的完整版本是 SEVIR 多阶段扩散系统。本项目新增的 `cascast` 只复用了其中的确定性 EarthFormer 主干，并在输入侧把 4 个雷达通道投影成 1 个通道，在输出侧增加天池标量回归头；它可以参加本项目训练，但不能称为完整的原始 CasCast。

`PostRainBench` 原本使用 NWP/AWS 变量和空间后处理标签。本项目新增的 `postrainbench` 使用其 Swin-Unet 主干，将 15 帧、4 通道雷达压缩成 3 个时间组后形成 12 通道输入，再将回归图池化成一个天池标量；原始分类头和 NWP 多任务损失没有使用。

`PFST-LSTM` 原仓库存在大量 `.cuda()`、固定 batch size 和固定 5→10 帧解码假设，直接塞入当前 NPY 标量训练循环会造成设备和目标错位，因此暂时保留为参考，不在命令行中伪装成可用模型。

## 当前可训练模型

```bash
python train_modified.py --model simvp ...
python train_modified.py --model tau ...
python train_modified.py --model fourcastlstm ...
python train_modified.py --model cascast ...
python train_modified.py --model postrainbench ...
```

推荐先与 `convlstm` 使用相同的数据划分、损失和随机种子进行比较。`simvp` 和 `tau` 来自 OpenSTL，分别使用 gSTA 和 Temporal Attention Unit；`fourcastlstm` 会先把 41×41 输入补齐到 48×48，再按 15 个时刻顺序编码。

## 依赖

OpenSTL 和 FourCastLSTM 都需要 `timm`；FourCastLSTM 还使用 `einops`。请在服务器的 PyTorch/CUDA 环境已经准备好后安装：

```bash
python -m pip install -r requirements.txt
```

外部源码保持在 `vendor/external_models/`，适配逻辑集中在 `src/tianchi_rainfall/models.py`，便于在报告中区分原作者代码和本项目改造代码，并按比赛要求注明开源仓库与许可证。
