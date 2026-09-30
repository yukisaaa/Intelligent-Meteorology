"""PredRNN ST-LSTM encoder with a rainfall regression head.

The recurrent cell follows the PredRNN spatiotemporal LSTM design. The CIKM
records carry a scalar rainfall target, so we pool the final recurrent state
and regress that target instead of decoding future video frames.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden = hidden
        self.forget_bias = 1.0
        self.conv_x = nn.Conv2d(in_channels, 7 * hidden, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden, 4 * hidden, kernel_size, padding=padding)
        self.conv_m = nn.Conv2d(hidden, 3 * hidden, kernel_size, padding=padding)
        self.conv_o = nn.Conv2d(2 * hidden, hidden, kernel_size, padding=padding)
        self.conv_last = nn.Conv2d(2 * hidden, hidden, kernel_size=1)

    def forward(self, x, h, c, memory):
        x_i, x_f, x_g, x_ip, x_fp, x_gp, x_o = self.conv_x(x).chunk(7, dim=1)
        h_i, h_f, h_g, h_o = self.conv_h(h).chunk(4, dim=1)
        m_i, m_f, m_g = self.conv_m(memory).chunk(3, dim=1)

        c_new = torch.sigmoid(x_f + h_f + self.forget_bias) * c
        c_new = c_new + torch.sigmoid(x_i + h_i) * torch.tanh(x_g + h_g)
        m_new = torch.sigmoid(x_fp + m_f + self.forget_bias) * memory
        m_new = m_new + torch.sigmoid(x_ip + m_i) * torch.tanh(x_gp + m_g)
        merged = torch.cat((c_new, m_new), dim=1)
        o = torch.sigmoid(x_o + h_o + self.conv_o(merged))
        h_new = o * torch.tanh(self.conv_last(merged))
        return h_new, c_new, m_new


class PredRNNRainRegressor(nn.Module):
    """Encode 15 observations (4 radar heights per observation) into rainfall."""

    def __init__(self, in_channels: int = 4, hidden: int = 16, layers: int = 2):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.hidden = hidden
        self.layers = layers
        self.cells = nn.ModuleList(
            SpatioTemporalLSTMCell(in_channels if i == 0 else hidden, hidden)
            for i in range(layers)
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, sequence):
        # [B, T, 4, 101, 101], normalized radar intensities
        batch, steps, _, height, width = sequence.shape
        x = sequence.reshape(batch * steps, sequence.shape[2], height, width)
        # Pooling to ~26x26 keeps the full temporal/height information while
        # making recurrent CPU training feasible on this scalar-regression task.
        x = F.avg_pool2d(x, kernel_size=4, ceil_mode=True)
        _, channels, reduced_h, reduced_w = x.shape
        x = x.reshape(batch, steps, channels, reduced_h, reduced_w)

        h = [x.new_zeros((batch, self.hidden, reduced_h, reduced_w)) for _ in self.cells]
        c = [torch.zeros_like(state) for state in h]
        memory = torch.zeros_like(h[0])
        for t in range(steps):
            layer_input = x[:, t]
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer], memory = cell(layer_input, h[layer], c[layer], memory)
                layer_input = h[layer]
        features = h[-1].mean(dim=(-2, -1))
        return self.regressor(features).squeeze(-1)
