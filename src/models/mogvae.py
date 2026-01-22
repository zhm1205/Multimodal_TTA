# src/models/generators/vae_delta.py
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from monai.networks.nets import VarAutoEncoder

from src.utils.config import get_config
from src.registry import register_model

@register_model("vae_delta_mog")
class VAEDeltaMoG(VarAutoEncoder):
    """
    VarAutoEncoder 的 δ 生成器版本：
    - 维持 MONAI 编码/解码结构
    - 在 latent 处用 K 分量的混合高斯先验做“软混合重参数化”：z = sum_k pi_tilde_k * (mu_k + sigma_k * eps_k)
    - 可选条件门控：pi_tilde(x) = softmax(log_pi + gate(x)); 默认用全局 log_pi
    - forward(x) -> delta_raw in [-inf, +inf]，外层照旧做 L∞ / 像素盒投影
    """
    def __init__(self, cfg: DictConfig | Dict[str, Any],
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # 读取 VAE 结构
        c_in  = in_channels  if in_channels  is not None else int(get_config(cfg, "in_channels", 3))
        c_out = out_channels if out_channels is not None else int(get_config(cfg, "out_channels", 1))
        latent_size = int(get_config(cfg, "latent_size", 128))
        channels = list(get_config(cfg, "channels", [32, 64, 128, 256, 512]))
        strides  = list(get_config(cfg, "strides",  [2, 2, 2, 2]))

        super().__init__(
            spatial_dims=2,
            in_channels=c_in,
            out_channels=c_out,     # 直接输出 δ 的通道数（通常 1）
            latent_size=latent_size,
            channels=channels,
            strides=strides,
            num_res_units=int(get_config(cfg, "num_res_units", 0)),
            act=get_config(cfg, "act", "relu"),
            norm=get_config(cfg, "norm", "BATCH"),
            dropout=float(get_config(cfg, "dropout", 0.0)),
        )

        # ------ MoG 先验参数 ------
        K = int(get_config(cfg, "mog.K", 16))
        self.K = K
        self.mu_k     = nn.Parameter(torch.zeros(K, latent_size))
        self.logsig_k = nn.Parameter(torch.zeros(K, latent_size))  # sigma_k = softplus or exp(logsig)
        self.logpi    = nn.Parameter(torch.zeros(K))               # global mixing logits

        # 可选门控（条件先验）
        use_gate = bool(get_config(cfg, "mog.use_gate", False))
        self.use_gate = use_gate
        if use_gate:
            gate_dim = int(get_config(cfg, "mog.gate_hidden", 256))
            # 从瓶颈特征做全局池化得到门控向量
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.gate_mlp = nn.Sequential(
                nn.Conv2d(channels[-1], gate_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_dim, K, kernel_size=1),
            )

        # sigma 的下界，避免数值不稳
        self.sigma_min = float(get_config(cfg, "mog.sigma_min", 1e-3))

    def _soft_sigma(self) -> torch.Tensor:
        # softplus/exp 均可；这里用 softplus 并加下界
        return F.softplus(self.logsig_k).clamp_min(self.sigma_min)   # [K, Dz]

    def _mix_weights(self, enc_feat: Optional[torch.Tensor]) -> torch.Tensor:
        # enc_feat: 编码器瓶颈特征 [N, Cb, Hb, Wb]
        if self.use_gate and enc_feat is not None:
            g = self.gate_mlp(self.gap(enc_feat)).squeeze(-1).squeeze(-1)  # [N,K]
            logits = self.logpi.unsqueeze(0) + g
            pi_tilde = torch.softmax(logits, dim=-1)                       # [N,K]
        else:
            pi_tilde = torch.softmax(self.logpi, dim=-1).unsqueeze(0)      # [1,K]
        return pi_tilde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回 delta_raw；外层照旧做 L∞ 投影与像素盒裁剪
        """
        # MONAI VAE: 编码得到 mu_post/logvar_post + 瓶颈特征（用于门控）
        # 注：VarAutoEncoder 没直接返回瓶颈特征，这里复用其 encode 路径拿到最后一层特征。
        # 方案：沿用父类 encode 算 mu/logvar 的同时，借助 _fc层之前的特征（需要 forward hook）。
        # 为最小入侵，我们复用父类的 reparameterize 接口，但 z 用我们 MoG 软混合替换。

        # 1) 标准 VAE 的后验参数（不用它来采样，但可做监控/正则）
        mu_post, logvar_post = self._encode(x)              # [N,Dz], [N,Dz]
        # 2) MoG 混合权
        enc_feat = self._enc_feat   # 在 _encode 里通过 hook 暂存；见下方 _encode 定义
        pi_tilde = self._mix_weights(enc_feat)              # [N,K] 或 [1,K]
        # 3) 软混合 reparam：z = sum_k pi_tilde_k * (mu_k + sigma_k * eps_k)
        N = x.size(0); Dz = mu_post.size(1)
        mu_k   = self.mu_k.unsqueeze(0).expand(N, -1, -1)           # [N,K,Dz]
        sig_k  = self._soft_sigma().unsqueeze(0).expand(N, -1, -1)  # [N,K,Dz]
        eps_k  = torch.randn_like(sig_k)                             # [N,K,Dz]
        z_k    = mu_k + sig_k * eps_k                                # [N,K,Dz]
        w      = pi_tilde.unsqueeze(-1)                               # [N,K,1]
        z      = (w * z_k).sum(dim=1)                                # [N,Dz]

        # 4) 解码得到 δ
        delta = self.decode(z)                                       # [N,C_out,H,W]
        return delta

    # --- 轻覆写 encode，额外留出瓶颈特征（用 hook 方式最省心） ---
    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        复制 VarAutoEncoder.encode 逻辑，但把瓶颈特征暂存到 self._enc_feat，方便门控
        """
        # VarAutoEncoder 的 encoder 是 self.encoder（Sequential/Net），最后接 self.fc_mu / self.fc_var
        h = self.encoder(x)          # [N,Cb,Hb,Wb]
        self._enc_feat = h
        h_gap = F.adaptive_avg_pool2d(h, 1).flatten(1)  # 与父类一致：GAP 后接全连接
        mu     = self.fc_mu(h_gap)
        logvar = self.fc_var(h_gap)
        return mu, logvar
