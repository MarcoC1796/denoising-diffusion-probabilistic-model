import math
import os
from enum import Enum
from typing import Type, TypeVar

import torch
import torch.nn.functional as F
from config import UNetConfig
from torch import Tensor, nn

I = TypeVar("I", bound="UNet")  # noqa: E741


class LayerType(Enum):
    CONVOLUTION = 1
    RESIDUAL_BLOCK = 2
    DOWNSAMPLE = 3
    UPSAMPLE = 4


class CustomConv2d(nn.Conv2d):
    def forward(self, input: Tensor, t_embd: Tensor | None = None) -> Tensor:
        return super().forward(input)


class DownSampleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor, t_embd: Tensor):
        B, C, H, W = x.shape
        x = self.downsample(x)
        assert x.shape == (B, C, H // 2, W // 2)
        return x


class UpSampleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        padding = "same"
        self.upsample = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=1, padding=padding
        )

    def forward(self, x: Tensor, t_embd: Tensor):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=(2 * H, 2 * W), mode="nearest-exact")
        x = self.upsample(x)
        assert x.shape == (B, C, 2 * H, 2 * W)
        return x


class TimeEmbeddingNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        self.dense_0 = nn.Linear(config.n_ch, config.n_ch * 4)
        self.silu = nn.SiLU()
        self.dense_1 = nn.Linear(config.n_ch * 4, config.n_ch * 4)

    def forward(self, t: Tensor) -> Tensor:
        t_embd = self.get_timestep_embedding(t)
        t_embd = self.dense_0(t_embd)
        t_embd = self.silu(t_embd)
        t_embd = self.dense_1(t_embd)
        return t_embd

    def get_timestep_embedding(self, t: Tensor) -> Tensor:
        half_n_ch = self.config.n_ch // 2
        scale = torch.log(torch.tensor((10_000.0,), dtype=t.dtype, device=t.device)) / (
            half_n_ch - 1
        )
        with torch.no_grad():
            emb = torch.exp(
                torch.arange(half_n_ch, device=t.device) * -scale.view(1, 1)
            )
            emb = t.view(-1, 1) * emb.view(1, -1)
            emb = torch.concat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb


class ResConvBlock(nn.Module):
    def __init__(self, config: UNetConfig, in_ch: int, out_ch: int):
        super().__init__()
        padding = "same"
        num_groups = 32
        kernel_size = 3
        self.config = config
        self.out_ch = out_ch
        self.conv_group_1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, self.out_ch, kernel_size, padding=padding),
        )
        self.dense_t = nn.Linear(config.n_ch * 4, self.out_ch)
        self.conv_group_2 = nn.Sequential(
            nn.GroupNorm(num_groups, self.out_ch),
            nn.SiLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size, padding=padding),
        )
        self.c_proj = nn.Linear(in_ch, self.out_ch)

    def forward(self, x: Tensor, t_embd: Tensor) -> Tensor:
        B, _, _, _ = x.shape
        y = self.conv_group_1(x)
        # adding time embedding
        y += self.dense_t(t_embd).reshape(
            B, self.out_ch, 1, 1
        )  # x: (B, out_ch, H, W) | t_embd: (B, out_ch)
        y = self.conv_group_2(y)
        x = self.c_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        assert x.shape == y.shape
        return x + y


class ResAttnBlock(nn.Module):
    def __init__(self, config: UNetConfig, ch: int):
        super().__init__()
        num_groups = 32
        self.config = config
        self.ch = ch
        self.gn = nn.GroupNorm(num_groups, self.ch)
        self.c_attn = nn.Linear(self.ch, 3 * self.ch)
        self.c_proj = nn.Linear(self.ch, self.ch)

    def forward(self, x: Tensor, t_embd: Tensor) -> Tensor:
        B, C, S, _ = x.shape
        norm_x = self.gn(x)
        qkv = self.c_attn(norm_x.permute(0, 2, 3, 1))
        q, k, v = qkv.split(self.ch, dim=3)
        att = torch.einsum("bhwc, bHWc -> bhwHW", q, k) * (1.0 / math.sqrt(C))
        att = F.softmax(att.reshape(B, S, S, S * S), dim=-1).reshape(B, S, S, S, S)
        att = torch.einsum("bhwHW, bHWc -> bhwc", att, v)
        y = self.c_proj(att).permute(0, 3, 1, 2)
        assert x.shape == y.shape
        return x + y


class ResNetBlock(nn.Module):
    def __init__(
        self, config: UNetConfig, in_ch: int, out_ch: int, add_attention=False
    ):
        super().__init__()
        self.config = config
        self.add_attention = add_attention
        self.conv_block = ResConvBlock(self.config, in_ch, out_ch)
        self.att_block = (
            ResAttnBlock(self.config, out_ch) if self.add_attention else None
        )

    def forward(self, x, t_embd):
        x = self.conv_block(x, t_embd)
        if self.add_attention:
            x = self.att_block(x, t_embd)
        return x


class SamplingBlock(nn.Module):
    def __init__(self, config: UNetConfig, layout: tuple[list, list, list, list]):
        super().__init__()
        self.config = config
        self.sampling = nn.ModuleList()
        for in_ch, out_ch, S, layer_type in zip(*layout):
            match layer_type:
                case LayerType.CONVOLUTION:
                    self.sampling.append(
                        CustomConv2d(in_ch, out_ch, kernel_size=3, padding="same")
                    )
                case LayerType.RESIDUAL_BLOCK:
                    self.sampling.append(
                        ResNetBlock(
                            self.config,
                            in_ch,
                            out_ch,
                            add_attention=S in self.config.attn_resolutions,
                        )
                    )
                case LayerType.DOWNSAMPLE:
                    self.sampling.append(DownSampleConv(in_ch, out_ch))
                case LayerType.UPSAMPLE:
                    self.sampling.append(UpSampleConv(in_ch, out_ch))
                case _:
                    raise ValueError(f"Unknown layer type: {layer_type}")


class DownSamplingBlock(SamplingBlock):
    def forward(self, x: Tensor, t_emb: Tensor) -> tuple[Tensor, list[Tensor]]:
        hs = []
        for block in self.sampling:
            x = block(x, t_emb)
            hs.append(x)
        return x, hs


class MiddleBlock(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        ch = config.n_ch * config.ch_mult[-1]
        self.res1 = ResNetBlock(self.config, in_ch=ch, out_ch=ch)
        self.att = ResAttnBlock(self.config, ch=ch)
        self.res2 = ResNetBlock(self.config, in_ch=ch, out_ch=ch)

    def forward(self, x: Tensor, t_embd: Tensor) -> Tensor:
        x = self.res1(x, t_embd)
        x = self.att(x, t_embd)
        x = self.res2(x, t_embd)
        return x


class UpSamplingBlock(SamplingBlock):
    def forward(self, x: Tensor, t_embd: Tensor, hs: list[Tensor]) -> Tensor:
        for block in self.sampling:
            if isinstance(block, ResNetBlock):
                h = hs.pop()
                x = torch.cat([x, h], dim=1)
            x = block(x, t_embd)
        return x


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        num_groups = 32
        self.t_embd_net = TimeEmbeddingNet(self.config)
        ds_layout = self._generate_ds_layout()
        self.downsampling = DownSamplingBlock(self.config, ds_layout)
        self.middle = MiddleBlock(self.config)
        us_layout = self._generate_us_layout(ds_layout)
        self.upsampling = UpSamplingBlock(self.config, us_layout)
        self.gn = nn.GroupNorm(num_groups, self.config.n_ch)
        self.silu = nn.SiLU()
        self.conv = nn.Conv2d(
            self.config.n_ch,
            self.config.in_img_C,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        _, C, S, _ = x.shape
        assert S == x.shape[3]
        assert S == self.config.in_img_S
        assert C == self.config.in_img_C
        t_embd = self.t_embd_net(t)
        x, hs = self.downsampling(x, t_embd)
        x = self.middle(x, t_embd)
        x = self.upsampling(x, t_embd, hs)
        x = self.silu(self.gn(x))
        x = self.conv(x)
        return x

    def _generate_ds_layout(self) -> tuple[list, list, list, list]:
        input_channels = [self.config.in_img_C]
        output_channels = [self.config.n_ch]
        spatial_sizes = [self.config.in_img_S]
        layer_types = [LayerType.CONVOLUTION]

        for level_idx, channel_multiplier in enumerate(self.config.ch_mult):
            current_out_channels = self.config.n_ch * channel_multiplier
            output_channels.extend([current_out_channels] * self.config.num_res_blocks)
            spatial_sizes.extend([spatial_sizes[-1]] * self.config.num_res_blocks)
            layer_types.extend([LayerType.RESIDUAL_BLOCK] * self.config.num_res_blocks)

            if level_idx < len(self.config.ch_mult) - 1:
                output_channels.append(current_out_channels)
                spatial_sizes.append(spatial_sizes[-1] // 2)
                layer_types.append(LayerType.DOWNSAMPLE)

        input_channels.extend(output_channels[:-1])
        return input_channels, output_channels, spatial_sizes, layer_types

    def _generate_us_layout(
        self, ds_layout: tuple[list, list, list, list]
    ) -> tuple[list, list, list, list]:
        _, ds_output_channels, ds_spatial_sizes, _ = ds_layout

        input_channels = [self.config.n_ch * self.config.ch_mult[-1]]
        output_channels = []
        spatial_sizes = [self.config.in_img_S // (2 ** (len(self.config.ch_mult) - 1))]
        layer_types = []

        for level_idx, channel_multiplier in reversed(
            list(enumerate(self.config.ch_mult))
        ):
            current_out_channels = self.config.n_ch * channel_multiplier
            output_channels.extend(
                [current_out_channels] * (self.config.num_res_blocks + 1)
            )
            spatial_sizes.extend([spatial_sizes[-1]] * (self.config.num_res_blocks + 1))
            layer_types.extend(
                [LayerType.RESIDUAL_BLOCK] * (self.config.num_res_blocks + 1)
            )

            if level_idx != 0:
                output_channels.append(current_out_channels)
                spatial_sizes.append(spatial_sizes[-1] * 2)
                layer_types.append(LayerType.UPSAMPLE)

        input_channels.extend(output_channels[:-1])
        for i, data in enumerate(list(zip(spatial_sizes, layer_types))):
            spatial_size, layer_type = data
            if layer_type == LayerType.RESIDUAL_BLOCK:
                assert spatial_size == ds_spatial_sizes.pop()
                input_channels[i] += ds_output_channels.pop()

        assert len(ds_output_channels) == 0
        return input_channels, output_channels, spatial_sizes, layer_types

    @classmethod
    def from_pth(cls: Type[I], config: UNetConfig, pth_path: str) -> I:
        model = cls(config)
        if os.path.exists(pth_path):
            model.load_state_dict(torch.load(pth_path, weights_only=True))
            print(f"Loaded model from {pth_path}")
        else:
            raise FileNotFoundError(f"Error: File {pth_path} not found!")
        return model


if __name__ == "__main__":
    import time

    device = device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device)
    config = UNetConfig()
    model = UNet(config).to(device)
    T = 1000
    B, C, H, W = 64, config.in_img_C, config.in_img_S, config.in_img_S
    x = torch.rand(B, C, H, W, device=device)
    t = torch.randint(low=1, high=T + 1, size=(B,), device=device)
    print("x:", x.shape)
    start = time.perf_counter()
    y = model(x, t)
    end = time.perf_counter() - start
    print("y:", y.shape)
    print("inference time:", end)
