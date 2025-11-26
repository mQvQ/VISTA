'''
st: reslinear
'''
from abc import abstractmethod
from functools import partial
import math
from typing import Iterable
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import STSpatialTransformer


# dummy replace
def convert_module_to_f16(x):
    pass


def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, st, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, st, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                 st = layer(st, emb)
            elif isinstance(layer, STSpatialTransformer):
                st = layer(st, context)
            else:
                st = layer(st)
        return st


class LinearUpsample(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, in_dim*2)

    def forward(self, x):
        assert x.shape[2] == self.in_dim
        return self.linear(x)


class LinearDownsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, in_dim//2)

    def forward(self, x):
        assert x.shape[2] == self.in_dim
        return self.linear(x)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        in_dim=1024,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        st_dilation=1,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.st_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(1, channels, self.out_channels, 1, padding='same', dilation=st_dilation),

        )
        self.updown = up or down
        self.in_dim = in_dim
        if up:
            self.ah_upd = LinearUpsample(in_dim)
            self.ax_upd = LinearUpsample(in_dim)
        elif down:
            self.ah_upd = LinearDownsample(in_dim)
            self.ax_upd = LinearDownsample(in_dim)
        else:
            self.ah_upd = self.ax_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.st_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(1, self.out_channels, self.out_channels, 1, padding='same'))
        )

        if self.out_channels == channels:
            self.st_skip_connection = nn.Identity()
        elif use_conv:
            self.st_skip_connection = conv_nd(1, channels, self.out_channels, 3, padding=1)
        else:
            self.st_skip_connection = conv_nd(1, channels, self.out_channels, 1)

    def forward(self, st, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (st, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, st, emb):
        if self.updown:
            st_in_rest, st_in_conv = self.st_in_layers[:-1], self.st_in_layers[-1]
            st_h = st_in_rest(st)
            st_h = self.ah_upd(st_h)
            st = self.ax_upd(st)
            st_h = st_in_conv(st_h)

        else:
            st_h = self.st_in_layers(st)
        emb_out = self.emb_layers(emb).type(st.dtype)
        if self.use_scale_shift_norm:
            st_emb_out = emb_out[..., None]
            scale, shift = th.chunk(st_emb_out, 2, dim=1)
            st_out_norm, st_out_rest = self.st_out_layers[0], self.st_out_layers[1:]
            st_h = st_out_norm(st_h) * (1 + scale) + shift
            st_h = st_out_rest(st_h)
        else:
            st_emb_out = emb_out
            st_h = st_h + st_emb_out
            st_h = self.st_out_layers(st_h)
        return self.st_skip_connection(st)+st_h



def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])



class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class InitialBlock(nn.Module):
    def __init__(
            self,
            st_in_channels,
            st_out_channels,
    ):
        super().__init__()
        self.st_conv = conv_nd(1, st_in_channels, st_out_channels, 1, padding='same') # stride=1

    def forward(self, st):
        return self.st_conv(st)


class STUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        input_st_num,
        st_size,
        in_channels,
        model_channels,
        st_out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        ckpt_path=None,
        ignore_keys=[],  # ignore keys for loading checkpoint
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"
        self.st_size = st_size

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.st_out_channels = st_out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.input_st_num = input_st_num

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.st_linear_in = nn.Linear(self.input_st_num, self.st_size[1])
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(
                InitialBlock(self.st_size[0], st_out_channels=model_channels))
            ])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        dilation = 1
        max_dila = 10
        len_st_conv = 1
        in_dim = self.st_size[-1]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        in_dim=in_dim,
                        out_channels=int(mult * model_channels),
                        st_dilation=2 ** (dilation % max_dila),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                dilation += len_st_conv
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        STSpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1: # down-sample
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            in_dim=in_dim,
                            out_channels=out_ch,
                            dims=dims,
                            st_dilation=2 ** (dilation % max_dila),
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        # if resblock_updown
                        # else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                dilation += len_st_conv
                ch = out_ch
                in_dim //= 2
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                st_dilation=2 ** (dilation % max_dila),
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            STSpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                st_dilation=2 ** (dilation % max_dila),
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        dilation -= len_st_conv
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        st_dilation=2 ** (dilation % max_dila),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                dilation -= len_st_conv
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                         STSpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            in_dim=in_dim,
                            out_channels=out_ch,
                            dims=dims,
                            st_dilation=2 ** (dilation % max_dila),
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        # if resblock_updown
                        # else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    in_dim *= 2
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.st_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, st_out_channels, 1)),
        )
        self.st_linear_out = nn.Linear(self.st_size[1], self.input_st_num)

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(2, model_channels, n_embed, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, st, timesteps=None, context=None, label_embedding=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param label_embedding: tensor of label embeddings
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        st_hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if label_embedding is not None:
            # label to embedding done elsewhere
            emb = emb + label_embedding

        if self.num_classes is not None:
            assert y.shape == (st.shape[0])
            emb = emb + self.label_emb(y)

        st = st.type(self.dtype)

        st = self.st_linear_in(st)

        for module in self.input_blocks:
            st = module(st, emb, context)
            st_hs.append(st)
        st = self.middle_block(st, emb, context)
        for module in self.output_blocks:
            st = th.cat([st, st_hs.pop()], dim=1)
            st = module(st, emb, context)

        st = self.st_out(st)
        st = self.st_linear_out(st)
        return st


# class JointEncoderUNetModel(nn.Module):

if __name__ == '__main__':
    import time
    model_channels = 192
    emb_channels = 128
    in_channels = 1
    input_st_num = 838
    st_size = [1, 1024]
    st_out_channels = 1
    num_heads = 2
    num_res_blocks = 1
    attention_resolutions = [8, 4, 2]
    lr = 0.0001
    channel_mult = (1, 2, 3, 5)
    context_dim = 768
    model = STUNetModel(
        input_st_num=input_st_num,
        st_size=st_size,
        in_channels=in_channels,
        model_channels=model_channels,
        st_out_channels=st_out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        context_dim=context_dim,
        num_heads=num_heads,
        use_scale_shift_norm=True,
        use_checkpoint=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params / 1_000_000} M")

    optim = th.optim.SGD(model.parameters(), lr=lr)
    model.train()
    while True:
        time_start = time.time()
        st = th.randn([2, 1, 838])
        context = th.randn([2, 77, 768])
        time_index = th.tensor([1, 2])
        st_out = model(st, time_index, context)
        print(st_out.shape)
        st_target = th.randn_like(st_out)
        loss =  F.mse_loss(st_target, st_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"loss:{loss} time:{time.time() - time_start}")

