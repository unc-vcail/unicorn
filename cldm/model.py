import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, Upsample
from ldm.models.diffusion.ddpm_multi import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim_multi import DDIMSampler
from cldm.cldm_encoders import get_encoder


def modulated_conv2d(x, w, s, demodulate=False, padding=0, input_gain=None, bias=None, stride=1, dilation=1):
    """Grouped depthwise conv with per-sample style modulation (StyleGAN3)."""
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    w = w.unsqueeze(0)
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(input=x, weight=w.to(x.dtype), bias=bias, stride=stride, padding=padding, dilation=dilation, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, weights=None, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += torch.sum(torch.stack(control.pop(), dim=1) * weights.pop(), dim=1)

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + (torch.sum(torch.stack(control.pop(), dim=1) * weights.pop(), dim=1))], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class MixerBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.conv1_1 = conv_nd(2, channels, 1, 3, padding=1)
        self.conv2_1 = conv_nd(2, channels, channels, 1)
        self.mn = nn.Softmax(dim=1)

    def forward(self, x_s, bs, task_emb=None):
        _, _, xh, xw = x_s.shape
        if task_emb is not None:
            w1 = modulated_conv2d(x_s, self.conv1_1.weight, task_emb.repeat(x_s.shape[0], 1), padding=1) + self.conv1_1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            w2 = modulated_conv2d(x_s, self.conv2_1.weight, task_emb.repeat(x_s.shape[0], 1)) + self.conv2_1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            w1 = self.conv1_1(x_s)
            w2 = self.conv2_1(x_s)
        w1_as = torch.chunk(w1, x_s.shape[0] // bs, dim=0)
        w2_as = torch.chunk(w2, x_s.shape[0] // bs, dim=0)
        return self.mn(torch.stack(w1_as, dim=1) * torch.stack(w2_as, dim=1))


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_subchannels,
            hint_levels,
            encoder_type,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            all_tasks_num=13,
            encoder_scales=None,
            num_control_heads=None,
            duplicate_heads=None,
            activate_heads=None,
            specular_mask=None
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None
        if context_dim is not None:
            assert use_spatial_transformer
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("num_res_blocks must be int or list matching channel_mult length")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.hint_subchannels = hint_subchannels
        self.hint_levels = hint_levels
        self.encoder_scales = encoder_scales
        self.num_control_heads = num_control_heads
        self.duplicate_heads = duplicate_heads
        self.activate_heads = activate_heads
        self.specular_mask = specular_mask
        self.encoder_type = encoder_type
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.task_id_hypernet = nn.Sequential(
            linear(768, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.task_id_layernet = []

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        self.task_id_layernet.append(linear(time_embed_dim, model_channels))
        self.zero_convs_diffusion = nn.ModuleList([nn.ModuleList([self.make_zero_conv(model_channels) if dh else None
                                        for _, dh in zip(range(self.num_control_heads), self.activate_heads)])])
        self.zero_convs_control = nn.ModuleList([nn.ModuleList([self.make_zero_conv(model_channels) if dh else None
                                        for _, dh in zip(range(self.num_control_heads), self.activate_heads)])])
        self.moes = nn.ModuleList([MixerBlock(model_channels, num_control_heads)])

        self.all_hints_blocks = nn.ModuleList([])
        for hx, dh in zip(range(len(self.hint_subchannels)), self.activate_heads):
            if dh:
                cntr = 1
                out_channels = 16
                hint_blocks = nn.ModuleList([nn.ModuleList([]) for _ in range(len(self.hint_levels[hx]))])
                while cntr <= 4:
                    in_channels = out_channels
                    for hlx, hl in enumerate(self.hint_levels[hx]):
                        if hl >= cntr:
                            if cntr == 4:
                                enc = get_encoder(encoder_type[hx][hlx], in_channels, 256, hl, 4)
                                hint_blocks[hlx].append(enc)
                                out_channels = 256
                            elif cntr == 3:
                                enc = get_encoder(encoder_type[hx][hlx], in_channels, in_channels * self.encoder_scales[encoder_type[hx][hlx]][cntr - 1], hl, 3)
                                hint_blocks[hlx].append(enc)
                                out_channels = in_channels * self.encoder_scales[encoder_type[hx][hlx]][cntr - 1]
                            elif cntr == 2:
                                enc = get_encoder(encoder_type[hx][hlx], in_channels, in_channels * self.encoder_scales[encoder_type[hx][hlx]][cntr - 1], hl, 2)
                                hint_blocks[hlx].append(enc)
                                out_channels = in_channels * self.encoder_scales[encoder_type[hx][hlx]][cntr - 1]
                            elif cntr == 1:
                                enc = get_encoder(encoder_type[hx][hlx], self.hint_subchannels[hx][hlx], in_channels * self.encoder_scales[encoder_type[hx][hlx]][cntr - 1], hl, 1)
                                hint_blocks[hlx].append(enc)
                                out_channels = in_channels * self.encoder_scales[encoder_type[hx][hlx]][cntr - 1]
                    cntr += 1
                self.all_hints_blocks.append(hint_blocks)
            else:
                out_channels = 256
                self.all_hints_blocks.append(None)

        self.input_hint_block_zeroconv_0 = nn.ModuleList(
            [nn.ModuleList([(conv_nd(dims, out_channels, out_channels, 3, padding=1)),
                            (conv_nd(dims, out_channels, out_channels, 3, padding=1))]) if dh else None
             for _, dh in zip(range(self.num_control_heads), self.activate_heads)])
        self.task_id_layernet_zeroconv_0 = linear(time_embed_dim, out_channels)

        self.input_hint_block_share = TimestepEmbedSequential(
            conv_nd(dims, out_channels, model_channels, 3, padding=1),
            nn.SiLU(),
        )

        self.input_hint_block_zeroconv_1 = nn.ModuleList(
            [nn.ModuleList([(conv_nd(dims, model_channels, model_channels, 3, padding=1)),
                            (conv_nd(dims, model_channels, model_channels, 3, padding=1))])
             for _ in range(self.num_control_heads)])
        self.task_id_layernet_zeroconv_1 = linear(time_embed_dim, model_channels)

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims,
                                   use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(disable_self_attentions) else False
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                                           num_head_channels=dim_head, use_new_attention_order=use_new_attention_order)
                            if not use_spatial_transformer else
                            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                               disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                               use_checkpoint=use_checkpoint)
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.task_id_layernet.append(linear(time_embed_dim, ch))
                self.zero_convs_diffusion.append(nn.ModuleList([self.make_zero_conv(ch) if dh else None for _, dh in zip(range(self.num_control_heads), self.activate_heads)]))
                self.zero_convs_control.append(nn.ModuleList([self.make_zero_conv(ch) if dh else None for _, dh in zip(range(self.num_control_heads), self.activate_heads)]))
                self.moes.append(MixerBlock(ch, num_control_heads))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                        if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.task_id_layernet.append(linear(time_embed_dim, ch))
                self.zero_convs_diffusion.append(nn.ModuleList([self.make_zero_conv(ch) if dh else None for _, dh in zip(range(self.num_control_heads), self.activate_heads)]))
                self.zero_convs_control.append(nn.ModuleList([self.make_zero_conv(ch) if dh else None for _, dh in zip(range(self.num_control_heads), self.activate_heads)]))
                self.moes.append(MixerBlock(ch, num_control_heads))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head,
                           use_new_attention_order=use_new_attention_order)
            if not use_spatial_transformer else
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                               disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                               use_checkpoint=use_checkpoint),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self.middle_block_out = nn.ModuleList([self.make_zero_conv(ch) if dh else None for _, dh in zip(range(self.num_control_heads), self.activate_heads)])
        self._feature_size += ch
        self.moe_middle = MixerBlock(ch, num_control_heads)
        self.task_id_layernet = nn.ModuleList(self.task_id_layernet)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def concat_heads(self, hls, task_id):
        ln = sum(task_id)
        l = []
        if self.activate_heads[0]:
            l = [hls[0]]
            ln += 1
        for ix, (_h, _tid) in enumerate(zip(hls[1:], task_id)):
            if _tid:
                l.append(_h)
        return torch.cat(l, dim=0), ln

    def split_heads(self, h, task_id, bs):
        ln = sum(task_id)
        if self.activate_heads[0]:
            ln += 1
        hs = torch.chunk(h, ln, dim=0)
        cnt = 0
        if self.activate_heads[0]:
            l = [hs[0]]
            cnt = 1
        else:
            l = [None]
        for ix, _tid in enumerate(task_id):
            if _tid:
                l.append(hs[cnt])
                cnt += 1
            else:
                l.append(None)
        return l

    def get_contructed_hints(self, hints, task_id, logging=False):
        duplicate_heads = self.duplicate_heads
        if logging:
            construct_hints = [hints[:, 0:self.hint_subchannels[0][0]]]
        else:
            if self.activate_heads[0]:
                construct_hints = [[hints[:, 0:self.hint_subchannels[0][0]]]]
            else:
                construct_hints = [[None]]
        cntr = self.hint_subchannels[0][0]
        for i, _task_id, duplicate_head in zip(range(1, self.num_control_heads), task_id, duplicate_heads[1:]):
            if _task_id:
                _construct_hints = []
                if duplicate_head and not logging:
                    _construct_hints.append(hints[:, 0:self.hint_subchannels[0][0]])
                for hsc in (self.hint_subchannels[i][1:] if duplicate_head else self.hint_subchannels[i]):
                    if logging:
                        _construct_hints.append(hints[:, cntr:(cntr + hsc)].expand(hints.shape[0], 3, hints.shape[2], hints.shape[3]))
                    else:
                        _construct_hints.append(hints[:, cntr:(cntr + hsc)])
                    cntr += hsc
                if logging:
                    construct_hints.extend(_construct_hints)
                else:
                    construct_hints.append(_construct_hints)
            else:
                construct_hints.append(None)
        return construct_hints

    def forward_hint_block(self, hid, hint_blocks, hints, emb, context):
        cntr = 1
        hint_paths = hints
        while cntr <= 4:
            for hlx, (hl, hb) in enumerate(zip(self.hint_levels[hid], hint_blocks)):
                if hl >= cntr:
                    hint_paths[hlx] = hb[cntr - 1](hint_paths[hlx], emb, context)
                if hlx != 0 and cntr == hl:
                    hint_paths[0] = hint_paths[0] + hint_paths[hlx]
            cntr += 1
        return hint_paths[0]

    def forward(self, x, hints, timesteps, context, **kwargs):
        BS_Real = x.shape[0]
        if kwargs is not None:
            task_id = kwargs['task']['id'].detach().int().tolist()
            task_feature = kwargs['task']['features']
            task_id_emb = self.task_id_hypernet(task_feature.squeeze(0))

        construct_hints = self.get_contructed_hints(hints, task_id)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = []
        if self.activate_heads[0]:
            h.append(self.forward_hint_block(0, self.all_hints_blocks[0], construct_hints[0], emb, context))
        else:
            h.append(None)
        for ix, _tid in enumerate(task_id):
            if _tid:
                h.append(self.forward_hint_block(ix + 1, self.all_hints_blocks[ix + 1], construct_hints[ix + 1], emb, context))
            else:
                h.append(None)

        _hs = []
        if self.activate_heads[0]:
            _hs.append(modulated_conv2d(h[0], self.input_hint_block_zeroconv_0[0][0].weight, self.task_id_layernet_zeroconv_0(task_id_emb).repeat(BS_Real, 1), padding=1) + self.input_hint_block_zeroconv_0[0][1](h[0]) + self.input_hint_block_zeroconv_0[0][0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        else:
            _hs.append(h[0])
        for ix, (_h, _tid) in enumerate(zip(h[1:], task_id)):
            if _tid:
                _hs.append(modulated_conv2d(_h, self.input_hint_block_zeroconv_0[ix + 1][0].weight, self.task_id_layernet_zeroconv_0(task_id_emb).repeat(BS_Real, 1), padding=1) + self.input_hint_block_zeroconv_0[ix + 1][1](_h) + self.input_hint_block_zeroconv_0[ix + 1][0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))
            else:
                _hs.append(None)

        _h, ln = self.concat_heads(_hs, task_id)
        h = self.input_hint_block_share(_h, emb, context)
        hs = self.split_heads(h, task_id, BS_Real)

        _hs = []
        if self.activate_heads[0]:
            _hs.append(modulated_conv2d(hs[0], self.input_hint_block_zeroconv_1[0][0].weight, self.task_id_layernet_zeroconv_1(task_id_emb).repeat(BS_Real, 1), padding=1) + self.input_hint_block_zeroconv_1[0][1](hs[0]) + self.input_hint_block_zeroconv_1[0][0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        else:
            _hs.append(hs[0])
        for ix, (_h, _tid) in enumerate(zip(hs[1:], task_id)):
            if _tid:
                _hs.append(modulated_conv2d(_h, self.input_hint_block_zeroconv_1[ix + 1][0].weight, self.task_id_layernet_zeroconv_1(task_id_emb).repeat(BS_Real, 1), padding=1) + self.input_hint_block_zeroconv_1[ix + 1][1](_h) + self.input_hint_block_zeroconv_1[ix + 1][0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))
            else:
                _hs.append(None)

        outs = []
        weights = []

        h = x.type(self.dtype)
        guided_hints = True
        for module, zero_conv_d, zero_conv_c, task_layer, moe in zip(self.input_blocks, self.zero_convs_diffusion, self.zero_convs_control, self.task_id_layernet, self.moes):
            if guided_hints is not None:
                h = module(h, emb, context)
                if self.activate_heads[0]:
                    _hs[0] += h
                for ix, (_h, _tid) in enumerate(zip(_hs[1:], task_id)):
                    if _tid:
                        _hs[ix + 1] += h
                guided_hints = None
                chs = list(_hs)
            else:
                _h, ln = self.concat_heads(_hs, task_id)
                h = module(_h, torch.tile(emb, [ln] + [1] * (emb.dim() - 1)), torch.tile(context, [ln] + [1] * (context.dim() - 1)))
                _hs = self.split_heads(h, task_id, BS_Real)

                chs = [None] * len(_hs)
                if self.activate_heads[0]:
                    chs[0] = _hs[0] + modulated_conv2d(_hs[0], zero_conv_c[0][0].weight, task_layer(task_id_emb).repeat(BS_Real, 1)) + zero_conv_c[0][0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                for ix, (_h, _tid) in enumerate(zip(_hs[1:], task_id)):
                    if _tid:
                        chs[ix + 1] = _h + modulated_conv2d(_h, zero_conv_c[ix + 1][0].weight, task_layer(task_id_emb).repeat(BS_Real, 1)) + zero_conv_c[ix + 1][0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            oh = [None] * len(_hs)
            if self.activate_heads[0]:
                oh[0] = modulated_conv2d(_hs[0], zero_conv_d[0][0].weight, task_layer(task_id_emb).repeat(BS_Real, 1)) + zero_conv_d[0][0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            for ix, (_h, _tid) in enumerate(zip(_hs[1:], task_id)):
                if _tid:
                    oh[ix + 1] = modulated_conv2d(_h, zero_conv_d[ix + 1][0].weight, task_layer(task_id_emb).repeat(BS_Real, 1)) + zero_conv_d[ix + 1][0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            h, ln = self.concat_heads(oh, task_id)
            weights.append(moe(h, BS_Real, None))
            outs.append([*torch.chunk(h, ln, dim=0)])
            _hs = chs

        h = self.middle_block(h, torch.tile(emb, [ln] + [1] * (emb.dim() - 1)), torch.tile(context, [ln] + [1] * (context.dim() - 1)))
        hs = self.split_heads(h, task_id, BS_Real)

        h = []
        if self.activate_heads[0]:
            h.append(self.middle_block_out[0](hs[0], emb, context))
        else:
            h.append(None)
        for ix, (_h, _tid) in enumerate(zip(hs[1:], task_id)):
            if _tid and self.activate_heads[ix + 1]:
                h.append(self.middle_block_out[ix + 1](hs[ix + 1], emb, context))
            else:
                h.append(_h)

        h, ln = self.concat_heads(h, task_id)
        weights.append(self.moe_middle(h, BS_Real, None))
        outs.append([*torch.chunk(h, ln, dim=0)])

        return outs, weights


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, sd_locked, num_control_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_loss_ema = torch.zeros(num_control_heads,)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.sd_locked = sd_locked
        self.control_scales = [1.0] * 13
        self.num_control_heads = num_control_heads

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        task_id = batch['task_id'][0]
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        task_dic = {}
        task_dic['id'] = task_id
        c_task = batch['task_features'][0]
        task_dic['features'] = c_task[:, :1]
        return x, dict(c_crossattn=[c], c_concat=[control], task=task_dic)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        task = cond['task']
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            controls, weights = self.control_model(x=x_noisy, hints=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, task=task)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=controls, only_mid_control=self.only_mid_control, weights=weights)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True, **kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c_cross, task = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["task"]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["hints_input"] = self.control_model.get_contructed_hints((c_cat - 0.5) * 2, task['id'].detach().tolist(), logging=True)

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c_cross], "task": task},
                                             batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
