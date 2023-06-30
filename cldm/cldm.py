import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
)
from ldm.models.diffusion.ddpm import (
    ImageEmbeddingConditionedLatentDiffusion,
    LatentDiffusion,
)
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):
    # def forward(self, x, hint, timesteps, context, y=None, **kwargs):
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        y=None,
        only_mid_control=False,
        **kwargs,
    ):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)

            if self.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
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
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
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
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

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

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)),
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
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
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, y=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):
    def __init__(
        self, control_stage_config, control_key, only_mid_control, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        control_txt = cond_txt
        if "c_control_crossattn" in cond:
            control_txt = torch.cat(cond["c_control_crossattn"], 1)

        if cond["c_concat"] is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            control = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                timesteps=t,
                context=control_txt,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
            )

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img(
            (512, 512), batch[self.cond_stage_key], size=16
        )

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c]},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c]},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, batch_size, shape, cond, verbose=False, **kwargs
        )
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


class CrossControlLDM(ControlLDM):
    def __init__(
        self,
        style_stage_config=None,
        style_key=None,
        control_dropout=0,
        style_dropout=0,
        guidance_scale=7,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.style_key = style_key
        self.control_dropout = control_dropout
        self.style_dropout = style_dropout
        self.guidance_scale = guidance_scale

        self.has_style_stage = False
        if style_stage_config is not None:
            self.instantiate_style_stage(style_stage_config)

    def instantiate_style_stage(self, config):
        self.has_style_stage = True
        self.style_stage_trainable = config.style_stage_trainable

        if not self.style_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as style stage.")
                self.style_encoder = self.first_stage_model
            elif config == "__is_unconditional__":
                print(
                    f"Training {self.__class__.__name__} as an unconditional style model."
                )
                self.style_encoder = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.style_encoder = model.eval()
                self.style_encoder.train = disabled_train
                for param in self.style_encoder.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.style_encoder = model

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        bs=None,
        dropout_control=True,
        dropout_style=True,
        *args,
        **kwargs,
    ):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        c = c["c_crossattn"][0]

        style = None
        embed = None

        style = batch[self.style_key]
        if bs is not None:
            style = style[:bs]

        style = style.to(self.device)

        # style img comes in [-1, 1]
        # style_encoder expects [0, 255] uint8
        style = (style + 1) * 127.5
        style = style.type(torch.uint8)

        style = self.style_encoder(style)
        style, embed = style.last_hidden_state, style.pooler_output

        if dropout_style:
            # if used, could potentially add noise here similar to unclip
            embed = (
                torch.bernoulli(
                    (1.0 - self.style_dropout)
                    * torch.ones(embed.shape[0], device=embed.device)[:, None]
                )
                * embed
            )
            # check whether this should really just be 0
            # or clip img embedding of "neutral" image

            # maybe noise here as well? maybe bit different though because
            # it's unpooled embedding
            style = (
                torch.bernoulli(
                    (1.0 - self.style_dropout)
                    * torch.ones(style.shape[0], device=style.device)[:, None, None]
                )
                * style
            )

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()

        if dropout_control:
            control = (
                torch.bernoulli(
                    (1.0 - self.control_dropout)
                    * torch.ones(control.shape[0], device=control.device)[
                        :, None, None, None
                    ]
                )
                * control
            )

        conditioning = dict(c_crossattn=[c], c_concat=[control])

        if style is not None:
            conditioning["c_style"] = [style]

        if embed is not None:
            conditioning["c_embed"] = [embed]

        return x, conditioning

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        control_txt = (
            torch.cat(cond["c_style"], 1) if self.has_style_stage else cond_txt
        )
        c_embed = torch.cat(cond["c_embed"], 1) if self.has_style_stage else None

        # here I could maybe add my own cross attention for images

        if cond["c_concat"] is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            control = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                timesteps=t,
                context=control_txt,
                style_embed=c_embed,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
            )

        return eps

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        unconditional_guidance_scale = self.guidance_scale

        log = dict()
        # we can take the style from batch but we need to know how it was preprocessed
        # in the cross attention case, there is no preprocessing because the model already does it
        # in the sum/concat case, it was normalized to be between -1 and 1

        z, c = self.get_input(
            batch,
            self.first_stage_key,
            bs=N,
            dropout_control=False,
            dropout_style=False,
        )
        # c_style, _ = self.get_input(batch, self.style_key, bs=N)

        # fix below lol
        # it's not only used for displaying  but also for sampling hehe

        if self.style_key:
            style_img = batch[self.style_key][:N]
            style_img = einops.rearrange(style_img, "b h w c -> b c h w")
            log["style"] = style_img

        if "source" in batch:
            style_img = batch["source"][:N]
            style_img = einops.rearrange(style_img, "b h w c -> b c h w")
            log["style"] = style_img

        c_cat, c_crossattn, c_style, c_embed = (
            c["c_concat"][0][:N],
            c["c_crossattn"][0][:N],
            None,
            None,
        )
        c_full = {"c_concat": [c_cat], "c_crossattn": [c_crossattn]}

        uc_cross = self.get_unconditional_conditioning(N)
        uc_cat = torch.zeros_like(c_cat)
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        if "c_style" in c:
            c_style = c["c_style"][0][:N]
            c_full["c_style"] = [c_style]
            uc_style = torch.zeros_like(c_style)
            uc_full["c_style"] = [uc_style]

        if "c_embed" in c:
            c_embed = c["c_embed"][0][:N]
            c_full["c_embed"] = [c_embed]
            uc_embed = torch.zeros_like(c_embed)
            uc_full["c_embed"] = [uc_embed]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["conditioning"] = log_txt_as_img(
            (512, 512), batch[self.cond_stage_key], size=16
        )
        log["control"] = c_cat * 2.0 - 1.0
        log["reconstruction"] = self.decode_first_stage(z)

        samples_cfg, _ = self.sample_log(
            cond=c_full,
            batch_size=N,
            ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc_full,
        )
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, batch_size, shape, cond, verbose=False, **kwargs
        )
        return samples, intermediates


class ConcatControlLDM(ControlLDM):
    def __init__(
        self,
        style_key,
        control_dropout=0,
        style_dropout=0,
        guidance_scale=7,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.style_key = style_key
        self.control_dropout = control_dropout
        self.style_dropout = style_dropout
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        bs=None,
        dropout_control=True,
        dropout_style=True,
        *args,
        **kwargs,
    ):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        c = c["c_crossattn"][0]

        style = batch[self.style_key]
        if bs is not None:
            style = style[:bs]
        style = style.to(self.device)
        style = einops.rearrange(style, "b h w c -> b c h w")
        style = style.to(memory_format=torch.contiguous_format).float()
        # style img is in [-1, 1]
        # lets move it to [0, 1] to match the other controlnet input
        style = (style + 1.0) / 2.0

        if dropout_style:
            # maybe noise here as well? maybe bit different though because
            # it's unpooled embedding
            style = (
                torch.bernoulli(
                    (1.0 - self.style_dropout)
                    * torch.ones(style.shape[0], device=style.device)[
                        :, None, None, None
                    ]
                )
                * style
            )

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()

        if dropout_control:
            control = (
                torch.bernoulli(
                    (1.0 - self.control_dropout)
                    * torch.ones(control.shape[0], device=control.device)[
                        :, None, None, None
                    ]
                )
                * control
            )

        # we simply concat style and control along the channel axis
        control = torch.cat([control, style], axis=1)

        conditioning = dict(c_crossattn=[c], c_concat=[control])

        return x, conditioning

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)

        if cond["c_concat"] is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            control = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                timesteps=t,
                context=cond_txt,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
            )

        return eps

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        unconditional_guidance_scale = self.guidance_scale

        log = dict()

        z, c = self.get_input(
            batch,
            self.first_stage_key,
            bs=N,
            dropout_control=False,
            dropout_style=False,
        )

        if self.style_key:
            style_img = batch[self.style_key][:N]
            style_img = einops.rearrange(style_img, "b h w c -> b c h w")
            log["style"] = style_img

        if "source" in batch:
            style_img = batch["source"][:N]
            style_img = einops.rearrange(style_img, "b h w c -> b c h w")
            log["style"] = style_img

        c_cat, c_crossattn = (
            c["c_concat"][0][:N],
            c["c_crossattn"][0][:N],
        )
        c_full = {"c_concat": [c_cat], "c_crossattn": [c_crossattn]}

        uc_cross = self.get_unconditional_conditioning(N)
        uc_cat = c_cat.clone()
        # only set the concated style image to 0
        # style is in [0, 1] so check if 0 makes sense
        uc_cat[:, 3:, :, :] = 0.0
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["conditioning"] = log_txt_as_img(
            (512, 512), batch[self.cond_stage_key], size=16
        )
        log["control"] = c_cat * 2.0 - 1.0
        log["reconstruction"] = self.decode_first_stage(z)

        samples_cfg, _ = self.sample_log(
            cond=c_full,
            batch_size=N,
            ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc_full,
        )
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, batch_size, shape, cond, verbose=False, **kwargs
        )
        return samples, intermediates


class ControlLDM(ImageEmbeddingConditionedLatentDiffusion):
    def __init__(
        self,
        control_stage_config,
        control_key,
        only_mid_control,
        control_dropout=0,
        txt_dropout=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.control_dropout = control_dropout
        self.txt_dropout = txt_dropout

    @torch.no_grad()
    def get_input(
        self, batch, k, bs=None, dropout_control=True, dropout_txt=True, *args, **kwargs
    ):
        # x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        inputs = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        x, c = inputs[0], inputs[1]
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]

        control = control.to(self.device)
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()

        if dropout_control:
            control = (
                torch.bernoulli(
                    (1.0 - self.control_dropout)
                    * torch.ones(control.shape[0], device=control.device)[
                        :, None, None, None
                    ]
                )
                * control
            )

        txt = c["c_crossattn"][0]
        if dropout_txt:
            keep_probs = torch.bernoulli(
                (1.0 - self.txt_dropout)
                * torch.ones(txt.shape[0], device=txt.device)[:, None, None]
            )
            txt = keep_probs * txt + (
                1 - keep_probs
            ) * self.get_unconditional_conditioning(txt.shape[0])

        c["c_concat"] = [control]
        c["c_crossattn"] = [txt]
        outputs = [x, c]
        outputs.extend(inputs[2:])
        return outputs

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        c_adm = cond["c_adm"]

        if cond["c_concat"] is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                y=c_adm,
                only_mid_control=self.only_mid_control,
            )
        else:
            control = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                timesteps=t,
                context=cond_txt,
                y=c_adm,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                y=c_adm,
                only_mid_control=self.only_mid_control,
            )

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=7.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        # z, c, x, xrec, xc = self.get_input(
        #     batch, self.first_stage_key, bs=N, dropout_embedding=False
        # )
        z, c = self.get_input(
            batch,
            self.first_stage_key,
            bs=N,
            dropout_embedding=False,
            dropout_control=False,
            dropout_txt=False,
        )
        c_cat, c_crossattn, c_adm = (
            c["c_concat"][0][:N],
            c["c_crossattn"][0][:N],
            c["c_adm"][:N],
        )
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        style_img = batch[self.embed_key]
        style_img = rearrange(style_img[:N], "b h w c -> b c h w")

        uc_cross = self.get_unconditional_conditioning(N)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_adm = torch.zeros_like(c_adm)
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "c_adm": uc_adm}
        samples_cfg, _ = self.sample_log(
            cond={"c_concat": [c_cat], "c_crossattn": [c_crossattn], "c_adm": c_adm},
            batch_size=N,
            ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc_full,
        )
        x_samples_cfg = self.decode_first_stage(samples_cfg)

        log = dict()
        # x and xrec are basically the same image
        # just one went through the auto-encoder
        # log["inputs"] = x
        log["control"] = (
            c_cat * 2.0 - 1.0
        )  # structure of x or xrec, structure frame at t (e.g. HED)
        log["style"] = style_img  # appearance frame at t-1
        log["conditioning"] = log_txt_as_img(
            (512, 512), batch[self.cond_stage_key], size=16
        )
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        log["reconstruction"] = self.decode_first_stage(z)  # xrec  # actual frame at t

        return log

    # @torch.no_grad()
    # def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
    #     ddim_sampler = DDIMSampler(self)
    #     b, c, h, w = cond["c_concat"][0].shape
    #     shape = (self.channels, h // 8, w // 8)
    #     samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
    #     return samples, intermediates

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
