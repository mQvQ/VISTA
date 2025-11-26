"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import (
    log_txt_as_img,
    exists,
    default,
    ismap,
    isimage,
    mean_flat,
    count_params,
    instantiate_from_config,
)
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import (
    normal_kl,
    DiagonalGaussianDistribution,
)
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)
from ldm.models.diffusion.ddim import DDIMSampler
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torchvision import transforms


__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}

from ldm.models.diffusion.ddpm import DDPM, disabled_train, uniform_on_device

class CellProportionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=26):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.LogSoftmax(dim=1)  # 使用LogSoftmax配合KL散度
        )

    def forward(self, x):
        return self.main(x)


class STLatentDiffusion(DDPM):
    """main class"""
    '''
    latent diffusion model for spatial transcriptomics data
    '''

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        track_fid=False,
        fid_path=None,
        st_size=16384,
        energy_weight=0.1,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.st_size = st_size
        self.energy_weight = energy_weight

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.n_classes = cond_stage_config.get("params", {}).get("n_classes", None)

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.validation_step_outputs = []

        for param in self.cond_stage_model.parameters():
            print(param.requires_grad)


    # ckpt_path == img_ckpt_path
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        )
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert self.scale_factor == 1.0, "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:

                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        st_x = batch['rna'] # b, 1, input_st_num
        st_x = st_x.to(memory_format=torch.contiguous_format).float()

        if bs is not None:
            st_x = st_x[:bs]
        st_x = st_x.to(self.device)
        st_z = st_x

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ["caption", "coordinates_bbox", 'conds', 'rna', 'unconds', 'image']:
                    xc = batch[cond_key]
                elif cond_key in ["class_label", "hybrid", 'multiconds']:
                    xc = batch
                    # Set label to null token with a probability
                    # Original paper uses p_uncond = 0.1
                    # p_uncond = 0.1
                    # xc['class_label'][np.random.rand(len(xc['class_label'])) < p_uncond] = self.n_classes-1
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = v_x
            # if not self.cond_stage_trainable or force_c_encode:
            #     if isinstance(xc, dict) or isinstance(xc, list):
            #         # import pudb; pudb.set_trace()
            #         c = self.get_learned_conditioning(xc)
            #     else:
            #         c = self.get_learned_conditioning(xc.to(self.device))
            # else:
            #     c = xc
            if isinstance(xc, dict) or isinstance(xc, list):
                # import pudb; pudb.set_trace()
                c = self.get_learned_conditioning(xc)
            else:
                c = self.get_learned_conditioning(xc.to(self.device))

            if bs is not None:
                if isinstance(c, list):
                    c[0] = c[0][:bs]
                    c[1] = c[1][:bs]

                c = c[:bs]
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, "pos_x": pos_x, "pos_y": pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {"pos_x": pos_x, "pos_y": pos_y}
        out = [st_z, c]
        if return_original_cond:
            out.append(xc)
        return out

    def shared_step(self, batch, **kwargs):
        st_x, c = self.get_input(batch, self.first_stage_key)
        loss = self(st_x, c)
        return loss

    def forward(self, st_x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (st_x.shape[0],), device=self.device).long()
        # print('check forward dtype', x.dtype, c.dtype)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(st_x, c, t, *args, **kwargs)

    def apply_model(self, st_x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = v_x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(v_x_noisy, ks, stride)

            z = unfold(v_x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if (
                self.cond_stage_key in ["image", "LR_image", "segmentation", "bbox_img"] and self.model.conditioning_key
            ):  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert len(c) == 1  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]


            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(
                output_list[0], tuple
            )  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            st_x_recon = self.model(st_x_noisy, t, **cond)

        return [st_x_recon]

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def get_energy_loss(self, pred, t, st_x_start, ):
        # calc consist between celltype-pred head output based on pred/st_x_start
        proj_head = CellProportionPredictor(input_dim=pred.shape[1])
        proj_head = proj_head.load_state_dict(torch.load('/home/tma/ST-Diffusion/proportion/best_model.pth')).to(self.device)
        proj_head.eval()
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        with torch.no_grad():
            pred_proj = proj_head(pred)
            target_proj = proj_head(st_x_start)
            loss = loss_fn(pred_proj, target_proj)
        return loss.mean()


    def get_loss(self, pred, target, mean=True, mask=None):
        '''

        :param pred: LIST OF multimodal prediction
        :param target:
        :param mean:
        :return:
        '''
        if self.loss_type == "l1":
            l_st = (target[0] - pred[0]).abs()
        elif self.loss_type == "l2":
            l_st = torch.nn.functional.mse_loss(target[0], pred[0], reduction="none")

        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        if l_st.ndim == 4:
            loss = l_st.mean([1, 2, 3])
        else: loss = l_st.mean([1, 2])
        if mean:
            loss = loss.mean()

        return loss

    def p_losses(self, st_x_start, cond, t, noise=None, mask=None):
        st_noise = default(noise, lambda: torch.randn_like(st_x_start))
        st_x_noisy = self.q_sample(x_start=st_x_start, t=t, noise=st_noise)
        model_output = self.apply_model(st_x_noisy, t, cond)
        loss_dict = {}

        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            st_target = st_x_start
        elif self.parameterization == "eps":
            st_target = st_noise
        else:
            raise NotImplementedError()
        target = [st_target]
        loss_simple = self.get_loss(model_output, target, mean=False, mask=mask)
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})
        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        # add energy loss
        # if t < 100:
        #     loss_energy = self.get_energy_loss(model_output, t, st_x_start)
        #     loss += self.energy_weight * loss_energy
        #     loss_dict.update({f"{prefix}/loss_energy": loss_energy})

        loss_vlb = self.get_loss(model_output, target, mean=False, mask=mask)
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def p_mean_variance(
        self,
        st_x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        st_out = self.apply_model(st_x, t_in, c, return_ids=return_codebook_ids)

        if self.parameterization == "eps":
            st_x_recon = self.predict_start_from_noise(st_x, t=t, noise=st_out)

        elif self.parameterization == "x0":
            st_x_recon = st_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            st_x_recon.clamp_(-1.0, 1.0)
        st_model_mean, st_posterior_variance, st_posterior_log_variance = self.q_posterior(x_start=st_x_recon, x_t=st_x, t=t)

        return st_model_mean, st_posterior_variance, st_posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        st_x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = *st_x.shape, st_x.device
        outputs = self.p_mean_variance(
            st_x=st_x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )

        st_model_mean, _, st_model_log_variance = outputs
        st_noise = noise_like(st_x.shape, device, repeat_noise) * temperature

        if noise_dropout > 0.0:
            st_noise = torch.nn.functional.dropout(st_noise, p=noise_dropout)
        # no noise when t == 0
        st_nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(st_x.shape) - 1)))

        return  st_model_mean + st_nonzero_mask * (0.5 * st_model_log_variance).exp() * st_noise

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        st_shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        st_x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        st_x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else v_shape[0]
            st_shape = [batch_size] + list(st_shape)
        else:
            b = batch_size = v_shape[0]
        if st_x_T is None:
            st = torch.randn(st_shape, device=self.device)
        else: st = st_x_T

        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive Generation",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            st, st_x0_partial = self.p_sample(
                st,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial) # todo: add supp. for st
            if callback:
                callback(i)

        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        v_shape,
        st_shape,
        return_intermediates=False,
        v_x_T=None,
            st_x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        v_x0=None,
            st_x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = v_shape[0]
        if v_x_T is None:
            img = torch.randn(v_shape, device=device)
        else:
            img = v_x_T

        if st_x_T is None:
            st = torch.randn(st_shape, device=device)
        else:
            st = st_x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert v_x0 is not None
            assert v_x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, st = self.p_sample(
                img,
                st,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
            )
            if mask is not None:
                img_orig = self.q_sample(v_x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
                intermediates.append(st)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, st, intermediates
        return img, st

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(
            cond,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            st_shape = self.st_size
            shape = st_shape
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def calculate_activation_statistics(self, images, dims=2048, batch_size=32):
        model = self.inception

        pred_arr = np.empty((len(images), dims))
        start_idx = 0
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            input_tensor = torch.stack([train_transforms(img) for img in chunk]).to(self.device)

            pred = model(input_tensor)[0]

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx: start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        super().validation_step(batch, batch_idx)
        st_x, c = self.get_input(batch, self.first_stage_key)

        samples, _ = self.sample_log(
            cond=c,
            batch_size=st_x.shape[0],
            ddim=True,
            ddim_steps=50,
            eta=1,
            use_tqdm=False,
        )
        self.validation_step_outputs.append(samples)

    @torch.no_grad()
    def on_validation_epoch_end(self):

        out = self.validation_step_outputs
        out = torch.cat(out, dim=0)
        print('sample st mean & std:', out.flatten().mean(), out.flatten().std())


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x



# class Layout2STDiffusion(STLatentDiffusion):
#     # TODO: move all layout-specific hacks to this class
#     def __init__(self, cond_stage_key, *args, **kwargs):
#         assert cond_stage_key == "coordinates_bbox", 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
#         super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)
#
#     def log_images(self, batch, N=8, *args, **kwargs):
#
#         return logs
