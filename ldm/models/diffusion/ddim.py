"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from ldm.util import default

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_tqdm=False,
        sample_uncond_separate=False,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        # if conditioning is not None:
        #     if isinstance(conditioning, dict):
        #         if isinstance(conditioning[list(conditioning.keys())[0]], dict):
        #             cbs = conditioning[list(conditioning.keys())[0]]["image"].shape[0]
        #
        #         cbs = conditioning[list(conditioning.keys())[0]].shape[0]
        #         if cbs != batch_size:
        #             print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        #     else:
        #         if hasattr(conditioning, "shape") and conditioning.shape[0] != batch_size:
        #             print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        size = tuple([batch_size] + list(shape))
        # C, H, W = shape
        # size = (batch_size, C, H, W)
        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            sample_uncond_separate=sample_uncond_separate,
            use_tqdm=use_tqdm,
        )
        return samples, intermediates
    
    @torch.no_grad()
    def sample_denoise(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        timesteps=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_tqdm=False,
        sample_uncond_separate=False,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        # if conditioning is not None:
        #     if isinstance(conditioning, dict):
        #         if isinstance(conditioning[list(conditioning.keys())[0]], dict):
        #             cbs = conditioning[list(conditioning.keys())[0]]["image"].shape[0]
        #
        #         cbs = conditioning[list(conditioning.keys())[0]].shape[0]
        #         if cbs != batch_size:
        #             print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        #     else:
        #         if hasattr(conditioning, "shape") and conditioning.shape[0] != batch_size:
        #             print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        size = tuple([batch_size] + list(shape))
        # C, H, W = shape
        # size = (batch_size, C, H, W)
        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            timesteps=timesteps,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            sample_uncond_separate=sample_uncond_separate,
            use_tqdm=use_tqdm,
        )
        return samples, intermediates


    @torch.no_grad()
    def joint_condition_sample( # joint generation 有部分是非随机初始化
            self,
            S,
            batch_size,
            v_shape,
            st_shape,
            conditioning=None,
            callback=None,
            normals_sequence=None,
            img_callback=None,
            quantize_x0=False,
            eta=0.0,
            v_mask=None,
            st_mask=None,
            v_x0=None,
            st_x0=None,
            temperature=1.0,
            noise_dropout=0.0,
            score_corrector=None,
            corrector_kwargs=None,
            timesteps=None,
            verbose=True,
            v_x_T=None,
            st_x_T=None,
            log_every_t=100,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            use_tqdm=False,
            sample_uncond_separate=False,
            class_scale=3.0,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if hasattr(conditioning, "shape") and conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = v_shape
        v_size = (batch_size, C, H, W)
        C_st, L = st_shape
        st_size = (batch_size, C_st, L)

        samples, intermediates = self.joint_ddim_condition_sampling(
            conditioning,
            v_size,
            st_size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            v_mask=v_mask,
            st_mask=st_mask,
            v_x0=v_x0,
            st_x0=st_x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            v_x_T=v_x_T,
            st_x_T=st_x_T,
            timesteps=timesteps,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            sample_uncond_separate=sample_uncond_separate,
            use_tqdm=use_tqdm,
            class_scale=class_scale
        )
        return samples, intermediates

    @torch.no_grad()
    def joint_condition_sample_v2( # joint generation 有部分是非随机初始化
            self,
            S,
            batch_size,
            v_shape,
            st_shape,
            conditioning=None,
            callback=None,
            normals_sequence=None,
            img_callback=None,
            quantize_x0=False,
            eta=0.0,
            v_mask=None,
            st_mask=None,
            v_x0=None,
            st_x0=None,
            temperature=1.0,
            noise_dropout=0.0,
            score_corrector=None,
            corrector_kwargs=None,
            verbose=True,
            v_x_T=None,
            st_x_T=None,
            log_every_t=100,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            use_tqdm=False,
            sample_uncond_separate=False,
            class_scale=3.0,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if hasattr(conditioning, "shape") and conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = v_shape
        v_size = (batch_size, C, H, W)
        C_st, L = st_shape
        st_size = (batch_size, C_st, L)

        samples, intermediates = self.joint_ddim_condition_sampling_v2(
            conditioning,
            v_size,
            st_size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            v_mask=v_mask,
            st_mask=st_mask,
            v_x0=v_x0,
            st_x0=st_x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            v_x_T=v_x_T,
            st_x_T=st_x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            sample_uncond_separate=sample_uncond_separate,
            use_tqdm=use_tqdm,
            class_scale=class_scale
        )
        return samples, intermediates


    @torch.no_grad()
    def joint_sample(
            self,
            S,
            batch_size,
            v_shape,
            st_shape,
            conditioning=None,
            callback=None,
            normals_sequence=None,
            img_callback=None,
            quantize_x0=False,
            eta=0.0,
            v_mask=None,
            st_mask=None,
            v_x0=None,
            st_x0=None,
            temperature=1.0,
            noise_dropout=0.0,
            score_corrector=None,
            corrector_kwargs=None,
            verbose=True,
            v_x_T=None,
            st_x_T=None,
            log_every_t=100,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            use_tqdm=False,
            sample_uncond_separate=False,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if hasattr(conditioning, "shape") and conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = v_shape
        v_size = (batch_size, C, H, W)
        C_st, L = st_shape
        st_size = (batch_size, C_st, L)

        samples, intermediates = self.joint_ddim_sampling(
            conditioning,
            v_size,
            st_size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            v_mask=v_mask,
            st_mask=st_mask,
            v_x0=v_x0,
            st_x0=st_x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            v_x_T=v_x_T,
            st_x_T=st_x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            sample_uncond_separate=sample_uncond_separate,
            use_tqdm=use_tqdm,
        )
        return samples, intermediates
    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_tqdm=True,
        sample_uncond_separate=False,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None: # denoise 起点
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            # 0-49
            # ddim_timesteps:[  1  21  41  61  81 101 121 141 161 181 201 221 241 261 281 301 321 341 361 381 401 421 441 461 481 501 521 541 561 581 601 621 641 661 681 701 721 741 761 781 801 821 841 861 881 901 921 941 961 981]
            print('timesteps', timesteps)
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        
        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print('time_range', time_range)
        # print('ddim_use_original_steps', ddim_use_original_steps)
        # print('timesteps', timesteps)
        if use_tqdm:
            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        else:
            iterator = time_range

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                sample_uncond_separate=sample_uncond_separate,
            )
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    def joint_ddim_condition_sampling(self,
        cond,
        v_shape,
        st_shape,
        v_x_T = None,
        st_x_T = None,
        ddim_use_original_steps = False,
        callback = None,
        timesteps = None,
        quantize_denoised = False,
        v_mask = None,
        st_mask = None,
        v_x0 = None,
        st_x0 = None,
        img_callback = None,
        log_every_t = 100,
        temperature = 1.0,
        noise_dropout = 0.0,
        score_corrector = None,
        corrector_kwargs = None,
        unconditional_guidance_scale = 1.0,
        class_scale=3.0,
        unconditional_conditioning = None,
        use_tqdm = True,
        sample_uncond_separate = False,

    ):
        '''
            text-condition + image-condition predict st, or
            text-condition + st-condition predict image, etc
        '''
        device = self.model.betas.device
        b = v_shape[0]
        if v_x_T is None:
            img = torch.randn(v_shape, device=device)
        else:
            img = v_x_T

        if st_x_T is None:
            st = torch.randn(st_shape, device=device)
        else:
            st = st_x_T
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img], 'st_x_inter': [st], 'st_pred_x0': [st]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        if use_tqdm:
            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        else:
            iterator = time_range

        class_scale = class_scale

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if st_mask is not None:
                assert st_x0 is not None
                st_orig = self.model.q_sample(st_x0, ts)  # TODO: deterministic forward pass?
                st = st_orig * st_mask + (1.0 - st_mask) * st

            if v_x0 is not None:  # image condition
                v_x0 = v_x0.to(device)
                img = self.model.q_sample(v_x0, ts)
                previous_step_condition = self.model.q_sample(v_x0, ts-1)
                # st.requires_grad_()

            if st_x0 is not None and st_mask is None and v_x0 is None:  # st cond -> image generation
                st_x0 = st_x0.to(device)
                st = self.model.q_sample(st_x0, ts)
                previous_step_condition = self.model.q_sample(st_x0, ts - 1)
                # img.requires_grad_()
            with torch.enable_grad():
                if v_x0 is not None:
                    st = st.detach().requires_grad_()
                if st_x0 is not None and st_mask is None:
                    img = img.detach().requires_grad_()
                outs = self.joint_p_sample_ddim(
                    img,
                    st,
                    cond,
                    ts,
                    index=index,
                    use_original_steps=ddim_use_original_steps,
                    quantize_denoised=quantize_denoised,
                    temperature=temperature,
                    noise_dropout=noise_dropout,
                    score_corrector=score_corrector,
                    corrector_kwargs=corrector_kwargs,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    sample_uncond_separate=sample_uncond_separate,
                )
                out_img, pred_x0, out_st, st_pred_x0 = outs
                # img, pred_x0, st, st_pred_x0 = outs
                loss_scale = 1.
                if v_x0 is not None:  # target: st; visual condition st gen
                    none_zero_mask = (ts != 0).float().view(-1, *([1] * (len(st.shape) - 1)))
                    loss = mean_flat((out_img - previous_step_condition) ** 2)
                    # 释放计算图
                    grad = torch.autograd.grad(loss.mean() * loss_scale, st, retain_graph=True)[0]
                    st = out_st - none_zero_mask * grad * class_scale * self.model.sqrt_alphas_cumprod[
                        i]
                if st_x0 is not None and st_mask is None:  # target: image; 且非st-imputation任务
                    none_zero_mask = (ts != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                    loss = mean_flat((out_st - previous_step_condition) ** 2)
                    grad = torch.autograd.grad(loss.mean() * loss_scale, img, retain_graph=True)[0]
                    img = out_img - none_zero_mask * grad * class_scale * self.model.sqrt_alphas_cumprod[i]



                if callback:
                    callback(i)
                if img_callback:
                    img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates["x_inter"].append(img)
                    intermediates["pred_x0"].append(pred_x0)
                    intermediates["st_x_inter"].append(st)
                    intermediates['st_pred_x0'].append(st_pred_x0)

        return [img, st], intermediates

    def joint_ddim_condition_sampling_v2(self,
        cond,
        v_shape,
        st_shape,
        v_x_T = None,
        st_x_T = None,
        ddim_use_original_steps = False,
        callback = None,
        timesteps = None,
        quantize_denoised = False,
        v_mask = None,
        st_mask = None,
        v_x0 = None,
        st_x0 = None,
        img_callback = None,
        log_every_t = 100,
        temperature = 1.0,
        noise_dropout = 0.0,
        score_corrector = None,
        corrector_kwargs = None,
        unconditional_guidance_scale = 1.0,
        class_scale=3.0,
        unconditional_conditioning = None,
        use_tqdm = True,
        sample_uncond_separate = False,

    ):
        '''
            text-condition + image-condition predict st, or
            text-condition + st-condition predict image, etc
        '''
        device = self.model.betas.device
        b = v_shape[0]
        if v_x_T is None:
            img = torch.randn(v_shape, device=device)
        else:
            img = v_x_T

        if st_x_T is None:
            st = torch.randn(st_shape, device=device)
        else:
            st = st_x_T
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img], 'st_x_inter': [st], 'st_pred_x0': [st]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        if use_tqdm:
            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        else:
            iterator = time_range

        class_scale = class_scale
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if st_mask is not None:  # imputation 的情况
                assert st_x0 is not None
                st_orig = self.model.q_sample(st_x0, ts)  # TODO: deterministic forward pass?
                st = st_orig * st_mask + (1.0 - st_mask) * st

            if v_x0 is not None:  # image condition
                v_x0 = v_x0.to(device)
                # img = self.model.q_sample(v_x0, ts)
                previous_step_condition = self.model.q_sample(v_x0, ts-1)
                st.requires_grad_()

            if st_x0 is not None and st_mask is None:  # image cond -> st prediction
                st_x0 = st_x0.to(device)
                st = self.model.q_sample(st_x0, ts)
                previous_step_condition = self.model.q_sample(st_x0, ts - 1)
                img.requires_grad_()

            with torch.enable_grad():
                outs = self.joint_p_sample_ddim(
                    img,
                    st,
                    cond,
                    ts,
                    index=index,
                    use_original_steps=ddim_use_original_steps,
                    quantize_denoised=quantize_denoised,
                    temperature=temperature,
                    noise_dropout=noise_dropout,
                    score_corrector=score_corrector,
                    corrector_kwargs=corrector_kwargs,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    sample_uncond_separate=sample_uncond_separate,
                )
                out_img, pred_x0, out_st, st_pred_x0 = outs
                # img, pred_x0, st, st_pred_x0 = outs
                loss_scale = 1.
                if v_x0 is not None:  # target: st; visual condition st gen
                    none_zero_mask = (ts != 0).float().view(-1, *([1] * (len(st.shape) - 1)))
                    loss = mean_flat((out_img - previous_step_condition) ** 2)
                    grad = torch.autograd.grad(loss.mean() * loss_scale, st)[0]
                    st = out_st - none_zero_mask * grad * class_scale * self.model.sqrt_alphas_cumprod[
                        i]
                    img = out_img

                if st_x0 is not None and st_mask is None:  # target: image; 且非st-imputation任务
                    none_zero_mask = (ts != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                    loss = mean_flat((out_st - previous_step_condition) ** 2)
                    grad = torch.autograd.grad(loss.mean() * loss_scale, img)[0]
                    img = out_img - none_zero_mask * grad * class_scale * self.model.sqrt_alphas_cumprod[i]

                if callback:
                    callback(i)
                if img_callback:
                    img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates["x_inter"].append(img)
                    intermediates["pred_x0"].append(pred_x0)
                    intermediates["st_x_inter"].append(st)
                    intermediates['st_pred_x0'].append(st_pred_x0)

        return [img, st], intermediates



    def joint_ddim_sampling(
        self,
        cond,
        v_shape,
        st_shape,
        v_x_T=None,
        st_x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        v_mask=None,
        st_mask=None,
        v_x0=None,
        st_x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_tqdm=True,
        sample_uncond_separate=False,
    ):
        device = self.model.betas.device
        b = v_shape[0]
        if v_x_T is None:
            img = torch.randn(v_shape, device=device)
        else:
            img = v_x_T

        if st_x_T is None:
            st = torch.randn(st_shape, device=device)
        else:
            st = st_x_T
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img], 'st_x_inter': [st], 'st_pred_x0': [st]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        if use_tqdm:
            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        else:
            iterator = time_range

        class_scale=3.0
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if st_mask is not None: # imputation 的情况
                assert st_x0 is not None
                st_orig = self.model.q_sample(st_x0, ts)  # TODO: deterministic forward pass?
                st = st_orig * st_mask + (1.0 - st_mask) * st

            outs = self.joint_p_sample_ddim(
                img,
                st,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                sample_uncond_separate=sample_uncond_separate,
            )
            # out_img, pred_x0, out_st, st_pred_x0 = outs
            img, pred_x0, st, st_pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)
                intermediates["st_x_inter"].append(st)
                intermediates['st_pred_x0'].append(st_pred_x0)

        return [img, st],  intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        sample_uncond_separate=False,
    ):
        b, *_, device = *x.shape, x.device
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(x, t, c)
        else:
            if sample_uncond_separate:
                e_t = self.model.apply_model(x, t, c)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(c, dict):
                    pass
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                tmp_pred = self.model.apply_model(x_in, t_in, c_in)
                if isinstance(tmp_pred, list): tmp_pred = tmp_pred[0]

                e_t_uncond, e_t = tmp_pred.chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        if isinstance(e_t, list): e_t = e_t[0]
        if x.dim() == 3:
            a_t = torch.full((b, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1), sqrt_one_minus_alphas[index], device=device)
        else:
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    def joint_p_sample_ddim(
            self,
            v_x,
            st_x,
            c,
            t,
            index,
            repeat_noise=False,
            use_original_steps=False,
            quantize_denoised=False,
            temperature=1.0,
            noise_dropout=0.0,
            score_corrector=None,
            corrector_kwargs=None,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            sample_uncond_separate=False,
    ):
        b, *_, device = *v_x.shape, v_x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            v_e_t, st_e_t = self.model.apply_model(v_x, st_x, t, c)
        else:
            if sample_uncond_separate:
                v_e_t, st_e_t = self.model.apply_model(v_x, st_x, t, c)
                v_e_t_uncond, st_e_t_uncond = self.model.apply_model(v_x, st_x, t, unconditional_conditioning)
            else:
                v_x_in = torch.cat([v_x] * 2)
                st_x_in = torch.cat([st_x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                v_out, st_out = self.model.apply_model(v_x_in, st_x_in, t_in, c_in)
                v_e_t_uncond, v_e_t = v_out.chunk(2)
                st_e_t_uncond, st_e_t = st_out.chunk(2)

            v_e_t = v_e_t_uncond + unconditional_guidance_scale * (v_e_t - v_e_t_uncond)
            st_e_t = st_e_t_uncond + unconditional_guidance_scale * (st_e_t - st_e_t_uncond)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        # current prediction for x_0
        pred_x0 = (v_x - sqrt_one_minus_at * v_e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
        v_dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * v_e_t
        v_noise = sigma_t * noise_like(v_x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            v_noise = torch.nn.functional.dropout(v_noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + v_dir_xt + v_noise
        # for st generation
        a_t = a_t.squeeze(1)
        a_prev = a_prev.squeeze(1)
        sigma_t = sigma_t.squeeze(1)
        sqrt_one_minus_at = sqrt_one_minus_at.squeeze(1)
        st_pred_x0 = (st_x - sqrt_one_minus_at * st_e_t) / a_t.sqrt()
        st_dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * st_e_t
        st_noise = sigma_t * noise_like(st_x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            st_noise = torch.nn.functional.dropout(st_noise, p=noise_dropout)

        st_x_prev = a_prev.sqrt() * st_pred_x0 + st_dir_xt + st_noise

        return x_prev, pred_x0, st_x_prev, st_pred_x0

