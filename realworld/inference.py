import torch
from diffusers import DDPMScheduler, DDIMScheduler

from model_prior import ConceptPriorDiffusion 
from model_likelihood import ConceptFlowModel


class DPSInferencePipeline:
    def __init__(self, prior_ckpt, likelihood_ckpt, model, model_dim, depth, stats_path=None, scheduler_type='ddpm', device='cuda'):
        self.device = device
        
        # Load the diffusion prior model
        print(f"Loading Prior from {prior_ckpt}...")
        self.prior_model = ConceptPriorDiffusion(
                                                num_concepts=64, 
                                                concept_dim=64,
                                                hidden_size=1024, 
                                                depth=12, 
                                                num_heads=8
                                            ).to(device)
        self.prior_model.load_state_dict(torch.load(prior_ckpt, map_location=device))
        self.prior_model.eval()

        # The scheduler controls the reverse-diffusion update rule used during inference.
        scheduler_kwargs = {
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "prediction_type": "epsilon",
            "clip_sample": False
        }
        
        if scheduler_type == 'ddim':
            print("Using DDIMScheduler")
            self.scheduler = DDIMScheduler(**scheduler_kwargs)
        else:
            print("Using DDPMScheduler")
            self.scheduler = DDPMScheduler(**scheduler_kwargs)


        # Load the likelihood model
        print(f"Loading Likelihood from {likelihood_ckpt}...")
        print(f"model={model}")
        if model == "cflow":
            MODELNAME = ConceptFlowModel

        self.likelihood_model = MODELNAME(
                                        num_image_concepts=64,
                                        num_text_concepts=64,
                                        image_dim=64, 
                                        text_dim=768,
                                        flow_hidden=model_dim,
                                        flow_layers=depth
                                    ).to(device)

        self.likelihood_model.load_state_dict(torch.load(likelihood_ckpt, map_location=device))
        self.likelihood_model.eval()

        # Optional statistics are used to normalize targets and unnormalize the final image latents.
        if stats_path:
            stats = torch.load(stats_path)
            self.zI_mean = stats["zI_mean"].unsqueeze(0).to(device)
            self.zI_std = stats["zI_std"].unsqueeze(0).to(device)
            self.zT_mean = stats["zT_mean"].unsqueeze(0).to(device)
            self.zT_std = stats["zT_std"].unsqueeze(0).to(device)


    @torch.enable_grad()
    def dps_sampling(
        self,
        target_z_T,
        scale=1.0,
        num_inference_steps=None,
        seed=None,
        seeds=None,
    ):
        # Accept either a single seed or a list of seeds for batched sampling.
        if seeds is None:
            assert seed is not None
            seeds = [seed]

        seeds = [int(s) for s in seeds]

        B_one = target_z_T.shape[0]
        num_seed_batch = len(seeds)

        # Normalize the target text concepts using the stored dataset statistics.
        target_z_T = target_z_T.to(self.device)
        target_z_T = (target_z_T - self.zT_mean) / (self.zT_std + 1e-6)

        # Repeat the conditioning target when sampling multiple seeds in parallel.
        if num_seed_batch > 1:
            target_z_T = target_z_T.repeat(num_seed_batch, 1, 1)

        B_total = target_z_T.shape[0]
        shape = (B_total, 64, 64)

        # Default to the full reverse-diffusion schedule when no custom step count is given.
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.config.num_train_timesteps

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Convert the guidance scale into a tensor so it can be broadcast per sample.
        if isinstance(scale, (int, float)):
            scale_tensor = torch.full(
                (B_total, 1, 1),
                float(scale),
                device=self.device,
            )
        else:
            scale = scale.to(self.device)

            if scale.dim() == 1:
                if num_seed_batch > 1:
                    scale = scale.repeat(num_seed_batch)
                scale_tensor = scale.view(B_total, 1, 1)
            else:
                if num_seed_batch > 1:
                    scale = scale.repeat(num_seed_batch, 1, 1)
                scale_tensor = scale.view(B_total, 1, 1)

        # Start from seeded Gaussian noise in latent space.
        latent_chunks = []
        for sd in seeds:
            g = torch.Generator(device=self.device).manual_seed(sd)
            latent_chunks.append(
                torch.randn(
                    (B_one, 64, 64),
                    generator=g,
                    device=self.device,
                )
            )

        latents = torch.cat(latent_chunks, dim=0)
        assert latents.shape == shape

        for step_idx, t in enumerate(timesteps):
            # Re-enable gradients because the likelihood term differentiates through the current latent.
            latents = latents.detach().requires_grad_(True)
            B = latents.shape[0]

            # Predict the noise residual and recover the denoised sample estimate.
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            t_batch = torch.full(
                (B_total,),
                t.item(),
                device=self.device,
                dtype=torch.long,
            )

            noise_pred = self.prior_model(latent_model_input, t_batch)

            step_output = self.scheduler.step(noise_pred, t, latents)
            x_prev = step_output.prev_sample
            x_0_hat = step_output.pred_original_sample

            # Compute the likelihood guidance gradient toward the target text concepts.
            loss = self.likelihood_model(zI=x_0_hat, zT=target_z_T)[0] * B_total
            grad = torch.autograd.grad(loss, latents)[0]

            # Clip very large per-sample gradients so the guidance update stays stable.
            max_grad_norm = 1.0 
            individual_norms = grad.view(B, -1).norm(dim=1, keepdim=True).view(B, 1, 1)
            grad = torch.where(individual_norms > max_grad_norm, 
                            grad * (max_grad_norm / (individual_norms + 1e-8)), 
                            grad)

            # Apply the guidance step to the reverse-diffusion trajectory.
            latents = x_prev - scale_tensor * grad
        # Convert the final latent sample back to the original image-concept scale.
        latents = latents * self.zI_std + self.zI_mean
        return latents


