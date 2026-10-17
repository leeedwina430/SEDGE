# Code for the real-world experiments in _SEDGE: Structural Extrapolated Data Generation_

This repository contains the core model components used for the real-world inference pipeline for the submission. This snapshot is intentionally minimal: it provides the diffusion prior, the conditional likelihood model, and the diffusion posterior sampling pipeline that combines them for guided generation in latent concept space.

The real-world dataset is based on a modified version of a public text-to-image reasoning dataset, FLUX-Reason-6M. The concepts used in this repository are obtained from the ConceptAligner model, fine-tuned on the modified dataset; see the paper references for details.

## Layout

```
SEDGE/
├── inference.py         # DPSInferencePipeline: loads checkpoints and runs guided sampling
├── model_prior.py       # ConceptPriorDiffusion: diffusion prior over image concepts
├── model_likelihood.py  # ConceptFlowModel: conditional likelihood p(z_T | z_I)
└── README.md
```

## What this code covers

The real-world pipeline is organized into three pieces:

- `ConceptPriorDiffusion` in `model_prior.py` is a DiT-style diffusion prior over image-concept latents `z_I`.
- `ConceptFlowModel` in `model_likelihood.py` models the text-conditioned likelihood term used for guidance.
- `DPSInferencePipeline` in `inference.py` combines the two models with a DDPM or DDIM scheduler and performs diffusion posterior sampling (DPS).

This repository is therefore the model-definition and inference layer. It expects external assets such as trained checkpoints, normalization statistics, and upstream latent features to already exist.

## Checkpoint and input requirements

To run inference you need:

- a prior checkpoint compatible with `ConceptPriorDiffusion`
- a likelihood checkpoint compatible with `ConceptFlowModel`
- a stats file containing `zI_mean`, `zI_std`, `zT_mean`, and `zT_std`
- target text latents `target_z_T` with shape `(batch, 64, 768)`

The sampled output is an image-concept latent tensor with shape `(batch, 64, 64)` after unnormalization.

## Inference workflow

The intended workflow is:

1. Load the prior model checkpoint.
2. Load the conditional likelihood model checkpoint.
3. Normalize the target text concepts with the saved dataset statistics.
4. Start from Gaussian latent noise.
5. Run reverse diffusion while adding the likelihood gradient at every step.
6. Unnormalize the final image-concept latents.

The implementation for this procedure lives in `DPSInferencePipeline.dps_sampling`.

## Training setup

For the text-to-image experiments, we trained the prior and likelihood models on AMD Instinct MI210 GPUs with 64 GB of memory per GPU. Prior training and likelihood training each took on the order of half a day.

## Module details

### `model_prior.py`

Defines `ConceptPriorDiffusion`, a transformer-based diffusion prior with:

- sinusoidal timestep embeddings
- adaptive LayerNorm conditioning in each transformer block
- learned positional embeddings across the concept sequence

The forward pass takes noisy concept latents `x` and diffusion steps `t`, and predicts the noise residual used by the scheduler.

### `model_likelihood.py`

Defines `ConceptFlowModel`, which uses one conditional normalizing flow per text concept. Each flow scores one `z_T[i]` against a masked view of the image concepts `z_I`, and the model returns:

- the average negative log-likelihood
- the learned soft concept mask

This likelihood is differentiated during DPS to guide sampling toward latents that better explain the target text concepts.

### `inference.py`

Defines `DPSInferencePipeline`, which:

- instantiates the prior and likelihood models
- loads checkpoints and normalization statistics
- configures either a DDPM or DDIM scheduler
- performs gradient-guided reverse diffusion

## Scope note

This repository snapshot does not include dataset preparation, representation extraction, checkpoint training scripts, or evaluation notebooks. Those pieces are assumed to live in the broader real-world experiment pipeline.
