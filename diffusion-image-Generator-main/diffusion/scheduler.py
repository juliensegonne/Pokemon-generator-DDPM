import torch
import math

# def make_beta_schedule(schedule="linear", timesteps=1000, start=1e-4, end=0.02):
#     if schedule == "linear":
#         return torch.linspace(start, end, timesteps)
#     elif schedule == "cosine":
#         # Cosine schedule comme dans DDPM amélioré
#         steps = timesteps + 1
#         t = torch.linspace(0, timesteps, steps) / timesteps
#         alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
#         alphas_cumprod = alphas_cumprod / alphas_cumprod[0]         # Assurer que alpha_1 (= alphas_cumprod[0])= 1 (pas de bruit au début)
#         betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#         betas = betas * (end - start) + start
#         return torch.clamp(betas, start, end)  
#     elif schedule == "quadratic":
#         # Schedule quadratique
#         betas = torch.linspace(start, end, timesteps)
#         return betas ** 2


# Version inspirer from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
def make_beta_schedule(schedule="linear", timesteps=1000, start=1e-4, end=0.02):
    if schedule == "linear":
        scale = 1000 / timesteps
        beta_start = start * scale
        beta_end = end * scale
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == "cosine":
        # Cosine schedule comme dans DDPM amélioré
        # portion du cosinus ou la courbe est croissante et convexe
        alpha_bar = lambda t : math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(timesteps):
            t1 = i / timesteps
            t2 = (i + 1) / timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        betas = torch.tensor(betas, dtype=torch.float32)
        return betas
    elif schedule == "quadratic":
        # Schedule quadratique
        betas = torch.linspace(start, end, timesteps)
        return betas ** 2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule}")