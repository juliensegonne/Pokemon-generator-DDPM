# DDPM Pokémon Generator – Project Summary

## Authors

**Alexandre Dréan, Julien Segonne, Ryan Kaddour**  
Affiliations: Polytechnique Montréal, ENSTA Paris, Centrale Méditerranée  
GitHub: [github.com/Alexndrs/diffusion-image-Generator](https://github.com/Alexndrs/diffusion-image-Generator)

## Overview

This project explores Denoising Diffusion Probabilistic Models (DDPM) and their improved variants to generate 64×64 Pokémon-like images from noise. The work is part of a reproduction effort and aims to deepen theoretical and practical understanding of diffusion-based generative models.

### Data sample
<div style="display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(4, 1fr); gap: 10px; aspect-ratio: 3/4;">
    <img src="Data sample/Poke1.png" width="10%">
    <img src="Data sample/Poke2.png" width="10%">
    <img src="Data sample/Poke3.png" width="10%">

## Project Structure

### 1. Data Preprocessing

- Used a publicly available Pokémon dataset (~10K images).
- Applied normalization to [−1,1] and multiple augmentations: flipping, rotation, jitter, zoom, and optional noise.
- Ensured data diversity and training stability.

### 2. UNet Architecture

- Implemented a basic UNet with sinusoidal time embeddings to condition the model on diffusion steps.
- Improved UNet using:
  - Multi-head attention (4 heads),
  - Feature-wise linear modulation (FiLM),
  - Gradient checkpointing,
  - Mixed precision training (float16/32).

### 3. Diffusion Modeling

- Followed the DDPM framework from Ho et al. (2020) and improvements by Nichol & Dhariwal (2021).
- Used cosine noise schedule to better handle low-resolution images.
- Training loss was simplified to predicting added noise ϵ, leading to an MSE objective.

## Implementation

- All algorithms were implemented from scratch in:
  - `diffusion/`, `model/`, `scripts/`
- Verified correctness with unit tests and visualizations of noise schedules.
- Training with the initial implementation yielded poor results due to scheduler and clipping issues.
- Switched to OpenAI's optimized implementation from improved-diffusion GitHub repo for better performance.

## Results

### With Own Implementation

- Training plateaued quickly.
- Generated images were blurry or over/under-saturated.

### With Improved DDPM

- Training over 120 epochs showed continuous improvement.
- Generated samples closely resembled stylized Pokémon creatures.
- Evaluated by LLMs (ChatGPT, Gemini, Claude, Mistral) which:
  - Often recognized creature-like features and related them to games like Pokémon or Digimon.
  - Confirmed that visual features aligned with the dataset's style.


<div style="display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(4, 1fr); gap: 10px; aspect-ratio: 3/4;">
    <img src="Final results/1.png" width="10%">
    <img src="Final results/2.png" width="10%">
    <img src="Final results/3.png" width="10%">
    <img src="Final results/4.png" width="10%">
    <img src="Final results/5.png" width="10%">
    <img src="Final results/6.png" width="10%">
    <img src="Final results/7.png" width="10%">
    <img src="Final results/8.png" width="10%">
    <img src="Final results/9.png" width="10%">
    <img src="Final results/10.png" width="10%">
    <img src="Final results/11.png" width="10%">
    <img src="Final results/12.png" width="10%">
</div>

## Key Takeaways

- DDPMs are powerful yet sensitive to hyperparameters and implementation details (e.g., scheduler, clipping).
- The improved DDPM with cosine scheduler produced visually coherent and interpretable Pokémon-like images.

## References

- Ho et al., 2020 (DDPM)
- Nichol & Dhariwal, 2021 (Improved DDPM)
- Ronneberger et al., 2015 (UNet)
- Sohl-Dickstein et al., 2015 (Langevin Dynamics)
