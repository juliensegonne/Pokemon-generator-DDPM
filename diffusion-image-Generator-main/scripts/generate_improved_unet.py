# 1. Importer le modèle DDPM entrainé
# 2. Créer des images à partir du modèle DDPM et les enregistrer dans le dossier ./genrations

from preprocessing.dataset import DatasetLoader, download_kaggle_dataset
from model.improved_unet import UNetModel
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.scheduler import make_beta_schedule
import torch
from PIL import Image
import torchvision.transforms as T
import os
import subprocess
import time
import numpy as np

# Fonction pour convertir un tensor en image PIL
def tensor_to_pil(tensor):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)
    # Dénormaliser de [-1, 1] à [0, 1]
    tensor = (tensor + 1) / 2
    # Clamp pour s'assurer que les valeurs sont entre 0 et 1
    tensor = torch.clamp(tensor, 0, 1)
    # Convertir en image PIL
    return T.ToPILImage()(tensor)


def save_sample_sequence(sample_sequence, seed_dir):
    os.makedirs(seed_dir, exist_ok=True)
    for i, img in enumerate(sample_sequence):
        img = tensor_to_pil(img)
        img.save(f"{seed_dir}/sample_{i:03d}.png")


def create_video_from_sequence(seed_dir):
    output_video_path = f"{seed_dir}/video.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        # "-loglevel", "error"
        "-framerate", "50",
        "-i", f"{seed_dir}/sample_%03d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video_path
    ]
    # subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    subprocess.run(ffmpeg_cmd, check=True)
    return output_video_path


def cleanup_sequence(seed_dir, keep_steps=[0, 250, 500, 750, 999]):
    for i in range(1000):
        if i not in keep_steps:
            path = f"{seed_dir}/sample_{i:03d}.png"
            if os.path.exists(path):
                os.remove(path)



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"sample sur le device: {device}")

    model = UNetModel().to(device)

    betas = make_beta_schedule(schedule="cosine", timesteps=1000)
    # Créer le modèle DDPM
    timesteps = 1000
    rescale_timesteps = False
    betas = make_beta_schedule(schedule="cosine", timesteps=timesteps)
    ddpm = GaussianDiffusion(
        betas=betas,
        rescale_timesteps=rescale_timesteps,
    )

    # Chargement des poids
    start = time.time()
    # ddpm.load_state_dict(torch.load("ddpm_model.pth", map_location=device))
    checkpoint = torch.load("checkpoint_improved.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    end = time.time()
    print(f"Temps de chargement du modèle: {end - start:.2f} secondes")

    ### CONFIGURATION ###
    batch_size = 4  # <-- nombre d’images générées
    image_shape  = (batch_size, 3, 64, 64)

    print(f"Génération d’un batch de {batch_size} images...")
    start = time.time()
    # Génération d'images avec la méthode `p_sample_loop`
    progressive_samples = ddpm.sample_loop_progressive(
        model=model,
        shape=image_shape,
        noise=None,  # Bruit initial (None pour générer un bruit aléatoire)
        clip_denoised=True,  # Clip les valeurs générées entre [-1, 1]
        progress=True,  # Affiche une barre de progression
        device="cuda"  
    )

    all_timesteps = []
    for step in progressive_samples:
        all_timesteps.append(step["sample"].cpu())  # Sauvegarder chaque étape

    all_timesteps = torch.stack(all_timesteps)  # (1000, batch_size, 3, 64, 64)

    end = time.time()
    print(f"Temps de génération: {end - start:.2f} secondes")

    print("Traitement de chaque image du batch...")
    for img_idx in range(batch_size):
        seed = torch.randint(0, 1000000, (1,)).item()
        seed_dir = f"./scripts/generations/{seed}"

        # Extraire la séquence de cette image : (1000, 3, 64, 64)
        sample_sequence = all_timesteps[:, img_idx, :, :, :]

        save_sample_sequence(sample_sequence, seed_dir)
        create_video_from_sequence(seed_dir)
        cleanup_sequence(seed_dir)

        print(f"[✓] Image {img_idx+1}/{batch_size} terminée dans {seed_dir}")