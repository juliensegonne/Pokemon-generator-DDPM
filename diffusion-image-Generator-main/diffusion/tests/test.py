# Exécuter depuis : la racine avec
# python -m diffusion.tests.test

import torch
import torchvision.transforms as T
from PIL import Image
import os

from diffusion.diffusion import DDPM
from diffusion.scheduler import make_beta_schedule

# Test de la fonction forward_noise instantané en générant plusieurs images
def test_instant_forward_noise():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger et transformer l'image
    img_path = "./diffusion/tests/images/imageTest.png"
    image = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        # T.Resize((64, 64)),
        T.Resize((image.size[1] // 4, image.size[0] // 4)),  # Réduire la taille de l'image
        T.ToTensor(),  # convert to [0,1]
        T.Normalize([0.5]*3, [0.5]*3),  # normalize to [-1, 1]
    ])
    x_start = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 64, 64)

    # Créer le scheduler beta et le DDPM (model=None car inutile ici)
    betas = make_beta_schedule(schedule="linear", timesteps=1000)
    ddpm = DDPM(model=None, betas=betas, device=device)

    # Noiser à différents instants t
    t_values = [10, 100, 250, 500, 750, 999]
    for t in t_values:
        t_tensor = torch.tensor([t], device=device)
        noisy = ddpm.forward_noise(x_start, t_tensor)

        # Dénormaliser pour affichage
        unnormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        img_noisy = unnormalize(noisy.squeeze(0).cpu().clamp(-1, 1))
        img_pil = T.ToPILImage()(img_noisy)

        save_path = f"./diffusion/tests/images/instant_noising/imageTest_t{t}.png"
        img_pil.save(save_path)

    print("✅ Test de noising instantané terminé.")

# Ajouter cette fonction à test.py
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from tqdm import tqdm


# Test de la fonction forward_noise_progressive pour visualiser le bruitage progressif
def test_progressive_forward_noise():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")

    # Charger et transformer l'image
    # img_path = "./diffusion/tests/images/imageTest.png"
    img_path = "./diffusion/tests/images/test_pokemon.png"
    image = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        # T.Resize((image.size[1] // 4, image.size[0] // 4)),  # Réduire la taille
        T.ToTensor(),  # convert to [0,1]
        T.Normalize([0.5]*3, [0.5]*3),  # normalize to [-1, 1]
    ])
    x_start = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, H, W)

    # Créer le scheduler beta et le DDPM (model=None car inutile ici)
    timesteps = 1000  # Réduire pour une génération plus rapide
    schedule_type = "cosine"  
    betas = make_beta_schedule(schedule=schedule_type, timesteps=timesteps)
    ddpm = DDPM(model=None, betas=betas, device=device)

    # Générer la séquence complète d'images bruitées
    print("Génération de la séquence d'images bruitées...")
    noisy_seq = ddpm.forward_noise_progressive(x_start)
    
    # Fonction pour dénormaliser les images
    unnormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    
    # Créer la figure pour l'animation
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 1, 1], height_ratios=[1, 1])
    
    # Sous-plot pour l'image
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.set_title("Image avec bruit progressif")
    ax_img.axis('off')
    
    # Sous-plots pour les histogrammes des canaux
    ax_r = fig.add_subplot(gs[0, 1])
    ax_g = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[0, 3])
    ax_r.set_title("Canal Rouge")
    ax_g.set_title("Canal Vert")
    ax_b.set_title("Canal Bleu")
    
    # Sous-plot pour l'histogramme combiné
    ax_combined = fig.add_subplot(gs[1, 1:])
    ax_combined.set_title("Distribution combinée")
    
    # Fonction pour initialiser les plots
    def init():
        ax_img.clear()
        ax_r.clear()
        ax_g.clear()
        ax_b.clear()
        ax_combined.clear()
        return []
    
    # Fonction pour mettre à jour l'animation à chaque frame
    def update(frame):
        # Effacer les axes
        ax_img.clear()
        ax_r.clear()
        ax_g.clear()
        ax_b.clear()
        ax_combined.clear()
        
        # Récupérer l'image bruitée à l'étape frame
        noisy_img = noisy_seq[frame].squeeze(0).cpu()
        
        # Dénormaliser pour l'affichage
        img_display = unnormalize(noisy_img.clamp(-1, 1))
        img_np = np.clip(img_display.permute(1, 2, 0).numpy(), 0, 1)
        
        # Afficher l'image
        ax_img.imshow(img_np)
        ax_img.set_title(f"Image bruitée (t={frame}/{timesteps})")
        ax_img.axis('off')
        
        # Extraire les valeurs par canal (toujours normalisées entre -1 et 1)
        r_vals = noisy_img[0].flatten().numpy()
        g_vals = noisy_img[1].flatten().numpy()
        b_vals = noisy_img[2].flatten().numpy()
        
        # Histogrammes par canal
        ax_r.hist(r_vals, bins=200, color='red', alpha=0.7, density=True)
        ax_g.hist(g_vals, bins=200, color='green', alpha=0.7, density=True)
        ax_b.hist(b_vals, bins=200, color='blue', alpha=0.7, density=True)
        
        # Histogramme combiné avec courbe gaussienne de référence
        all_vals = np.concatenate([r_vals, g_vals, b_vals])
        ax_combined.hist(all_vals, bins=200, alpha=0.6, color='purple', label='Distribution réelle', density=True)
        
        # Ajouter une courbe gaussienne de référence N(0,1)
        if frame > 0:  # Seulement après la première frame
            x = np.linspace(-3, 3, 100)
            y = timesteps / frame * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)  # Ajuster l'amplitude
            ax_combined.plot(x, y, 'k--', linewidth=2, label='Gaussienne N(0,1)')
        
        # Configuration des axes
        for ax in [ax_r, ax_g, ax_b, ax_combined]:
            ax.set_xlim(-2, 2)
            ax.grid(True, alpha=0.3)
        
        # Légende pour l'histogramme combiné
        ax_combined.legend()
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return []
    
    # Créer l'animation
    print("Création de l'animation...")
    frames = list(range(0, len(noisy_seq)))  # échantillonnage des frames
    if frames[-1] != len(noisy_seq) - 1:
        frames.append(len(noisy_seq) - 1)  # s'assurer d'inclure la dernière frame
    
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    
    # Sauvegarder l'animation
    image_name = os.path.basename(img_path).split('.')[0]
    save_path = f"./diffusion/tests/images/progressive_noising/animation_{image_name}_{schedule_type}.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    frames = list(range(0, len(noisy_seq), 10))  # échantillonnage des frames
    if frames[-1] != len(noisy_seq) - 1:
        frames.append(len(noisy_seq) - 1)  # s'assurer d'inclure la dernière frame

    # Créer un writer pour ffmpeg
    writer = FFMpegWriter(fps=18, metadata=dict(artist='Me'), bitrate=1800)

    # Sauvegarder les frames avec une barre de progression
    with writer.saving(fig, save_path, dpi=50):
        for frame in tqdm(frames, desc="Sauvegarde des frames"):
            update(frame)  # Mettre à jour la frame
            writer.grab_frame()  # Sauvegarder la frame actuelle

    print("✅ Animation sauvegardée avec succès.")



# Afficher la courbe d'évolution des différents schedules sur un même graphique
def plot_beta_schedules():
    timesteps = 1000
    schedules = ["linear", "cosine", "quadratic"]
    betas_list = [make_beta_schedule(schedule=schedule, timesteps=timesteps) for schedule in schedules]

    print(len(betas_list[0]), len(betas_list[1]), len(betas_list[2]))

    plt.figure(figsize=(10, 6))
    for betas, schedule in zip(betas_list, schedules):
        plt.plot(betas.numpy(), label=schedule)

    plt.title("Beta Schedules")
    plt.xlabel("Timesteps")
    plt.ylabel("Beta Value")
    plt.legend()
    plt.grid()
    plt.show()

def plot_alpha_cumprod_schedules():
    timesteps = 1000
    schedules = ["linear", "cosine", "quadratic"]
    betas_list = [make_beta_schedule(schedule=schedule, timesteps=timesteps) for schedule in schedules]

    alphas_cumprod_list = [torch.cumprod(1 - betas, dim=0) for betas in betas_list]

    print(len(alphas_cumprod_list[0]), len(alphas_cumprod_list[1]), len(alphas_cumprod_list[2]))
    plt.figure(figsize=(10, 6))
    for alphas_cumprod, schedule in zip(alphas_cumprod_list, schedules):
        plt.plot(alphas_cumprod.numpy(), label=schedule)
    
    plt.title("Alpha Cumulative Product Schedules (quantity of signal)")
    plt.xlabel("Timesteps")
    plt.ylabel("Alpha Cumulative Product Value")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # test_instant_forward_noise()
    # test_progressive_forward_noise()
    # plot_beta_schedules()
    plot_alpha_cumprod_schedules()