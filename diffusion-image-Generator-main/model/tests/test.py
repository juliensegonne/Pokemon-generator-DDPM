# Exécuter depuis : la racine avec
# python -m model.tests.test

import torch
import torchvision.transforms as T
from PIL import Image
import os

#mesurer le temps d'exécution
import time

# import sys
# import numpy as np

# # Ajouter le dossier parent au path pour importer les modules correctement
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.unet import UNet
from model.improved_unet import UNetModel

def test_unet_forward(model_to_test="UNet"):
    # Vérifier si CUDA est disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")
    
    # Charger et transformer l'image
    # img_path = os.path.join(os.path.dirname(__file__), "images/imageTest.png")
    img_path = os.path.join(os.path.dirname(__file__), "images/test_pokemon.png")
    image = Image.open(img_path).convert("RGB")
    
    # Transformation de l'image
    transform = T.Compose([
        T.Resize((64, 64)),  # Redimensionner à une taille fixe
        T.ToTensor(),          # Convertir en tensor [0,1]
        T.Normalize([0.5]*3, [0.5]*3),  # Normaliser à [-1, 1]
    ])
    
    x_original = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 256, 256)
    batch_size, channels, height, width = x_original.shape
    print(f"Dimensions de l'image originale: {x_original.shape}")
    
    # Bruiter l'image (simulation d'un processus de diffusion)
    # On génère un bruit gaussien et on le mélange avec l'image
    noise = torch.randn_like(x_original)
    noise_level = 0.5  # Niveau de bruit entre 0 et 1
    x_noisy = (1 - noise_level) * x_original + noise_level * noise
    
    # Création du modèle UNet
    n_channels = 64  # Nombre de canaux de base
    ch_mults = (1, 2, 2, 4)  # Multiplicateurs de canaux par niveau
    is_attn = (False, False, True, True)  # Attention par niveau
    n_blocks = 2  # Nombre de blocs résiduels par niveau
    
    if model_to_test == "UNet":
        model = UNet(
            image_channels=channels,
            n_channels=n_channels,
            ch_mults=ch_mults,
            is_attn=is_attn,
            n_blocks=n_blocks
        ).to(device)
    else:
        model = UNetModel().to(device)
    
    # Afficher le nombre de paramètres du modèle
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres du modèle: {n_params:,}")
    
    # Créer un tenseur de timestep (simulant une étape de diffusion)
    # Dans un vrai processus de diffusion, ce serait l'étape temporelle t
    t = torch.tensor([500], device=device)  # Une seule étape temporelle pour toutes les images du batch
    
    # Passer l'image bruitée dans le modèle
    print("Exécution du forward pass...")
    try:
        with torch.no_grad():  # Pas besoin de gradient pour le test
            
            start_time = time.time()  # Démarrer le 
            output = model(x_noisy, t)
            end_time = time.time()
            print(f"Temps d'exécution du forward pass: {end_time - start_time:.4f} secondes")
        
        # Vérifier les dimensions de sortie
        print(f"Dimensions de la sortie: {output.shape}")
        assert output.shape == x_original.shape, f"La sortie doit avoir la même forme que l'entrée: {output.shape} vs {x_original.shape}"
        
        # Vérifier les valeurs de sortie (approximativement dans la plage [-1, 1])
        min_val = output.min().item()
        max_val = output.max().item()
        print(f"Plage de valeurs de sortie: [{min_val:.4f}, {max_val:.4f}]")
        
        # Calculer la perte (MSE entre la sortie et l'image originale)
        # Dans un vrai modèle de diffusion, la perte serait différente
        mse_loss = torch.nn.functional.mse_loss(output, x_original).item()
        print(f"MSE entre sortie et image originale: {mse_loss:.6f}")
        
        print("✅ Test du forward pass du UNet réussi!")
        
        # Optionnel: Sauvegarder une visualisation
        save_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(save_dir, exist_ok=True)
        
        # Fonction pour convertir un tensor en image PIL
        def tensor_to_pil(tensor):
            # Dénormaliser de [-1, 1] à [0, 1]
            tensor = (tensor + 1) / 2
            # Clamp pour s'assurer que les valeurs sont entre 0 et 1
            tensor = torch.clamp(tensor, 0, 1)
            # Convertir en image PIL
            return T.ToPILImage()(tensor)
        
        # Sauvegarder les images originale, bruitée et prédite
        original_pil = tensor_to_pil(x_original.squeeze(0).cpu())
        noisy_pil = tensor_to_pil(x_noisy.squeeze(0).cpu())
        output_pil = tensor_to_pil(output.squeeze(0).cpu())
        
        original_pil.save(os.path.join(save_dir, "original.png"))
        noisy_pil.save(os.path.join(save_dir, "noisy.png"))
        output_pil.save(os.path.join(save_dir, "output.png"))
        print(f"Images sauvegardées dans le dossier {save_dir}")
        
    except Exception as e:
        print(f"❌ Test échoué avec l'erreur: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_unet_forward(model_to_test="UNetImproved")