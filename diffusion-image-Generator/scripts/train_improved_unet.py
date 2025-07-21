# 1. Importer DatasetLoader, DenoiserUNet, DDPM(DenoiserUNet) 
# 2. Faire la pipeline d'entrainement de DDPM
# 3. Enregistrer le modèle entrainé

# Exécuter ce script depuis la racine du projet avec la commande suivante:
# python -m scripts.train

from preprocessing.dataset import DatasetLoader, download_kaggle_dataset
from model.improved_unet import UNetModel
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.scheduler import make_beta_schedule
import torch


import os
import json
import time






if __name__ == "__main__":
    path = download_kaggle_dataset("brilja/pokemon-mugshots-from-super-mystery-dungeon")
    
    # Créer et configurer le dataloader
    loader = DatasetLoader(
        dataset_path=path,
        img_size=64,
        batch_size=32,
        train_ratio=1,
        normalize=True,
        augmentation=True
    )
    
    # Charger les données
    loader.load_data()
    
    # Obtenir les dataloaders
    train_loader, val_loader, all_loader = loader.get_data()
    
    # Afficher des informations sur les dataloaders
    print(f"Nombre de batchs d'entraînement: {len(train_loader)}")
    print(f"Nombre de batchs de validation: {len(val_loader)}")
    print(f"Taille du batch: {loader.batch_size}")
    print(f"Dimensions des images: {next(iter(train_loader)).shape}")
    print(f"Nombre total d'images: {len(all_loader)}")




    # Créer le modèle UNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")

    channels = 3            #RGB
    n_channels = 64         # Nombre de canaux de base
    ch_mults = (1, 2, 2, 4)  # Multiplicateurs de canaux par niveau
    is_attn = (False, False, True, True)  # Attention par niveau
    n_blocks = 2  # Nombre de blocs résiduels par niveau


    # model = UNet(
    #     image_channels=channels,
    #     n_channels=n_channels,
    #     ch_mults=ch_mults,
    #     is_attn=is_attn,
    #     n_blocks=n_blocks
    # ).to(device)


    model = UNetModel().to(device)



    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres du modèle: {n_params:,}")


    # Créer le modèle DDPM
    timesteps = 1000
    rescale_timesteps = False
    betas = make_beta_schedule(schedule="cosine", timesteps=timesteps)
    ddpm = GaussianDiffusion(
        betas=betas,
        rescale_timesteps=rescale_timesteps,
    )

    print(f"Nombre de timesteps: {len(ddpm.betas)}")


    # Entraîner le modèle
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # charger un checkpoint si disponible
    checkpoint_path = "checkpoint_improved.pt"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Checkpoint chargé. Reprise à l'epoch {start_epoch}")
    else:
        print("Aucun checkpoint trouvé. Entraînement à partir de zéro.")

    # Logger
    log_file = "train_log.json"
    logs = []

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)

    num_epochs = 300
    save_every_n_batches = len(train_loader) // 2

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        loss_epoch = 0.0

        for i, batch in enumerate(train_loader):
            x_start = batch.to(device)
            optimizer.zero_grad()

            # Générer des timesteps aléatoires pour chaque batch
            t = torch.randint(0, ddpm.num_timesteps, (x_start.size(0),), device=device)

            # Calculer la perte avec la méthode `training_losses`
            loss_batch = ddpm.training_losses(model, x_start, t)
            loss_batch = loss_batch.mean()  # Moyenne de la perte sur le batch

            # Backpropagation et mise à jour des poids
            loss_batch.backward()
            optimizer.step()

            loss_epoch += loss_batch.item()

            # Log dans la console
            print(f"\rEpoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss_batch:.4f}", end="")

            # Sauvegarde intermédiaire
            if (i + 1) % save_every_n_batches == 0:
                torch.save({
                    "epoch": epoch,
                    "batch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, checkpoint_path)
                print(f"\nCheckpoint sauvegardé à l'epoch {epoch}, batch {i}")

        avg_loss = loss_epoch / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Sauvegarde du modèle complet
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        # Ajout au log
        logs.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time_sec": epoch_time
        })

        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"\nEpoch {epoch + 1} terminée, perte moyenne: {avg_loss:.4f}, durée: {epoch_time:.2f}s")
