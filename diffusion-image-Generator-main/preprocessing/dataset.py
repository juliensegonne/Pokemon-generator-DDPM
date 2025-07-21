# Importer le dataset et implémenter le préprocessing et la classe pour accéder aux données finales (voir /preprocessing/README.md)

import os
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, RandomHorizontalFlip, RandomAffine, ColorJitter
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


class ImageDataset(Dataset):
    """Dataset personnalisé pour les images RGB"""
    
    def __init__(self, image_paths, transform=None):
        """
        inputs :
            image_paths (list): Liste des chemins d'accès aux images
            transform (callable, optional): Transformations à appliquer aux images
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Charger l'image et la convertir en RGB (pour assurer 3 canaux)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_path}: {e}")
            # Retourner une image noire en cas d'erreur
            default_img = torch.zeros((3, 64, 64)) if self.transform else Image.new('RGB', (64, 64), color='black')
            return default_img


class DatasetLoader:
    def __init__(self, dataset_path, img_size=64, batch_size=32, train_ratio=0.8, 
                 normalize=True, augmentation=True, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialise le chargeur de dataset
        
        inputs :
            dataset_path (str): Chemin vers le dossier contenant les images
            img_size (int): Taille désirée des images (carré)
            batch_size (int): Taille des batchs pour le DataLoader
            train_ratio (float): Proportion du dataset à utiliser pour l'entraînement
            normalize (bool): Si True, normalise les images dans [-1, 1]
            augmentation (bool): Si True, applique une augmentation du dataset (symmétries, rotations, zoom, etc...)
            device (str): Appareil sur lequel charger les tensors ('cuda' ou 'cpu')
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.augmentation = augmentation
        self.device = device
        
        # Les dataloaders seront initialisés par load_data()
        self.train_loader = None
        self.val_loader = None
        self.all_loader = None
        
        # Les chemins d'images seront stockés ici
        self.image_paths = []
        
        print(f"Initialisation du DatasetLoader avec device: {device}")
    

    def _find_images(self):
        """Trouve récursivement toutes les images dans le dossier dataset_path"""
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_paths = []
        
        # Parcourir récursivement le dossier
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            raise ValueError(f"Aucune image trouvée dans {self.dataset_path}")
            
        return image_paths
    
    def load_data(self):
        print(f"Chargement des images depuis {self.dataset_path}...")
        self.image_paths = self._find_images()
        print(f"Trouvé {len(self.image_paths)} images")


        base_transform = Compose([
            Resize((self.img_size, self.img_size)),
            ToTensor(),
            Normalize(mean=[0.5]*3, std=[0.5]*3) if self.normalize else lambda x: x
        ])
        base_dataset = ImageDataset(self.image_paths, transform=base_transform)


        if self.augmentation:
            aug_transform = Compose([
                Resize((self.img_size, self.img_size)),
                RandomHorizontalFlip(p=0.5),
                RandomAffine(degrees=30, scale=(1.1, 1.5), fill=(0, 0, 0)),  # Zoom léger + rotation
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                ToTensor(),
                Normalize(mean=[0.5]*3, std=[0.5]*3) if self.normalize else lambda x: x
            ])
            aug_dataset = ImageDataset(self.image_paths, transform=aug_transform)

            # Combiner les datasets : originaux + augmentés
            full_dataset = ConcatDataset([base_dataset, aug_dataset])
            print(f"Dataset étendu à {len(full_dataset)} images (avec augmentation)")
        else:
            full_dataset = base_dataset
        
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)

        train_size = int(len(full_dataset) * self.train_ratio)
        val_size = len(full_dataset) - train_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)


        print(f"Split du dataset: {train_size} images d'entraînement, {val_size} images de validation")

        self.train_loader = DataLoader(
        train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=self.device == "cuda")
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.device == "cuda")
        
        self.all_loader = DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=self.device == "cuda")

    
    def get_data(self):
        """Retourne les dataloaders pour l'entraînement et la validation"""
        if self.train_loader is None:
            self.load_data()
        
        return self.train_loader, self.val_loader, self.all_loader
    
    def visualize_batch(self, nrow=8, save_path=None):
        """Visualise un batch d'images pour vérifier le prétraitement"""
        if self.train_loader is None:
            self.load_data()
        
        # Récupérer un batch
        images = next(iter(self.train_loader))
        
        # Dénormaliser si nécessaire
        if self.normalize:
            # Convertir de [-1, 1] à [0, 1]
            images = (images + 1) / 2
        
        # Créer une grille d'images
        grid = make_grid_of_images(images, nrow=nrow)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(grid)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Visualisation sauvegardée dans {save_path}")
            
        plt.show()


def make_grid_of_images(images, nrow=8):
    """Crée une grille d'images à partir d'un batch de tensors"""
    # Convertir les tensors en numpy arrays
    images_np = images.detach().cpu().numpy()
    
    # Réorganiser les dimensions pour avoir [N, H, W, C]
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    # Calculer le nombre de lignes nécessaires
    n_images = images_np.shape[0]
    ncol = min(nrow, n_images)
    nrow = int(np.ceil(n_images / ncol))
    
    # Créer une grille vide
    padding = 2
    h, w = images_np.shape[1], images_np.shape[2]
    grid = np.zeros((h*nrow + padding*(nrow-1), w*ncol + padding*(ncol-1), 3))
    
    # Remplir la grille avec les images
    for idx, img in enumerate(images_np):
        i = idx // ncol
        j = idx % ncol
        grid[i*(h+padding):i*(h+padding)+h, j*(w+padding):j*(w+padding)+w, :] = img
        
    return grid


# Télécharger un dataset Kaggle localement et récupérer le chemin d'accès
def download_kaggle_dataset(dataset_id):
    print(f"Téléchargement du dataset Kaggle: {dataset_id}")
    path = kagglehub.dataset_download(dataset_id)
    print(f"Dataset téléchargé dans: {path}")
    return path

