# Exécuter depuis : la racine avec
# python -m preprocessing.tests.test

from preprocessing.dataset import DatasetLoader #, download_kaggle_dataset
from preprocessing.config import get_config


if __name__ == "__main__":
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, TRAIN_RATIO, NORMALIZE, AUGMENTATION = get_config()

    
    # Créer et configurer le dataloader
    loader = DatasetLoader(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        normalize=NORMALIZE,
        augmentation=AUGMENTATION,
    )
    
    # Charger les données
    loader.load_data()
    
    # Visualiser un batch
    loader.visualize_batch(nrow=4, save_path="./preprocessing/tests/batch_preview.png")
    
    # Obtenir les dataloaders
    train_loader, val_loader, all_loader = loader.get_data()
    
    # Afficher des informations sur les dataloaders
    print(f"Nombre de batchs d'entraînement: {len(train_loader)}")
    print(f"Nombre de batchs de validation: {len(val_loader)}")
    print(f"Taille du batch: {loader.batch_size}")
    print(f"Dimensions des images: {next(iter(train_loader)).shape}")
