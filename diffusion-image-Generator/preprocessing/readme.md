# Partie 1 : Préprocessing & Dataset Handling
*Responsable : Ryan Kaddour*

## TODO :
- **Importation et chargement du dataset** (images de Pokémon).
- Normalisation des images (résolution, canaux, format).
- Data augmentation : symmétries, rotations, rogner, ajout de bruit, etc...
- Vérifications : cohérence, images corrompues, distribution.
- Configuration des variables globales (dimensions des données, chemin dataset, batch_size, etc...)

- **Fournir un DataLoader PyTorch**

## Output attendu :
- Une classe `DatasetLoader` avec :
    - get_dataloader(train=True) (ou équivalent)
    - get_image_shape() (renvoie les dimensions des données prête a être utilisés dans le dataloader)

- Un script de test unitaire pour valider le chargement & les augmentations.
- Un fichier config.py avec les hyperparamètres globaux (dimensions, device, batch_size, etc.)
