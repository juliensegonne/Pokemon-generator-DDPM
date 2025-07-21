# Partie 2 : Modèle de prédiction
*Responsable : Julien Segonne*

## TODO :
- Conception du réseau de neurones (UNET modifié dans le papier DDPM)
- target : prédire le bruit ε_θ(x_t, t) à partir d’une image bruitée et du temps t.
- Bonne chance 😨

## Output attendu :
- Une classe `DenoiserUNet(nn.Module)` avec
    - `forward(x_t, t)` : prédit le bruit à partir de l’image bruitée à l’instant t.
    - `loss_fn(model, x_0) ` retourne la loss (par ex MSE) entre le bruit estimé et vrai bruit.
- Test unitaires pour valider le bon fonctionnement du modèle.

