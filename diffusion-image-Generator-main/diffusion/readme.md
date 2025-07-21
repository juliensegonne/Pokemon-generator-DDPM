# Partie 3 : Processus de diffusion (forward & reverse)
*Responsable : Alexandre Dréan*

## TODO :
- Gestion des scheduler de bruit
- Implémentation du forward process
- Implémentation du reverse process (sampling à partir du bruit).
- **Fournir une classe DDPM qui prendra en parametre le model utilisé (unet) et aura des méthode train; sample;**

## Output attendu :
- Une classe `DDPM(nn.Module)` avec
    - `forward_noise(x_0, t)` : applique le bruit à une image propre
    - `denoising_step(x_t, t, model)` : applique le modèle pour générer l'image précédente (un peu moins bruité)
    - `sample(model)` : pipeline complète de génération d'une image
    - `loss_fn(model, x_0) ` retourne la loss (par ex MSE) entre le bruit estimé et vrai bruit.
- Test unitaires pour valider le forward et reverse process
