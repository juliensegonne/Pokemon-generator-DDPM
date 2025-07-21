import torch
import torch.nn as nn
from .scheduler import make_beta_schedule


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, betas: torch.Tensor, device: str = "cuda"):
        """
        DDPM : Denoising Diffusion Probabilistic Model
        inputs : 
        - model : UNet (ou autre modèle de diffusion)
        - betas : beta schedule (torch.Tensor) (1D tensor de taille num_timesteps)
        """
        super().__init__()
        self.model = model  # UNet
        self.device = device

        # 1. Variables utiles à partir du scheduler de beta
        self.num_timesteps = len(betas)
        self.betas = betas.to(device)
        self.alphas = 1. - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar).to(device)
        self.sqrt_1m_alpha_bar = torch.sqrt(1. - self.alpha_bar).to(device)


    # 2. Noising instantané
    def forward_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """

        input : 
        - x_start : image de départ (batch_size, channels, height, width)
        - t : étape de diffusion (batch_size,)
        - noise : bruit (batch_size, channels, height, width) (même dim que x_start)

        - output : x_t : image bruitée (batch_size, channels, height, width)
        
        Ajoute du bruit à x_start à l'étape t selon la formule :
        x_t = sqrt(alpha_bar_t).x_start + sqrt(1 - alpha_bar_t) * noise
        """
        
        if noise is None:
            noise = torch.randn_like(x_start)


        # self.alpha_bar_sqrt[t] : comme t est un tensor 1D et self.alpha_bar_sqrt est un tensor 1D, self.alpha_bar_sqrt[t] est un tensor 1D de taille batch_size, on doit le reshaper pour qu'il ait la même taille que x_start : pour cela on utilise view(-1, 1, 1, 1) : le -1 signifie que la dimension batch_size est conservée, et les 3 autres dimensions sont mises à 1 
        
        x_t = (
            self.alpha_bar_sqrt[t].view(-1, 1, 1, 1) * x_start +
            self.sqrt_1m_alpha_bar[t].view(-1, 1, 1, 1) * noise
        )
        return x_t


    # 3. Noising successif (pour visualisation étape par étape)
    def forward_noise_progressive(self, x_start: torch.Tensor):
        """
        Génère une séquence d'images bruitées étape par étape
        
        input:
        - x_start: image de départ (batch_size, channels, height, width)
        
        output:%
        - x_seq: séquence d'images bruitées (num_timesteps, batch_size, channels, height, width)
        """
        batch_size = x_start.shape[0]
        x_seq = [x_start]
        x_t = x_start
        
        for t in range(self.num_timesteps):
            # Calcul du facteur de bruit pour cette étape spécifique
            alpha_t = self.alphas[t].to(self.device)
            beta_t = self.betas[t].to(self.device)
            
            # Génération de nouveau bruit pour cette étape
            noise = torch.randn_like(x_start).to(self.device)
            
            # Application du bruit à x_t pour obtenir x_{t+1}
            # x_{t+1} = sqrt(alpha_t) * x_t + sqrt(1 - alpha_t) * noise
            x_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1) * x_t + torch.sqrt(beta_t).view(-1, 1, 1, 1) * noise
            x_seq.append(x_t)
        
        # Convertir la liste en tensor
        return torch.stack(x_seq, dim=0)

        

    # 4. Calcul de la loss (training)
    def loss_fn(self, x_start: torch.Tensor):
        """Implémente l'algo 1
            
            x0 ~ q(x0)
            t ~ U(1, T)
            \eps ~ N(0, I)
            Loss = || \eps - \eps_theta(x_t, t)||^2 with x_t = q(x_t | x0)

            input : 
            - x_start : image de départ (batch_size, channels, height, width)
            
        """
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        noise = torch.randn_like(x_start).to(self.device)

        # forward noise
        x_t = self.forward_noise(x_start, t, noise)

        # prdiction de noise : les tenseurs x_t et t sont bien sur le meme device (GPU)
        noise_pred = self.model(x_t, t)

        # Loss : MSE entre les noises
        loss = nn.MSELoss()(noise_pred, noise)
        return loss
    

    # 5. Training
    def train_batch(self, batch: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
            Effectue une étape d'entraînement sur un batch d'images
            
            inputs:
            - batch: batch d'images (batch_size, channels, height, width)
            - optimizer: optimiseur pour mettre à jour les poids du modèle
            
            returns:
            - loss: valeur de la perte pour ce batch
        """
        # S'assurer que le batch est sur le bon device
        batch = batch.to(self.device)
        
        optimizer.zero_grad()
        loss = self.loss_fn(batch)
        
        # Backprop
        loss.backward()
        
        # Mise à jour des poids
        optimizer.step()
        
        return loss.item()


    # 6. Sampling (inférence)
    @torch.no_grad()
    def sample(self, shape: torch.Size, save_intermediate: bool = False):
        """Implémente l'algo 2
        
            x_T ~ N(0, I)
            for t = T, ..., 1:
                z ~ N(0, I) if t > 1 else 0
                x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * \eps_theta(x_t, t)) + z * sqrt(beta_t)
            endfor
            return x_0 
        
            inputs:
            - shape : (batch_size, channels, height, width)
            output : (time_steps, batch_size, channels, height, width) (si save_intermediate = True) ou (1, batch_size, channels, height, width) 
        """

        # Bruit blanc initial
        x_t = torch.randn(shape).to(self.device)
        res = []

        if save_intermediate:
            res.append(x_t.cpu().detach().clone())
            
        for t in range(self.num_timesteps - 1, -1, -1):
            # Prédiction du bruit pour chaque bruit du batch
            t_tensor = torch.full((shape[0],), t, device=self.device).long() # (batch_size,)

            # les tensors x_t et t_tensor sont bien sur le meme device (GPU)
            noise_pred = self.model(x_t, t_tensor)

            # Calcule de x_{t-1}
            alpha_t = self.alphas[t].to(self.device)
            alpha_bar_t = self.alpha_bar[t].to(self.device)
            beta_t = self.betas[t].to(self.device)
            alpha_bar_prev = self.alpha_bar[t-1] if t > 0 else torch.tensor(1.0).to(self.device)

            # Coefficient pour la prédiction
            c1 = torch.sqrt(1.0 / alpha_t)
            c2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            
            # Prédiction de la moyenne
            pred_mean = c1 * (x_t - c2 * noise_pred)

            if t > 0:
                noise = torch.randn_like(x_t).to(self.device)
                sigma_t = torch.sqrt(self.betas[t] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                x_t = pred_mean + sigma_t * noise
            else:
                x_t = pred_mean


            if save_intermediate:
                res.append(x_t.cpu().numpy())
        
        if not save_intermediate:
            res.append(x_t.cpu().numpy())

        return res
    
