import torch
import torch.nn as nn
from model.models import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self, device,out_features,latent_dim):
        super(VAE, self).__init__()
        self.device = device
        self.mean = nn.Linear(in_features=out_features, out_features=latent_dim).to(device)
        self.var = nn.Linear(in_features=out_features, out_features=latent_dim).to(device)
        # encoder
        self.encoder = Encoder(otput_mlp =out_features).to(device)  # Aseguramos que el encoder esté en el device
        # decoder
        self.decoder = Decoder(latent_dim=latent_dim,out_features= out_features).to(device)  # Aseguramos que el decoder esté en el device

    def encode(self, x):
        #print(f"ENCODER input: {x.shape}, Device: {x.device}")
        x = self.encoder(x)
        mean = self.mean(x)
        log_var = self.var(x)
        #print(f"ENCODER output (mean, log_var): {mean.shape}, {log_var.shape}, Device: {mean.device}, {log_var.device}")
        return mean, log_var

    def reparameterization(self, mean, log_var):
        #print(f"REPARAMETRIZACIÓN (mean, log_var): {mean.shape}, {log_var.shape}, Device: {mean.device}, {log_var.device}")
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std).to(self.device)  # Verifica que epsilon esté en el dispositivo correcto
        #print(f"Tamaño del tensor epsilon: {epsilon.shape}, Device: {epsilon.device}")
        #print(f"Tamaño del tensor std: {std.shape}, Device: {std.device}")
        z = std * epsilon + mean
        #print(f"Tamaño del tensor z: {z.shape}, Device: {z.device}")
        return z

    def decode(self, x):
        #print(f"DECODER input: {x.shape}, Device: {x.device}")
        x_hat = self.decoder(x)
        #print(f"DECODER output: {x_hat.shape}, Device: {x_hat.device}")
        return x_hat

    def forward(self, x):
        #print(f"Input tensor shape: {x.shape}, Device: {x.device}")
        mean, log_var = self.encode(x)
        #print(f"Mean shape: {mean.shape}, Log var shape: {log_var.shape}, Device: {mean.device}, {log_var.device}")
        z = self.reparameterization(mean, log_var)
        #print(f"Latent variable z shape: {z.shape}, Device: {z.device}")
        x_hat = self.decode(z)
        #print(f"Output x_hat shape: {x_hat.shape}, Device: {x_hat.device}")
        return x_hat, mean, log_var , z
