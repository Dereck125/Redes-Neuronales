import torch
import torch.nn.functional as F
import torch.nn as nn
#ELBO
def elbo_loss_function(x_hat, x, mean, log_var):
    # Pérdida de reconstrucción (Binary Cross Entropy o MSE)
    #Entre la imagen original y la reconstruida
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    # Divergencia KL: KL(q(z|x) || p(z))
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # Pérdida total = ELBO negativa
    return recon_loss + kl_loss




def compute_mmd_loss(z, z_prior, kernel='rbf'):
    """Calcula la Maximum Mean Discrepancy (MMD) entre z (posterior) y z_prior (prior)."""

    # Obtener el tamaño del batch
    n = z.size(0)

    # Expandir las dimensiones de z y z_prior para calcular las distancias de cada par de puntos
    z_expand = z.unsqueeze(1)  # (n, 1, dim)
    z_prior_expand = z_prior.unsqueeze(0)  # (1, n, dim)

    # Calcular la varianza sigma para el kernel RBF
    sigma = torch.mean(torch.cdist(z, z_prior)) ** 2  # Estimación de sigma usando la media de las distancias

    # Kernel RBF (Radial Basis Function)
    K_zz = torch.exp(-torch.cdist(z_expand, z_expand) ** 2 / (2 * sigma))  # Kernel entre z y z
    K_pp = torch.exp(-torch.cdist(z_prior_expand, z_prior_expand) ** 2 / (2 * sigma))  # Kernel entre z_prior y z_prior
    K_zp = torch.exp(-torch.cdist(z_expand, z_prior_expand) ** 2 / (2 * sigma))  # Kernel entre z y z_prior

    # Calcular el MMD^2
    mmd = K_zz.mean() + K_pp.mean() - 2 * K_zp.mean()

    return mmd


#Wasserstein Autoencoder (WAE)
def wae_loss_function(x_hat, x, z, z_prior, lambda_reg=10):
    # Pérdida de reconstrucción
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')

    # Regularización Wasserstein aproximada usando MMD
    mmd_loss = compute_mmd_loss(z, z_prior)

    # Pérdida total
    return recon_loss + lambda_reg * mmd_loss
