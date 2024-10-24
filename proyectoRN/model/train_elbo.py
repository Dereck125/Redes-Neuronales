import os
import torch
import torch.nn as nn
import numpy as np
from model.loss_functions import elbo_loss_function
from model.metrics import save_generated_images, calculate_ssim, calculate_psnr
import tempfile
from pytorch_fid import fid_score
import random
import csv
import matplotlib.pyplot as plt

# Función para inicializar los pesos
def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Función para cargar un checkpoint
def load_checkpoint(model, optimizer, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        print(f"Cargando checkpoint desde {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint cargado correctamente: Continuando desde la época {start_epoch}")
        return start_epoch
    else:
        raise FileNotFoundError(f"El archivo {checkpoint_path} no existe.")

# Función de entrenamiento con cálculo de FID, SSIM y PSNR
def train(model, optimizer, epochs, device, train_loader, val_loader, use_weight_init=True, checkpoint_path=None, scheduler=None):
    # Inicializar o cargar un checkpoint
    start_epoch = 0
    if use_weight_init:
        print("Inicializando pesos con weight_init.")
        model.apply(weight_init)  # Inicializar los pesos
    else:
        if checkpoint_path:
            start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
    
    epoch_losses = []  # Lista para almacenar la pérdida promedio por época
    val_losses = []  # Lista para almacenar la pérdida de validación por época
    #fid_scores = []  # Lista para almacenar los puntajes FID por época
    ssim_values = []  # Lista para almacenar los puntajes SSIM por época
    psnr_values = []  # Lista para almacenar los puntajes PSNR por época
    
    # Definir el archivo CSV donde se guardarán las métricas
    with open('training_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escribir los encabezados
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'SSIM', 'PSNR'])
        
        for epoch in range(start_epoch, epochs):  # Empezar desde la última época guardada
            model.train()  # Cambiar el modelo a modo de entrenamiento
            overall_loss = 0
    
            # Entrenamiento en el conjunto de entrenamiento
            for idx, (x, _) in enumerate(train_loader):
                x = x.to(device)  # Mover los datos al dispositivo
    
                optimizer.zero_grad()  # Reiniciar los gradientes
    
                # Forward pass
                x_hat, mean, log_var, _ = model(x)
    
                # Calcular la pérdida
                loss = elbo_loss_function(x_hat, x, mean, log_var)
                overall_loss += loss.item()  # Acumular la pérdida
    
                # Backward pass y optimización
                loss.backward()
                optimizer.step()
    
            # Pérdida promedio en el conjunto de entrenamiento
            epoch_loss = overall_loss / len(train_loader.dataset)
            epoch_losses.append(epoch_loss)
    
            # Validación en el conjunto de validación
            model.eval()  # Cambiar el modelo a modo de evaluación
            val_loss = 0
            ssim_epoch = 0  # Acumular SSIM
            psnr_epoch = 0  # Acumular PSNR
            n_val_batches = 0  # Para calcular el promedio
    
            with torch.no_grad():
                for x_val, _ in val_loader:
                    x_val = x_val.to(device)
                    x_hat_val, mean_val, log_var_val, _ = model(x_val)
                    loss_val = elbo_loss_function(x_hat_val, x_val, mean_val, log_var_val)
                    val_loss += loss_val.item()
    
                    # Convertir a numpy para SSIM y PSNR
                    x_val_np = x_val.cpu().numpy().squeeze() * 255
                    x_hat_val_np = x_hat_val.cpu().numpy().squeeze() * 255
    
                    # Calcular SSIM y PSNR para este batch
                    ssim_batch = calculate_ssim(x_val_np, x_hat_val_np)
                    psnr_batch = calculate_psnr(x_val_np, x_hat_val_np)
    
                    # Acumular los valores para promediarlos
                    ssim_epoch += ssim_batch * x_val.size(0)  # Multiplicar por el tamaño del batch
                    psnr_epoch += psnr_batch * x_val.size(0)  # Multiplicar por el tamaño del batch
                    n_val_batches += x_val.size(0)  # Acumular el número total de imágenes
    
            # Pérdida promedio en el conjunto de validación
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            # Actualizar el LR con el scheduler cíclico
            if scheduler is not None:
                scheduler.step(val_loss)
    
            # Promedio de SSIM y PSNR en el conjunto de validación
            ssim_epoch /= n_val_batches
            psnr_epoch /= n_val_batches
            ssim_values.append(ssim_epoch)
            psnr_values.append(psnr_epoch)
            
            print(f"Epoch {epoch + 1}, SSIM: {ssim_epoch}, PSNR: {psnr_epoch}")
    
            # # Calcular FID después de cada epoch
            # with tempfile.TemporaryDirectory() as real_image_dir, tempfile.TemporaryDirectory() as generated_image_dir:
            #     # Guardar imágenes reales y generadas en directorios temporales
            #     save_generated_images(x_val, real_image_dir, prefix="real")
            #     save_generated_images(x_hat_val, generated_image_dir, prefix="generated")
    
            #     # Calcular FID para todo el conjunto de validación
            #     fid_value = fid_score.calculate_fid_given_paths([real_image_dir, generated_image_dir], batch_size=64, device=device, dims=2048)
            #     fid_scores.append(fid_value)
    
            #     print(f"FID: {fid_value}")
    
            # Guardar modelo y checkpoints cada 25 épocas
            if (epoch + 1) % 25 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                model_filename = f"VAE_checkpoint_epoch_{epoch + 1}.pth"
                torch.save(checkpoint, model_filename)
                print(f"Checkpoint guardado como {model_filename}")

                # Guardar y mostrar las imágenes generadas
                inds = np.random.randint(0, len(x_val_np)) 
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(x_val_np[inds], cmap='gray')  # Imagen original en escala de grises
                axes[0].set_title("Original Image")
                axes[1].imshow(x_hat_val_np[inds], cmap='gray')  # Imagen reconstruida en escala de grises
                axes[1].set_title(f"Reconstructed Image (Epoch {epoch + 1})")
                plt.savefig(f"reconstructed_image_epoch_{epoch + 1}.png")
                plt.close(fig)  # Cerrar la figura para liberar memoria

            # Guardar métricas
            writer.writerow([epoch + 1, epoch_loss, val_loss, ssim_epoch, psnr_epoch])
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss}, Validation Loss: {val_loss}")
            
    return epoch_losses, val_losses, ssim_values, psnr_values
