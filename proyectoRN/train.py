import glob
import os
import torch
import tempfile
from model.models import Encoder
from model.models import Decoder
from model.vae import VAE
from model.datasetconfig import drawdataset
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import torch.optim as optim
from model.loss_functions import elbo_loss_function
from model.metrics import save_generated_images,calculate_ssim,calculate_psnr
from model.train_elbo import train
import csv


# Ruta de los archivos .npy
all_files = glob.glob(os.path.join("/home/dereck125/Documentos/GitHub/Redes-Neuronales/proyectoRN/data/", '*.npy'))

# Inicializar variables
classes = ['cat', 'dog', 'apple', 'tree', 'bird']# 'airplane', 'key', 'bicycle', 'banana', 'fish', 'eye', 'car']
data_list = []
label_list = []
class_names = []

# Definir el límite de matrices por archivo
limit_per_file = 10000

for file in all_files:
    # Cargar el archivo en modo de mapeo a memoria
    data = np.load(file, mmap_mode='r')
    #print(f"Procesando archivo: {file}")
    
    # Limitar la cantidad de matrices por archivo a 5000
    data = data[:limit_per_file, :]
    
    # Extraer el nombre de la clase desde el archivo
    class_name = os.path.splitext(os.path.basename(file))[0]
    print(f"Clase: {class_name}")
    
    # Verificar si la clase está en la lista
    if class_name in classes:
        class_idx = classes.index(class_name)
        
        # Crear las etiquetas para este archivo
        labels = np.full(data.shape[0], class_idx)
        print(f"Índice de la clase: {class_idx}")
        
        # Almacenar los datos y etiquetas en las listas
        data_list.append(data)
        label_list.append(labels)
        
        # Almacenar el nombre de la clase
        class_names.append(class_name)
    else:
        print(f"Clase {class_name} no encontrada en la lista de clases.")

# Convertir las listas en arreglos NumPy al final
x = np.vstack(data_list)
y = np.concatenate(label_list)

# Verificar si todo está bien asignado
# print(f"Clases procesadas: {class_names}")
# print(f"Tamaño de los datos: {x.shape}")
# print(f"Tamaño de las etiquetas: {y.shape}")

# Para depurar
encoder = Encoder(otput_mlp = 100)
input_tensor = torch.randn(32, 1, 28, 28)  # Batch de ejemplo con imágenes de 28x28
output1 = encoder(input_tensor)
print(output1.shape )
print("Encoder trabaja correctamente")
decode = Decoder(latent_dim=30,out_features= 100)
z = torch.randn(32,30)  # Batch de ejemplo con imágenes de 28x28
output2 = decode(z)
print(output2.shape)
print("Decoder trabaja correctamente")

transform = transforms.Compose([
    transforms.ToTensor(),
     transforms.Lambda(lambda x: x / 255.0)  # Normaliza los valores al rango [0, 1]
])

dataset = drawdataset(x_data=x,y_data=y,transform=transform)

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Crear DataLoader para cada conjunto
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=4, pin_memory=True)


print("TORCH CUDA DISPONIBLE : {}".format(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"ENTRENANDO CON {device}")
print("Número de clases = {}".format(len(classes)))

#Definimos el modelo VAE
model = VAE(device=device,out_features=60,latent_dim=30).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3, min_lr=1e-5, verbose=True)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=2000, mode='triangular')

# Definir el path del checkpoint
checkpoint_path = "VAE_checkpoint_epoch_3125.pth" # Ruta del archivo .pth 

# Llamar a la función train con el checkpoint
epoch_losses, val_losses, fid_scores, ssim_values, psnr_values = train(
    model,
    optimizer,
    epochs=5000,  # Definir el número de épocas
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    use_weight_init=False,  # No inicializar pesos, ya que vamos a cargar un checkpoint
    checkpoint_path=checkpoint_path,  # Pasar el checkpoint path para cargar los pesos
    scheduler=scheduler
)


