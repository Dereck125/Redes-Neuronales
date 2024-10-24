import torch
from torch.utils.data import Dataset
import numpy as np

class drawdataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        """
        Args:
            x_data (numpy array): Arreglos con imágenes (aplanadas).
            y_data (numpy array): Etiquetas correspondientes.
            transform (callable, optional): Transformaciones que se aplicarán a las imágenes.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Obtener la imagen (aplanada) y redimensionarla
        image = self.x_data[idx].reshape(28, 28).astype(np.float32)  #imágenes son de 28x28 píxeles

        # Si hay transformaciones, aplicarlas
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).unsqueeze(0)

        # Obtener la etiqueta correspondiente
        label = torch.tensor(self.y_data[idx], dtype=torch.long)

        return image, label