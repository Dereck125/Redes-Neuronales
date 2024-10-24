import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,otput_mlp):
        super(Encoder, self).__init__()
        self.CNN1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.CNN2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.MLP1 = nn.Linear(in_features=4*7*7, out_features=otput_mlp)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.normalization1 = nn.BatchNorm2d(num_features=8)
        self.normalization2 = nn.BatchNorm2d(num_features=4)
        self.dropout = nn.Dropout(p=0.5)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print(f"Input tensor shape: {x.shape}")  # Imprimir forma de entrada
        x = self.leakyrelu(self.normalization1(self.CNN1(x)))
        #print(f"After CNN1: {x.shape}")  # Imprimir después de CNN1
        x = self.dropout(self.maxpool(x))
        #print(f"After maxpool: {x.shape}")  # Imprimir después del maxpool
        x = self.leakyrelu(self.normalization2(self.CNN2(x)))
        #print(f"After CNN2: {x.shape}")  # Imprimir después de CNN2
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        #print(f"After flattening: {x.shape}")  # Imprimir después de aplanar
        x = self.relu(self.MLP1(x))
        #print(f"After MLP1: {x.shape}")  # Imprimir después de la capa MLP1
        return x


class Decoder(nn.Module):
    def __init__(self,latent_dim,out_features):
        super(Decoder, self).__init__()
        self.MLP1 = nn.Linear(in_features=latent_dim, out_features=out_features)
        self.MLP2 = nn.Linear(in_features=out_features, out_features=128)
        self.MLP3 = nn.Linear(in_features=128, out_features=16*7*7) # Transformar la salida a 16 canales, 7x7

        self.TCNN1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1) #14X14X8
        self.TCNN2 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=2, stride=2, padding=0) #28X28X1

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(f"Tamaño del tensor antes de la primera capa fully connected: {x.shape}")
        x = self.relu(self.MLP1(x))
        #print(f"Tamaño del tensor después de la primera capa fully connected: {x.shape}")
        x = self.relu(self.MLP2(x))
        #print(f"Tamaño del tensor después de la segunda capa fully connected: {x.shape}")
        x = self.relu(self.MLP3(x))
        #print(f"Tamaño del tensor después de la tercera capa fully connected: {x.shape}")
        x = x.view(x.size(0), 16, 7, 7)
        #print(f"Tamaño del tensor después de la aplanamiento: {x.shape}")
        x = self.relu(self.TCNN1(x))  # Primera capa deconvolucional
        #print(f"Tamaño del tensor después de la primera capa deconvolucional: {x.shape}")
        x = self.sigmoid(self.TCNN2(x))  # Segunda capa deconvolucional, con salida final en [0, 1]
        #print(f"Tamaño del tensor después de la segunda capa deconvolucional: {x.shape}")
        return x
