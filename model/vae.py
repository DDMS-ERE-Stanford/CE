import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 

class Flatten(nn.Module): 
    def forward(self, input): 
        return input.view(input.size(0), -1)

class UnFlatten_case1(nn.Module):
    def forward(self, input, size = 256):
        return input.view(input.size(0), size, 6, 6)

class UnFlatten_case2(nn.Module):
    def forward(self, input, size = 256):
        return input.view(input.size(0), size, 5, 5)

class VAE(nn.Module):
    def __init__(self, vae_case=1, image_channels=1, h_dim = 36*256, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size = 3, stride = 2, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1), 
            nn.ReLU(), 
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(), 
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1), 
            nn.ReLU(),
            Flatten()
        )

        if (vae_case == 1): 
            h_dim = 36*256
        elif (vae_case == 2): 
            h_dim = 25*256

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        if (vae_case == 1): 
             self.decoder = nn.Sequential( 
                UnFlatten_case1(),
                nn.ConvTranspose2d(256, 128, kernel_size = 4, padding = 1, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size = 4, padding = 2, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size = 4, padding = 2, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size = 3, padding = 2, stride = 2),
                nn.Sigmoid(),
            )
        elif (vae_case == 2): 
             self.decoder = nn.Sequential( 
                UnFlatten_case2(),
                nn.ConvTranspose2d(256, 128, kernel_size = 4, padding = 1, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size = 3, padding = 1, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size = 4, padding = 1, stride = 2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size = 3, padding = 1, stride = 2),
                nn.Sigmoid(),
            )

    def reparameterize(self, mu, logvar): 
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z  

    def bottleneck(self, h): 
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

def loss_fn(recon_x, x, mu, logvar): 
    BCE = F.mse_loss(recon_x, x, size_average=False)

    KLD = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE+KLD, BCE, KLD

