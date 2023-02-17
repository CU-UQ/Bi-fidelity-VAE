"""
@author: NUOJIN
"""

from typing import List, Any, Tuple
from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import tqdm
import numpy as np

Tensor = torch.Tensor


class BFDataset(torch.utils.data.Dataset):
    
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError
        
class VAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: float = 1.0,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        input_dim = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],hidden_dims[i + 1]),
                    nn.ReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)

    def encode(self, x: Tensor) -> List[Tensor]:
        result = self.encoder(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = F.relu(result)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), x, mu, log_var]
    
    def loss_function(self,
                      data: Tensor
                      ) -> dict:
        mu,log_var = self.encode(data)
        
        data_hat = self.forward(data)[0]
        recons_loss = ((data - data_hat)**2).sum()

        kld_loss = torch.sum(-0.5 * (1 + log_var - mu ** 2 - torch.exp(2 * log_var)))

        loss = recons_loss + self.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
    
    def train(self, 
              trn_data: Tensor,
              batch_size: int = 64,
              epochs: int = 500):
        trn_dataloader = DataLoader(trn_data, batch_size=batch_size)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas= (0.9, 0.99))
        losses = []
        with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
            epoch_loss = 0
            for epoch in tepoch:
                for batch_index, data in enumerate(trn_dataloader):
                    opt.zero_grad()
                    trn_loss = self.loss_function(data)['loss']
                    trn_loss.backward()
                    epoch_loss += trn_loss.item()
                    opt.step()
    
                    epoch_loss += trn_loss
                epoch_loss /= len(trn_dataloader)
                losses.append(np.copy(epoch_loss.detach().numpy()))
                tepoch.set_postfix(loss=epoch_loss.detach().numpy())
    
        return losses
    
class BFVAE(BaseVAE):

    def __init__(self,
                 input_vae: VAE,
                 **kwargs) -> None:
        super(BaseVAE, self).__init__()
        # copy input vae parameters
        self.base_vae = copy.deepcopy(input_vae)
        self.latent_dim = self.base_vae.latent_dim
        self.encoder = self.base_vae.encoder
        self.fc_mu = self.base_vae.fc_mu
        self.fc_var = self.base_vae.fc_var
        self.decoder_input = self.base_vae.decoder_input
        self.decoder = self.base_vae.decoder
        self.final_layer = self.base_vae.final_layer
        self.mask = torch.eye(self.latent_dim, dtype=bool)
        # create a new latent layer for BF-VAE
        self.latent_layer = nn.Linear(self.latent_dim,self.latent_dim)
        # fix all parameters except the final layer
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False
        for param in self.fc_var.parameters():
            param.requires_grad = False
        for param in self.decoder_input.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        self.latent_layer.weight.data *= self.mask
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z = self.latent_layer(z)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      bf_data: Tuple
                      ) -> dict:
        lf_data,hf_data = bf_data
        recons = self.forward(lf_data)[0]
        recons_loss =F.mse_loss(recons, hf_data)

        return recons_loss

    def sample(self,
               num_samples:int,
               **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = self.latent_layer(z)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
    
    def train(self, 
              trn_data: BFDataset, 
              batch_size: int = 64,
              epochs: int = 500):
        trn_dataloader = DataLoader(trn_data, batch_size=batch_size)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas= (0.9, 0.99))
        losses = []
        with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
            epoch_loss = 0
            for epoch in tepoch:
                for batch_index, data in enumerate(trn_dataloader):
                    opt.zero_grad()
                    trn_loss = self.loss_function(data)
                    trn_loss.backward()
                    epoch_loss += trn_loss.item()
                    opt.step()
    
                    epoch_loss += trn_loss
                epoch_loss /= len(trn_dataloader)
                losses.append(np.copy(epoch_loss.detach().numpy()))
                tepoch.set_postfix(loss=epoch_loss.detach().numpy())
    
        return losses
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # import test data
    path = '../Data/beam_data.npz'
    file = np.load(path)
    idx = np.linspace(0, 512, 128).astype('int')
    xL = torch.from_numpy(file['beam_yL'][:10,idx]).type(torch.FloatTensor)
    xL2 = torch.from_numpy(file['beam_yL'][:4000,idx]).type(torch.FloatTensor)
    xH = torch.from_numpy(file['beam_yH'][:10,idx]).type(torch.FloatTensor)
    
    bf_trn_data = BFDataset(xL,xH)
    
    _,input_dim = xH.shape
    latent_dim = 4
    hidden_dims = [64, 16]
    beta = 1.25
    
    batch_size = 128
    hf_trn_dataloader = DataLoader(xH, batch_size=batch_size, shuffle=False)
    lf_trn_dataloader = DataLoader(xL2, batch_size=batch_size, shuffle=False)
    bf_trn_dataloader = DataLoader(bf_trn_data, batch_size=batch_size, shuffle=False)
    
    # claim and train hfvae
    hfvae = VAE(input_dim,latent_dim,hidden_dims,beta)
    hf_loss = hfvae.train(xH, batch_size=batch_size, epochs=1000)
    
    # claim and train bfvae
    lfvae = VAE(input_dim,latent_dim,hidden_dims,beta)
    lf_loss = lfvae.train(xL2, batch_size=batch_size, epochs=2000)
    bfvae = BFVAE(lfvae)
    bf_loss = bfvae.train(bf_trn_data, batch_size=batch_size, epochs=1000)
    