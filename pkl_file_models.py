import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict, OrderedDict
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def PSNR(img1, img2, PIXEL_MAX = 255.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        print("You are comparing two same images")
        return 100
    else:
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def data_pca(z):
    #PCA using SVD
	data_mean = torch.mean(z, axis=0)
	z_norm = z - data_mean
	u, s, v = torch.svd(z_norm)
	return s, v, data_mean


class LNBlock(nn.Module):
    """
    A residual block with layer normalization. The feature shapes are held constant.
    """

    def __init__(self, feature_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_shape[0], feature_shape[0], kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(feature_shape)
        self.conv2 = nn.Conv2d(feature_shape[0], feature_shape[0], kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(feature_shape)

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.ln1(y)
        y = nn.functional.relu(y)
        y = self.conv2(y)

        y += identity
        y = self.ln2(y)
        y = nn.functional.relu(y)
        return y

class SpectralResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

import torch
import torch.nn as nn

class SpectralResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class SpectralEncoder(nn.Module):
    def __init__(self, in_channels, freq_dim, time_dim, z_dim, n_res_blocks=3):
        super().__init__()
        
        # Initial projection to handle the frequency dimension
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Convolutional layers for temporal processing
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels * 128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[SpectralResBlock(128) for _ in range(n_res_blocks)]
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, z_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, freq, time)
        batch_size, channels, freq_dim, time_dim = x.shape
        
        # Reshape and process frequency dimension
        x = x.permute(0, 1, 3, 2)  # (batch, channels, time, freq)
        x = x.reshape(batch_size * channels * time_dim, freq_dim)
        x = self.freq_proj(x)
        x = x.reshape(batch_size, channels * 128, time_dim)
        
        # Process temporal dimension
        x = self.conv_layers(x)
        x = self.res_blocks(x)
        z = self.final_layers(x)
        
        return z, None  # Returning None as second output to match original interface

class SpectralDecoder(nn.Module):
    def __init__(self, out_channels, freq_dim, time_dim, z_dim, n_res_blocks=3):
        super().__init__()
        
        self.freq_dim = freq_dim
        self.time_dim = time_dim
        self.out_channels = out_channels
        
        # Initial projection from latent space
        self.initial_proj = nn.Sequential(
            nn.Linear(z_dim, 128 * time_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[SpectralResBlock(128) for _ in range(n_res_blocks)]
        )
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128 * out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(128 * out_channels),
            nn.ReLU()
        )
        
        # Final frequency projection
        self.freq_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, freq_dim)
        )
        
    def forward(self, z):
        batch_size = z.shape[0]
        
        # Project from latent space to temporal dimension
        x = self.initial_proj(z)
        x = x.reshape(batch_size, 128, self.time_dim)
        
        # Process through residual and convolutional layers
        x = self.res_blocks(x)
        x = self.conv_layers(x)
        
        # Reshape and project to frequency dimension
        x = x.reshape(batch_size * self.out_channels * self.time_dim, 128)
        x = self.freq_proj(x)
        x = x.reshape(batch_size, self.out_channels, self.time_dim, self.freq_dim)
        x = x.permute(0, 1, 3, 2)  # (batch, channels, freq, time)
        
        return x



class SpectralResE2D1(nn.Module):
    def __init__(self, z_dim1: int, z_dim2: int, n_res_blocks: int=3):
        super().__init__()
        
        # Define input shapes based on your data
        self.freq_dim = 1025
        self.time_dim = 600
        self.in_channels = 2  # magnitude, phase, db_scale
        
        self.enc1 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim1, n_res_blocks)
        self.enc2 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim2, n_res_blocks)
        self.dec = SpectralDecoder(self.in_channels, self.freq_dim, self.time_dim, z_dim1 + z_dim2, n_res_blocks)
        
        self.dimension_info = {}
    def get_dim_info(self):
        return  ["before_z1","before_z2","after_z1","after_z2"]

    def forward(self, obs1, obs2, clean_data=None, random_bottle_neck=True):
        # Process input data - stack magnitude, phase, and db_scale
        obs1_stacked = torch.stack([
            obs1['magnitude'],
            obs1['phase'],
            # obs1['db_scale']
        ], dim=1).float()  # Shape: (batch, 3, 1025, 123)
        
        obs2_stacked = torch.stack([
            obs2['magnitude'],
            obs2['phase'],
            # obs2['db_scale']
        ], dim=1).float()  # Shape: (batch, 3, 1025, 123)
        
        z1, _ = self.enc1(obs1_stacked)
        z2, _ = self.enc2(obs2_stacked)
        
        # Original data for reconstruction loss
        obs = obs1_stacked  # Using obs1 as target
        
        batch_size = z1.shape[0]
        num_features = z1.shape[1] + z2.shape[1]

        if random_bottle_neck:
            dim_p = torch.randint(8, int(num_features/2), (1,)).item()
            
            s_1, v_1, mu_1 = data_pca(z1)
            s_2, v_2, mu_2 = data_pca(z2)
            
            s_1_2 = torch.cat((s_1, s_2), 0)
            ind = torch.argsort(s_1_2, descending=True)
            ind = ind[:dim_p]
            ind_1 = ind[ind < s_1.shape[0]]
            ind_2 = ind[ind >= s_1.shape[0]] - s_1.shape[0]
            
            z1_p = torch.matmul(z1 - mu_1, v_1[:,ind_1])
            z2_p = torch.matmul(z2 - mu_2, v_2[:,ind_2])
            
            self.dimension_info = {
                "before_z1": z1.shape[1],
                "before_z2": z2.shape[1],
                "after_z1": z1_p.shape[1],
                "after_z2": z2_p.shape[1],
            }
            
            z1 = torch.matmul(z1_p, v_1[:,ind_1].T) + mu_1
            z2 = torch.matmul(z2_p, v_2[:,ind_2].T) + mu_2
        
        cos_sim = torch.nn.CosineSimilarity()
        cos_loss = torch.mean(cos_sim(z1, z2))
        z_sample = torch.cat((z1, z2), dim=1)
        
        obs_dec = self.dec(z_sample)
        
        # Calculate losses
        mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
        
        # Normalize latent representation
        z_sample = z_sample - z_sample.mean(dim=0)
        z_sample = z_sample / torch.norm(z_sample, p=2)
        nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size
        
        # Since we're working with spectral data, we'll replace PSNR with spectral SNR
        spec_snr = -10 * torch.log10(torch.mean((obs - obs_dec) ** 2) / torch.mean(obs ** 2))
        
        # Create a simplified spectral loss dictionary
        spec_loss = {
            "magnitude_loss": torch.mean((obs[:, 0] - obs_dec[:, 0]) ** 2),
            "phase_loss": torch.mean((obs[:, 1] - obs_dec[:, 1]) ** 2),
            # "db_scale_loss": torch.mean((obs[:, 2] - obs_dec[:, 2]) ** 2),
            "total_loss": torch.mean((obs - obs_dec) ** 2)
        }
        
        # Additional return values for consistency with other models
        total_mse = torch.mean(mse)
        total_nuc_loss = nuc_loss
        cross_recon_loss = torch.tensor(0)
        total_spec_loss = spec_loss["total_loss"]
        spec_loss1 = spec_loss
        total_spec_snr = spec_snr
        psnr_obs = 10 * torch.log10(1 / total_mse)
        psnr_clean = 10 * torch.log10(1 / total_mse)

        return obs_dec, total_mse, total_nuc_loss, cross_recon_loss, cos_loss, total_spec_loss, spec_loss1, total_spec_snr, psnr_obs, psnr_clean, self.dimension_info

# class SpectralResE4D1(nn.Module):
#     def __init__(self, z_dim1: int, z_dim2: int, z_dim3: int, z_dim4: int, n_res_blocks: int=3,random_bottle_neck=True):
#         super().__init__()
#         # Define input shapes based on spectral data
#         self.freq_dim = 1025
#         self.time_dim = 600
#         self.in_channels = 2  # magnitude, phase

#         # Initialize spectral encoders for each input
#         self.enc1 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim1, n_res_blocks)
#         self.enc2 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim2, n_res_blocks)
#         self.enc3 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim3, n_res_blocks)
#         self.enc4 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim4, n_res_blocks)
        
#         # Initialize decoder
#         self.dec = SpectralDecoder(
#             self.in_channels,
#             self.freq_dim * 2,  # Doubled frequency dimension for concatenated data
#             self.time_dim * 2,  # Doubled time dimension for concatenated data
#             z_dim1 + z_dim2 + z_dim3 + z_dim4,
#             n_res_blocks
#         )
        
#         self.dimension_info = {}
#     def get_dim_info(self):
#         return  ["before_z1","before_z2","before_z3","before_z4","after_z1","after_z2","after_z3","after_z4"]

#     def forward(self, obs1, obs2, obs3, obs4, clean_data=None, random_bottle_neck=True):
#         # Process input data - stack magnitude and phase for each observation
#         obs1_stacked = torch.stack([
#             obs1['magnitude'],
#             obs1['phase'],
#         ], dim=1).float()
        
#         obs2_stacked = torch.stack([
#             obs2['magnitude'],
#             obs2['phase'],
#         ], dim=1).float()
        
#         obs3_stacked = torch.stack([
#             obs3['magnitude'],
#             obs3['phase'],
#         ], dim=1).float()
        
#         obs4_stacked = torch.stack([
#             obs4['magnitude'],
#             obs4['phase'],
#         ], dim=1).float()

#         # Encode all inputs
#         z1, _ = self.enc1(obs1_stacked)
#         z2, _ = self.enc2(obs2_stacked)
#         z3, _ = self.enc3(obs3_stacked)
#         z4, _ = self.enc4(obs4_stacked)

#         # Concatenate observations for reconstruction target
#         obs12 = torch.cat((obs1_stacked, obs2_stacked), dim=3)  # Concatenate along time dimension
#         obs34 = torch.cat((obs3_stacked, obs4_stacked), dim=3)
#         obs = torch.cat((obs12, obs34), dim=2)  # Concatenate along frequency dimension

#         batch_size = z1.shape[0]
#         num_features = z1.shape[1] + z2.shape[1] + z3.shape[1] + z4.shape[1]

#         if random_bottle_neck:
#             dim_p = torch.randint(8, int(num_features/2), (1,)).item()
            
#             # Perform PCA on each latent representation
#             s_1, v_1, mu_1 = data_pca(z1)
#             s_2, v_2, mu_2 = data_pca(z2)
#             s_3, v_3, mu_3 = data_pca(z3)
#             s_4, v_4, mu_4 = data_pca(z4)
            
#             # Combine singular values and sort
#             s_1_2_3_4 = torch.cat((s_1, s_2, s_3, s_4), 0)
#             ind = torch.argsort(s_1_2_3_4, descending=True)
#             ind = ind[:dim_p]
            
#             # Split indices for each latent space
#             ind_1 = ind[ind < s_1.shape[0]]
#             ind_2 = ind[torch.logical_and(ind >= s_1.shape[0], ind < (s_1.shape[0] + s_2.shape[0]))] - s_1.shape[0]
#             ind_3 = ind[torch.logical_and(ind >= (s_1.shape[0] + s_2.shape[0]), 
#                                         ind < (s_1.shape[0] + s_2.shape[0] + s_3.shape[0]))] - (s_1.shape[0] + s_2.shape[0])
#             ind_4 = ind[ind >= (s_1.shape[0] + s_2.shape[0] + s_3.shape[0])] - (s_1.shape[0] + s_2.shape[0] + s_3.shape[0])
            
#             # Project to reduced dimension
#             z1_p = torch.matmul(z1 - mu_1, v_1[:,ind_1])
#             z2_p = torch.matmul(z2 - mu_2, v_2[:,ind_2])
#             z3_p = torch.matmul(z3 - mu_3, v_3[:,ind_3])
#             z4_p = torch.matmul(z4 - mu_4, v_4[:,ind_4])
            
#             # Store dimension information
#             self.dimension_info = {
#                 "before_z1": z1.shape[1],
#                 "before_z2": z2.shape[1],
#                 "before_z3": z3.shape[1],
#                 "before_z4": z4.shape[1],
#                 "after_z1": z1_p.shape[1],
#                 "after_z2": z2_p.shape[1],
#                 "after_z3": z3_p.shape[1],
#                 "after_z4": z4_p.shape[1],
#             }
            
#             # Project back to original space
#             z1 = torch.matmul(z1_p, v_1[:,ind_1].T) + mu_1
#             z2 = torch.matmul(z2_p, v_2[:,ind_2].T) + mu_2
#             z3 = torch.matmul(z3_p, v_3[:,ind_3].T) + mu_3
#             z4 = torch.matmul(z4_p, v_4[:,ind_4].T) + mu_4

#         # Calculate cosine similarity between all pairs
#         cos_sim = torch.nn.CosineSimilarity()
#         cos_loss = torch.mean(cos_sim(z1, z2) + cos_sim(z1, z3) + cos_sim(z1, z4) + 
#                             cos_sim(z2, z3) + cos_sim(z2, z4) + cos_sim(z3, z4))
        
#         # Concatenate latent vectors
#         z_sample = torch.cat((z1, z2, z3, z4), dim=1)
        
#         # Decode
#         obs_dec = self.dec(z_sample)
        
#         # Calculate losses
#         mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
        
#         # Normalize latent representation
#         z_sample = z_sample - z_sample.mean(dim=0)
#         z_sample = z_sample / torch.norm(z_sample, p=2)
#         nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size
        
#         # Calculate spectral SNR
#         spec_snr = -10 * torch.log10(torch.mean((obs - obs_dec) ** 2) / torch.mean(obs ** 2))
        
#         # Create spectral loss dictionary
#         spec_loss = {
#             "magnitude_loss": torch.mean((obs[:, 0] - obs_dec[:, 0]) ** 2),
#             "phase_loss": torch.mean((obs[:, 1] - obs_dec[:, 1]) ** 2),
#             "total_loss": torch.mean((obs - obs_dec) ** 2)
#         }
        
#         return obs_dec, torch.mean(mse), nuc_loss, torch.tensor(0), cos_loss, spec_loss["total_loss"], spec_loss, spec_snr, self.dimension_info


class SpectralResE4D1(nn.Module):
    def __init__(self, z_dim1: int, z_dim2: int, z_dim3: int, z_dim4: int, n_res_blocks: int=3,random_bottle_neck=True):
        super().__init__()
        # Define input shapes based on spectral data
        self.freq_dim = 1025
        self.time_dim = 600
        self.in_channels = 2  # magnitude, phase

        # Initialize spectral encoders for each input
        self.enc1 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim1, n_res_blocks)
        self.enc2 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim2, n_res_blocks)
        self.enc3 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim3, n_res_blocks)
        self.enc4 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim4, n_res_blocks)
        
        # Initialize decoder
        self.dec = SpectralDecoder(
            self.in_channels,
            self.freq_dim * 2,  # Doubled frequency dimension for concatenated data
            self.time_dim * 2,  # Doubled time dimension for concatenated data
            z_dim1 + z_dim2 + z_dim3 + z_dim4,
            n_res_blocks
        )
        
        self.dimension_info = {}
    def get_dim_info(self):
        return  ["before_z1","before_z2","before_z3","before_z4","after_z1","after_z2","after_z3","after_z4"]

    def forward(self, obs1, obs2, obs3, obs4, clean_data=None, random_bottle_neck=True):
        # Process input data - stack magnitude and phase for each observation
        obs1_stacked = torch.stack([
            obs1['magnitude'],
            obs1['phase'],
        ], dim=1).float()
        
        obs2_stacked = torch.stack([
            obs2['magnitude'],
            obs2['phase'],
        ], dim=1).float()
        
        obs3_stacked = torch.stack([
            obs3['magnitude'],
            obs3['phase'],
        ], dim=1).float()
        
        obs4_stacked = torch.stack([
            obs4['magnitude'],
            obs4['phase'],
        ], dim=1).float()

        # Encode all inputs
        z1, _ = self.enc1(obs1_stacked)
        z2, _ = self.enc2(obs2_stacked)
        z3, _ = self.enc3(obs3_stacked)
        z4, _ = self.enc4(obs4_stacked)

        # Concatenate observations for reconstruction target
        obs12 = torch.cat((obs1_stacked, obs2_stacked), dim=3)  # Concatenate along time dimension
        obs34 = torch.cat((obs3_stacked, obs4_stacked), dim=3)
        obs = torch.cat((obs12, obs34), dim=2)  # Concatenate along frequency dimension

        batch_size = z1.shape[0]
        num_features = z1.shape[1] + z2.shape[1] + z3.shape[1] + z4.shape[1]

        if random_bottle_neck:
            dim_p = torch.randint(8, int(num_features/2), (1,)).item()
            
            # Perform PCA on each latent representation
            s_1, v_1, mu_1 = data_pca(z1)
            s_2, v_2, mu_2 = data_pca(z2)
            s_3, v_3, mu_3 = data_pca(z3)
            s_4, v_4, mu_4 = data_pca(z4)
            
            # Combine singular values and sort
            s_1_2_3_4 = torch.cat((s_1, s_2, s_3, s_4), 0)
            ind = torch.argsort(s_1_2_3_4, descending=True)
            ind = ind[:dim_p]
            
            # Split indices for each latent space
            ind_1 = ind[ind < s_1.shape[0]]
            ind_2 = ind[torch.logical_and(ind >= s_1.shape[0], ind < (s_1.shape[0] + s_2.shape[0]))] - s_1.shape[0]
            ind_3 = ind[torch.logical_and(ind >= (s_1.shape[0] + s_2.shape[0]), 
                                        ind < (s_1.shape[0] + s_2.shape[0] + s_3.shape[0]))] - (s_1.shape[0] + s_2.shape[0])
            ind_4 = ind[ind >= (s_1.shape[0] + s_2.shape[0] + s_3.shape[0])] - (s_1.shape[0] + s_2.shape[0] + s_3.shape[0])
            
            # Project to reduced dimension
            z1_p = torch.matmul(z1 - mu_1, v_1[:,ind_1])
            z2_p = torch.matmul(z2 - mu_2, v_2[:,ind_2])
            z3_p = torch.matmul(z3 - mu_3, v_3[:,ind_3])
            z4_p = torch.matmul(z4 - mu_4, v_4[:,ind_4])
            
            # Store dimension information
            self.dimension_info = {
                "before_z1": z1.shape[1],
                "before_z2": z2.shape[1],
                "before_z3": z3.shape[1],
                "before_z4": z4.shape[1],
                "after_z1": z1_p.shape[1],
                "after_z2": z2_p.shape[1],
                "after_z3": z3_p.shape[1],
                "after_z4": z4_p.shape[1],
            }
            
            # Project back to original space
            z1 = torch.matmul(z1_p, v_1[:,ind_1].T) + mu_1
            z2 = torch.matmul(z2_p, v_2[:,ind_2].T) + mu_2
            z3 = torch.matmul(z3_p, v_3[:,ind_3].T) + mu_3
            z4 = torch.matmul(z4_p, v_4[:,ind_4].T) + mu_4

        # Calculate cosine similarity between all pairs
        cos_sim = torch.nn.CosineSimilarity()
        cos_loss = torch.mean(cos_sim(z1, z2) + cos_sim(z1, z3) + cos_sim(z1, z4) + 
                            cos_sim(z2, z3) + cos_sim(z2, z4) + cos_sim(z3, z4))
        
        # Concatenate latent vectors
        z_sample = torch.cat((z1, z2, z3, z4), dim=1)
        
        # Decode
        obs_dec = self.dec(z_sample)
        
        # Calculate losses
        mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
        
        # Normalize latent representation
        z_sample = z_sample - z_sample.mean(dim=0)
        z_sample = z_sample / torch.norm(z_sample, p=2)
        nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size
        
        # Calculate spectral SNR
        spec_snr = -10 * torch.log10(torch.mean((obs - obs_dec) ** 2) / torch.mean(obs ** 2))
        
        # Create spectral loss dictionary
        spec_loss = {
            "magnitude_loss": torch.mean((obs[:, 0] - obs_dec[:, 0]) ** 2),
            "phase_loss": torch.mean((obs[:, 1] - obs_dec[:, 1]) ** 2),
            "total_loss": torch.mean((obs - obs_dec) ** 2)
        }
        
        # Additional return values for E2D2 consistency
        total_mse = torch.mean(mse)
        total_nuc_loss = nuc_loss
        cross_recon_loss = torch.tensor(0)
        total_spec_loss = spec_loss["total_loss"]
        spec_loss1 = spec_loss
        total_spec_snr = spec_snr
        psnr_obs = 10 * torch.log10(1 / total_mse)
        psnr_clean = 10 * torch.log10(1 / total_mse)

        return obs_dec, total_mse, total_nuc_loss, cross_recon_loss, cos_loss, total_spec_loss, spec_loss1, total_spec_snr, psnr_obs, psnr_clean, self.dimension_info

class SpectralResE1D1(nn.Module):
    def __init__(self, z_dim: int, n_res_blocks: int=3):
        super().__init__()
        # Define input shapes based on spectral data
        self.freq_dim = 1025
        self.time_dim = 600
        self.in_channels = 2  # magnitude, phase
        
        # Initialize spectral encoder and decoder
        self.enc = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim, n_res_blocks)
        self.dec = SpectralDecoder(self.in_channels, self.freq_dim, self.time_dim, z_dim, n_res_blocks)
        
        self.dimension_info = {}

    def get_dim_info(self):
        return  ["before_z1","after_z1"]
    def forward(self, obs, clean, random_bottle_neck):
        # print(random_bottle_neck)
        # Process input data - stack magnitude and phase
        obs_stacked = torch.stack([
            obs['magnitude'],
            obs['phase'],
        ], dim=1).float()  # Shape: (batch, 2, 1025, 600)
        
        # Encode input
        z1, _ = self.enc(obs_stacked)
        
        # Split latent representation into private and shared components
        num_features = z1.shape[1] // 2
        batch_size = z1.shape[0]
        z1_private = z1[:, :num_features]
        z1_share = z1[:, num_features:]
        
        # Concatenate for decoding
        z_sample = torch.cat((z1_private, z1_share), dim=1)
        
        # Decode
        obs_dec = self.dec(z_sample)
        
        # Calculate MSE loss
        mse = 0.5 * torch.mean((obs_stacked - obs_dec) ** 2, dim=(1, 2, 3))
        
        # Calculate spectral losses
        spec_loss = {
            "magnitude_loss": torch.mean((obs_stacked[:, 0] - obs_dec[:, 0]) ** 2),
            "phase_loss": torch.mean((obs_stacked[:, 1] - obs_dec[:, 1]) ** 2),
            "total_loss": torch.mean((obs_stacked - obs_dec) ** 2)
        }
        
        # Calculate spectral SNR instead of PSNR
        spec_snr = -10 * torch.log10(torch.mean((obs_stacked - obs_dec) ** 2) / torch.mean(obs_stacked ** 2))
        
        # Normalize latent representation
        z_sample = z_sample - z_sample.mean(dim=0)
        
        # Calculate nuclear loss
        z_sample = z_sample / torch.norm(z_sample, p=2)
        nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size
        
        # Store dimension information
        self.dimension_info = {
            "before_z1": z1.shape[1],
            "after_z2": num_features
        }
        
        return obs_dec, torch.mean(mse), nuc_loss, torch.tensor(0), torch.tensor(0), spec_loss["total_loss"], spec_loss, spec_snr, self.dimension_info
    

# class SpectralResE2D2(nn.Module):
#     def __init__(self, z_dim1: int,z_dim2: int, n_res_blocks: int=3):
#         super().__init__()
#         # Define input shapes based on spectral data
#         self.freq_dim = 1025
#         self.time_dim = 600
#         self.in_channels = 2  # magnitude, phase
        
#         # Initialize spectral encoders and decoders for both branches
#         self.enc1 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim1, n_res_blocks)
#         self.enc2 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim2, n_res_blocks)
#         self.dec1 = SpectralDecoder(self.in_channels, self.freq_dim, self.time_dim, z_dim1, n_res_blocks)
#         self.dec2 = SpectralDecoder(self.in_channels, self.freq_dim, self.time_dim, z_dim2, n_res_blocks)
        
#         self.dimension_info = {}

#     def get_dim_info(self):
#         return ["before_z1", "before_z2", "after_z1", "after_z2"]

#     def forward(self, obs1,obs2, clean, random_bottle_neck):
#         # Process input data - stack magnitude and phase for both branches
#         obs_stacked = torch.stack([
#             obs1['magnitude'],
#             obs1['phase'],
#         ], dim=1).float()  # Shape: (batch, 2, 1025, 600)
        
#         clean_stacked = torch.stack([
#             obs2['magnitude'],
#             obs2['phase'],
#         ], dim=1).float()  # Shape: (batch, 2, 1025, 600)
        
#         # Encode both inputs
#         z1, _ = self.enc1(obs_stacked)
#         z2, _ = self.enc2(clean_stacked)
        
#         # Split latent representations into private and shared components
#         batch_size = z1.shape[0]
#         num_features = z1.shape[1] // 2
        
#         z1_private = z1[:, :num_features]
#         z1_share = z1[:, num_features:]
#         z2_private = z2[:, :num_features]
#         z2_share = z2[:, num_features:]
        
#         # Random bottleneck mixing of shared components if specified
#         if random_bottle_neck:
#             alpha = torch.rand(batch_size, 1, device=z1.device)
#             z_share_mixed = alpha * z1_share + (1 - alpha) * z2_share
#             z1_share = z2_share = z_share_mixed
        
#         # Concatenate for decoding
#         z1_sample = torch.cat((z1_private, z1_share), dim=1)
#         z2_sample = torch.cat((z2_private, z2_share), dim=1)
        
#         # Decode both branches
#         obs_dec = self.dec1(z1_sample)
#         clean_dec = self.dec2(z2_sample)
        
#         # Calculate MSE losses for both branches
#         mse1 = 0.5 * torch.mean((obs_stacked - obs_dec) ** 2, dim=(1, 2, 3))
#         mse2 = 0.5 * torch.mean((clean_stacked - clean_dec) ** 2, dim=(1, 2, 3))
        
#         # Calculate spectral losses for both branches
#         spec_loss1 = {
#             "magnitude_loss": torch.mean((obs_stacked[:, 0] - obs_dec[:, 0]) ** 2),
#             "phase_loss": torch.mean((obs_stacked[:, 1] - obs_dec[:, 1]) ** 2),
#             "total_loss": torch.mean((obs_stacked - obs_dec) ** 2)
#         }
        
#         spec_loss2 = {
#             "magnitude_loss": torch.mean((clean_stacked[:, 0] - clean_dec[:, 0]) ** 2),
#             "phase_loss": torch.mean((clean_stacked[:, 1] - clean_dec[:, 1]) ** 2),
#             "total_loss": torch.mean((clean_stacked - clean_dec) ** 2)
#         }
        
#         # Calculate spectral SNR for both branches
#         spec_snr1 = -10 * torch.log10(torch.mean((obs_stacked - obs_dec) ** 2) / torch.mean(obs_stacked ** 2))
#         spec_snr2 = -10 * torch.log10(torch.mean((clean_stacked - clean_dec) ** 2) / torch.mean(clean_stacked ** 2))
        
#         # Normalize latent representations
#         z1_sample = z1_sample - z1_sample.mean(dim=0)
#         z2_sample = z2_sample - z2_sample.mean(dim=0)
        
#         # Calculate nuclear losses
#         z1_sample = z1_sample / torch.norm(z1_sample, p=2)
#         z2_sample = z2_sample / torch.norm(z2_sample, p=2)
#         nuc_loss1 = torch.norm(z1_sample, p='nuc', dim=(0, 1)) / batch_size
#         nuc_loss2 = torch.norm(z2_sample, p='nuc', dim=(0, 1)) / batch_size
        
#         # Calculate cross reconstruction loss
#         cross_recon_loss = torch.mean((obs_dec - clean_dec) ** 2)
        
#         # Store dimension information
#         self.dimension_info = {
#             "before_z1": z1.shape[1],
#             "before_z2": z2.shape[1],
#             "after_z1": num_features,
#             "after_z2": num_features
#         }
        
#         # Calculate total losses
#         total_mse = torch.mean(mse1 + mse2)
#         total_nuc_loss = (nuc_loss1 + nuc_loss2) / 2
#         total_spec_loss = (spec_loss1["total_loss"] + spec_loss2["total_loss"]) / 2
#         total_spec_snr = (spec_snr1 + spec_snr2) / 2
        
#         return obs_dec, total_mse, total_nuc_loss, cross_recon_loss, torch.tensor(0), total_spec_loss, spec_loss1, total_spec_snr, self.dimension_info



class SpectralResE2D2(nn.Module):
    def __init__(self, z_dim1: int, z_dim2: int, n_res_blocks: int = 3):
        super().__init__()
        # Define input shapes based on spectral data
        self.freq_dim = 1025
        self.time_dim = 600
        self.in_channels = 2  # magnitude, phase
        
        # Initialize spectral encoders and decoders for both branches
        self.enc1 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim1, n_res_blocks)
        self.enc2 = SpectralEncoder(self.in_channels, self.freq_dim, self.time_dim, z_dim2, n_res_blocks)
        self.dec1 = SpectralDecoder(self.in_channels, self.freq_dim, self.time_dim, z_dim1, n_res_blocks)
        self.dec2 = SpectralDecoder(self.in_channels, self.freq_dim, self.time_dim, z_dim2, n_res_blocks)
        
        self.dimension_info = {}

    def get_dim_info(self):
        return ["before_z1", "before_z2", "after_z1", "after_z2"]

    def forward(self, obs1, obs2, clean, random_bottle_neck):
        # Process input data - stack magnitude and phase for both branches
        obs_stacked = torch.stack([
            obs1['magnitude'],
            obs1['phase'],
        ], dim=1).float()  # Shape: (batch, 2, 1025, 600)
        
        clean_stacked = torch.stack([
            obs2['magnitude'],
            obs2['phase'],
        ], dim=1).float()  # Shape: (batch, 2, 1025, 600)
        
        # Encode both inputs
        z1, _ = self.enc1(obs_stacked)
        z2, _ = self.enc2(clean_stacked)
        
        # Split latent representations into private and shared components
        batch_size = z1.shape[0]
        num_features = z1.shape[1] // 2
        
        z1_private = z1[:, :num_features]
        z1_share = z1[:, num_features:]
        z2_private = z2[:, :num_features]
        z2_share = z2[:, num_features:]
        
        # Random bottleneck mixing of shared components if specified
        if random_bottle_neck:
            alpha = torch.rand(batch_size, 1, device=z1.device)
            z_share_mixed = alpha * z1_share + (1 - alpha) * z2_share
            z1_share = z2_share = z_share_mixed
        
        # Concatenate for decoding
        z1_sample = torch.cat((z1_private, z1_share), dim=1)
        z2_sample = torch.cat((z2_private, z2_share), dim=1)
        
        # Decode both branches
        obs_dec = self.dec1(z1_sample)
        clean_dec = self.dec2(z2_sample)
        
        # Calculate MSE losses for both branches
        mse1 = 0.5 * torch.mean((obs_stacked - obs_dec) ** 2, dim=(1, 2, 3))
        mse2 = 0.5 * torch.mean((clean_stacked - clean_dec) ** 2, dim=(1, 2, 3))
        
        # Calculate PSNR
        max_pixel_value = 1.0  # Assuming inputs are normalized between 0 and 1
        psnr_obs = 10 * torch.log10(max_pixel_value ** 2 / mse1.mean())
        psnr_clean = 10 * torch.log10(max_pixel_value ** 2 / mse2.mean())
        
        # Calculate spectral losses for both branches
        spec_loss1 = {
            "magnitude_loss": torch.mean((obs_stacked[:, 0] - obs_dec[:, 0]) ** 2),
            "phase_loss": torch.mean((obs_stacked[:, 1] - obs_dec[:, 1]) ** 2),
            "total_loss": torch.mean((obs_stacked - obs_dec) ** 2)
        }
        
        spec_loss2 = {
            "magnitude_loss": torch.mean((clean_stacked[:, 0] - clean_dec[:, 0]) ** 2),
            "phase_loss": torch.mean((clean_stacked[:, 1] - clean_dec[:, 1]) ** 2),
            "total_loss": torch.mean((clean_stacked - clean_dec) ** 2)
        }
        
        # Calculate spectral SNR for both branches
        spec_snr1 = -10 * torch.log10(torch.mean((obs_stacked - obs_dec) ** 2) / torch.mean(obs_stacked ** 2))
        spec_snr2 = -10 * torch.log10(torch.mean((clean_stacked - clean_dec) ** 2) / torch.mean(clean_stacked ** 2))
        
        # Normalize latent representations
        z1_sample = z1_sample - z1_sample.mean(dim=0)
        z2_sample = z2_sample - z2_sample.mean(dim=0)
        
        # Calculate nuclear losses
        z1_sample = z1_sample / torch.norm(z1_sample, p=2)
        z2_sample = z2_sample / torch.norm(z2_sample, p=2)
        nuc_loss1 = torch.norm(z1_sample, p='nuc', dim=(0, 1)) / batch_size
        nuc_loss2 = torch.norm(z2_sample, p='nuc', dim=(0, 1)) / batch_size
        
        # Calculate cross reconstruction loss
        cross_recon_loss = torch.mean((obs_dec - clean_dec) ** 2)
        
        # Store dimension information
        self.dimension_info = {
            "before_z1": z1.shape[1],
            "before_z2": z2.shape[1],
            "after_z1": num_features,
            "after_z2": num_features
        }
        
        # Calculate total losses
        total_mse = torch.mean(mse1 + mse2)
        total_nuc_loss = (nuc_loss1 + nuc_loss2) / 2
        total_spec_loss = (spec_loss1["total_loss"] + spec_loss2["total_loss"]) / 2
        total_spec_snr = (spec_snr1 + spec_snr2) / 2
        

        return obs_dec, total_mse, total_nuc_loss, cross_recon_loss, torch.tensor(0), total_spec_loss, spec_loss1, total_spec_snr, psnr_obs, psnr_clean, self.dimension_info

