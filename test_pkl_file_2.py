import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from rich.progress import track
import torch.optim as optim
import csv
from pkl_file_models import *
import random
import torch.nn.functional as F
from collections import defaultdict


class SpectralDataset(Dataset):
    def __init__(self, clean_data_dir, noisy_data_dir, file_type='Train', device=None):
        """
        Args:
            clean_data_dir (str): Directory containing clean spectral data pkl files
            noisy_data_dir (str): Directory containing noisy spectral data pkl files
        """
        self.target_shape = (1025, 600)
        self.clean_data_dir = Path(os.path.join(clean_data_dir, file_type))
        
        # Initialize four noisy data directories
        self.noisy_data_dir1 = Path(noisy_data_dir, "complex_specs_S02_P08_U02.CH3", file_type)
        self.noisy_data_dir2 = Path(noisy_data_dir, "complex_specs_S02_P08_U02.CH3", file_type)
        # self.noisy_data_dir2 = Path(noisy_data_dir, "complex_specs_S02_P08_U03.CH3", file_type)
        self.noisy_data_dir3 = Path(noisy_data_dir, "complex_specs_S02_P08_U04.CH3", file_type)
        self.noisy_data_dir4 = Path(noisy_data_dir, "complex_specs_S02_P08_U04.CH3", file_type)
        # self.noisy_data_dir4 = Path(noisy_data_dir, "complex_specs_S02_P08_U05.CH3", file_type)
        
        self.device = device
        
        # Get all pkl files in clean and noisy directories
        self.clean_files = sorted(list(self.clean_data_dir.glob("*.pkl")))
        self.noisy_files1 = sorted(list(self.noisy_data_dir1.glob("*.pkl")))
        self.noisy_files2 = sorted(list(self.noisy_data_dir2.glob("*.pkl")))
        self.noisy_files3 = sorted(list(self.noisy_data_dir3.glob("*.pkl")))
        self.noisy_files4 = sorted(list(self.noisy_data_dir4.glob("*.pkl")))

        assert len(self.clean_files) > 0, f"No pkl files found in {clean_data_dir}"
        assert len(self.noisy_files1) > 0, f"No pkl files found in {self.noisy_data_dir1}"
        assert len(self.noisy_files2) > 0, f"No pkl files found in {self.noisy_data_dir2}"
        assert len(self.noisy_files3) > 0, f"No pkl files found in {self.noisy_data_dir3}"
        assert len(self.noisy_files4) > 0, f"No pkl files found in {self.noisy_data_dir4}"

    def pad_tensor(self, tensor, target_shape):
        # Calculate padding dimensions
        padding = (0, target_shape[1] - tensor.shape[1], 0, target_shape[0] - tensor.shape[0])
        # Pad the tensor and return it
        return F.pad(tensor, padding, "constant", 0)

    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Load clean data
        with open(self.clean_files[idx], 'rb') as f:
            clean_data = pickle.load(f)
            
        # Load corresponding noisy data files (four noisy versions per clean file)
        with open(self.noisy_files1[idx], 'rb') as f:
            noisy_data_1 = pickle.load(f)
        with open(self.noisy_files2[idx], 'rb') as f:
            noisy_data_2 = pickle.load(f)
        with open(self.noisy_files3[idx], 'rb') as f:
            noisy_data_3 = pickle.load(f)
        with open(self.noisy_files4[idx], 'rb') as f:
            noisy_data_4 = pickle.load(f)

        clean_magnitude = self.pad_tensor(torch.from_numpy(clean_data["magnitude"]).float(), self.target_shape).to(self.device)
        clean_phase = self.pad_tensor(torch.from_numpy(clean_data["phase"]).float(), self.target_shape).to(self.device)

        noisy1_magnitude = self.pad_tensor(torch.from_numpy(noisy_data_1["magnitude"]).float(), self.target_shape).to(self.device)
        noisy1_phase = self.pad_tensor(torch.from_numpy(noisy_data_1["phase"]).float(), self.target_shape).to(self.device)

        noisy2_magnitude = self.pad_tensor(torch.from_numpy(noisy_data_2["magnitude"]).float(), self.target_shape).to(self.device)
        noisy2_phase = self.pad_tensor(torch.from_numpy(noisy_data_2["phase"]).float(), self.target_shape).to(self.device)

        noisy3_magnitude = self.pad_tensor(torch.from_numpy(noisy_data_3["magnitude"]).float(), self.target_shape).to(self.device)
        noisy3_phase = self.pad_tensor(torch.from_numpy(noisy_data_3["phase"]).float(), self.target_shape).to(self.device)

        noisy4_magnitude = self.pad_tensor(torch.from_numpy(noisy_data_4["magnitude"]).float(), self.target_shape).to(self.device)
        noisy4_phase = self.pad_tensor(torch.from_numpy(noisy_data_4["phase"]).float(), self.target_shape).to(self.device)

        return {
            "clean_audio": {
                "magnitude": clean_magnitude,
                "phase": clean_phase,
                "params": clean_data["params"]
            },
            "noisy_audio_1": {
                "magnitude": noisy1_magnitude,
                "phase": noisy1_phase,
                "params": noisy_data_1["params"]
            },
            "noisy_audio_2": {
                "magnitude": noisy2_magnitude,
                "phase": noisy2_phase,
                "params": noisy_data_2["params"]
            },
            "noisy_audio_3": {
                "magnitude": noisy3_magnitude,
                "phase": noisy3_phase,
                "params": noisy_data_3["params"]
            },
            "noisy_audio_4": {
                "magnitude": noisy4_magnitude,
                "phase": noisy4_phase,
                "params": noisy_data_4["params"]
            }
        }



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

import torch
import numpy as np
import csv
import os
from torch.utils.data import DataLoader
from collections import defaultdict
from pkl_file_models import SpectralResE2D1  # Ensure this imports your model 

from train_pkl_file import SpectralDataset

def test_spectral_ae(batch_size=8, device=0, model_path="/home/ahmed/Task-Aware-audio-coding-perceptual/models/SpecResE4D1/model_epoch_100.pth"):
    # Set device
    device = torch.device("cpu") if device <= -1 else torch.device(f"cuda:{device}")

    # Define data directories (update as needed)
    clean_data_dir = "./Data/complex/complex_specs_S02_P08"  
    noisy_data_dir = "./Data/complex"  

    # Load test dataset and dataloader
    test_dataset = SpectralDataset(
        clean_data_dir=clean_data_dir,
        noisy_data_dir=noisy_data_dir,
        device=device,
        file_type="Test"
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # Initialize model and load the checkpoint
    z_dim = 32
    model = SpectralResE1D1(z_dim=int(z_dim/2), n_res_blocks=3).to(device)
    model_name = "SpectralResE1D1" 
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode

    # Initialize test metrics
    mse_losses, nuc_losses, cos_losses, spec_losses, spec_snrs = [], [], [], [], []
    mag_losses, phase_losses, total_losses = [], [], []
    total_psnr_obs = []
    total_psnr_clean = []
    dim_keys = sorted(model.get_dim_info())  # Store sorted keys for consistent order

    # Testing loop
    with torch.no_grad():  # Disable gradient computation for testing
        for data in test_loader:
            clean_audio = data["clean_audio"]
            noisy_audio_1 = data["noisy_audio_1"]
            noisy_audio_2 = data["noisy_audio_2"]
            noisy_audio_3 = data["noisy_audio_3"]
            noisy_audio_4 = data["noisy_audio_4"]

            # Forward pass
            decoded, mse_loss, nuc_loss, _, cos_loss, spec_loss, spec_loss_dict, spec_snr,psnr_obs, psnr_clean, dim_info = model(
                noisy_audio_1, 
                noisy_audio_2, 
                noisy_audio_3, 
                noisy_audio_4, 
                clean_audio,
                random_bottle_neck=True
            )

           
            # Collect metrics
           
            mse_losses.append(mse_loss.item())
            nuc_losses.append(nuc_loss.item())
            cos_losses.append(cos_loss.item())
            spec_losses.append(spec_loss.item())
            spec_snrs.append(spec_snr.item())
            mag_losses.append(spec_loss_dict["magnitude_loss"].item())
            phase_losses.append(spec_loss_dict["phase_loss"].item())
            total_losses.append(spec_loss_dict["total_loss"].item())
            total_psnr_clean.append(psnr_clean.item())
            total_psnr_obs.append(psnr_obs.item())
        


    # Calculate average metrics
    avg_mse_loss = np.mean(mse_losses)
    avg_nuc_loss = np.mean(nuc_losses)
    avg_cos_loss = np.mean(cos_losses)
    avg_spec_loss = np.mean(spec_losses)
    avg_spec_snr = np.mean(spec_snrs)
    avg_mag_loss = np.mean(mag_losses)
    avg_phase_loss = np.mean(phase_losses)
    avg_total_loss = np.mean(total_losses)
    avg_psnr_obs = np.mean(total_psnr_obs) 
    avg_psnr_clean = np.mean(total_psnr_clean)

    # Print results
    print("Test Results:")
    print(f"Avg MSE Loss: {avg_mse_loss:.4f}")
    print(f"Avg Nuclear Loss: {avg_nuc_loss:.4f}")
    print(f"Avg Cosine Loss: {avg_cos_loss:.4f}")
    print(f"Avg Spectral Loss: {avg_spec_loss:.4f}")
    print(f"Avg Spectral SNR: {avg_spec_snr:.2f} dB")
    print(f"Avg Magnitude Loss: {avg_mag_loss:.4f}")
    print(f"Avg Phase Loss: {avg_phase_loss:.4f}")
    print(f"Avg Total Loss: {avg_total_loss:.4f}")
    print(f"Avg PSNR Observed: {avg_psnr_obs:.2f} dB")  
    print(f"Avg PSNR Clean: {avg_psnr_clean:.2f} dB")

    # Save results to CSV
    results_csv = f'{model_name}_test_results.csv'
    # Prepare data for CSV row
    header = [
         "Epoch", "Avg_MSE_Loss", "Avg_Nuclear_Loss", "Avg_Cosine_Loss", 
        "Avg_Spectral_Loss", "Avg_Spectral_SNR", 
        "Avg_Magnitude_Loss", "Avg_Phase_Loss", "Avg_Total_Loss", "psnr_obs", "psnr_clean"
    ] + dim_keys

    # Write the header once
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    epoch_row = [
         avg_mse_loss, avg_nuc_loss, avg_cos_loss, 
            avg_spec_loss, avg_spec_snr, avg_mag_loss, avg_phase_loss, avg_total_loss,avg_psnr_obs,avg_psnr_clean
    ]
    
    # Add averaged dim_info values to the row
    for key in sorted(dim_info.keys()):
        # avg_dim_value = np.mean(epoch_dim_info[key])  # Average value for this key
        epoch_row.append(dim_info[key])
    
    # Save the row to CSV
    with open(results_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(epoch_row)

    print(f"\nTest results saved to {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spectral Auto-Encoder")
    parser.add_argument("-n", "--num_epochs", type=int, default=100)
    parser.add_argument("-z", "--z_dim", type=int, default=32)
    parser.add_argument("-l", "--lr", type=float, default=2e-4)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-r", "--beta_rec", type=float, default=0.1)
    parser.add_argument("-k", "--beta_kl", type=float, default=1.0)
    parser.add_argument("-w", "--weight_cross_penalty", type=float, default=0.1)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-p", "--randpca", type=bool, default=False)
    
    args = parser.parse_args()
    test_spectral_ae()