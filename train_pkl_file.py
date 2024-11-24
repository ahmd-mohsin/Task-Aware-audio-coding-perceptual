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
from pkl_file_model_updated import *
import random
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict
def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

class SpectralDataset(Dataset):
    def __init__(self, clean_data_dir, noisy_data_dir, file_type='Train', device=None):
        """
        Args:
            clean_data_dir (str): Directory containing clean spectral data pkl files
            noisy_data_dir (str): Directory containing noisy spectral data pkl files
        """
        self.target_shape = (1024, 592)
        self.clean_data_dir = Path(os.path.join(clean_data_dir, file_type))
        
        # Initialize four noisy data directories
        self.noisy_data_dir1 = Path(noisy_data_dir, "complex_specs_S02_P08_U02.CH3", file_type)
        # self.noisy_data_dir2 = Path(noisy_data_dir, "complex_specs_S02_P08_U02.CH3", file_type)
        self.noisy_data_dir2 = Path(noisy_data_dir, "complex_specs_S02_P08_U03.CH3", file_type)
        self.noisy_data_dir3 = Path(noisy_data_dir, "complex_specs_S02_P08_U04.CH3", file_type)
        # self.noisy_data_dir4 = Path(noisy_data_dir, "complex_specs_S02_P08_U04.CH3", file_type)
        self.noisy_data_dir4 = Path(noisy_data_dir, "complex_specs_S02_P08_U05.CH3", file_type)
        
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
        
        # print(noisy4_phase.shape)
        # -----------------------------------------------------------

        # -----------------------------------------------------------



        # Normalize and pad tensors
        # clean_magnitude = normalize_tensor(self.pad_tensor(torch.from_numpy(clean_data["magnitude"]).float(), self.target_shape)).to(self.device)
        # clean_phase = normalize_tensor(self.pad_tensor(torch.from_numpy(clean_data["phase"]).float(), self.target_shape)).to(self.device)

        # noisy1_magnitude = normalize_tensor(self.pad_tensor(torch.from_numpy(noisy_data_1["magnitude"]).float(), self.target_shape)).to(self.device)
        # noisy1_phase = normalize_tensor(self.pad_tensor(torch.from_numpy(noisy_data_1["phase"]).float(), self.target_shape)).to(self.device)

        # noisy2_magnitude = normalize_tensor(self.pad_tensor(torch.from_numpy(noisy_data_2["magnitude"]).float(), self.target_shape)).to(self.device)
        # noisy2_phase = normalize_tensor(self.pad_tensor(torch.from_numpy(noisy_data_2["phase"]).float(), self.target_shape)).to(self.device)

        # noisy3_magnitude = normalize_tensor(self.pad_tensor(torch.from_numpy(noisy_data_3["magnitude"]).float(), self.target_shape)).to(self.device)
        # noisy3_phase = normalize_tensor(self.pad_tensor(torch.from_numpy(noisy_data_3["phase"]).float(), self.target_shape)).to(self.device)

        # noisy4_magnitude = normalize_tensor(self.pad_tensor(torch.from_numpy(noisy_data_4["magnitude"]).float(), self.target_shape)).to(self.device)
        # noisy4_phase = normalize_tensor(self.pad_tensor(torch.from_numpy(noisy_data_4["phase"]).float(), self.target_shape)).to(self.device)

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

def train_spectral_ae(batch_size=32, num_epochs=100, beta_kl=1.0, beta_rec=0.0, 
                     weight_cross_penalty=0.1, device=0, lr=2e-4, seed=0, randpca=True, z_dim=64, total_feature_after= 256 ):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("random seed: ", seed)
        print("randpca: ", randpca)

    device = torch.device("cpu") if device <= -1 else torch.device(f"cuda:{device}")

    # Define data directories
    clean_data_dir = "./Data/complex/complex_specs_S02_P08"  
    noisy_data_dir = "./Data/complex"  

    # Create dataset and dataloader
    train_dataset = SpectralDataset(
        clean_data_dir=clean_data_dir,
        noisy_data_dir=noisy_data_dir,
        device=device
    )
    
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Initialize model
    
    freq_dim = 1024
    time_dim = 592
    in_channels = 2  # magnitude, phase, db_scale
    # model = ResE4D1((in_channels, freq_dim, time_dim), (in_channels, freq_dim, time_dim), (in_channels, freq_dim, time_dim), (in_channels, freq_dim, time_dim), int(z_dim/4), int(z_dim/4), int(z_dim/4), int(z_dim/4), 4, 1).to(device)
    # model = ResE2D1((in_channels*2, freq_dim, time_dim), (in_channels*2, freq_dim, time_dim), int(z_dim/2), int(z_dim/2), 4, 1).to(device)
    model = ResE1D1((in_channels*4, freq_dim, time_dim), int(z_dim) , 4, 1).to(device)
    

    model_name = model.get_model_name()
    model.train()
    # Create a CSV file and write the header
    csv_file = Path(f'{model_name}.csv')
    
    # Write the header row if the file does not exist
    if not csv_file.exists():
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Epoch", "Average Loss", "MSE Loss", "Nuclear Loss", "Cosine Loss",
                "Magnitude Loss", "Phase Loss", "Total Loss", "PSNR", "Spectral SNR"
            ])


    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_losses = []
        mse_losses, nuc_losses, cos_losses, spec_losses, spec_snrs = [], [], [], [], []
        mag_losses, phase_losses, total_losses = [], [], []
        total_psnr = []
        epoch_dim_info = defaultdict(list)  # Collect dynamic dim_info per epoch
        batch_idx = 0
        for batch_idx, data in enumerate(track(train_loader, description=f"Epoch {epoch+1}/{num_epochs}:   Batch {batch_idx}/{len(train_loader)} ")):
            clean_audio = data["clean_audio"]
            noisy_audio_1 = data["noisy_audio_1"]
            noisy_audio_2 = data["noisy_audio_2"]
            noisy_audio_3 = data["noisy_audio_3"]
            noisy_audio_4 = data["noisy_audio_4"]
            # print(randpca)
            # Forward pass
            # decoded, mse_loss, nuc_loss, _, cos_loss, spec_loss, spec_loss_dict, spec_snr,psnr_obs, psnr_clean, dim_info = model(
            
            obs1_stacked = torch.stack([
                noisy_audio_1['magnitude'],
                noisy_audio_1['phase'],
                # obs1['db_scale']
            ], dim=1).float()  # Shape: (batch, 3, 1025, 123)
            
            obs2_stacked = torch.stack([
                noisy_audio_2['magnitude'],
                noisy_audio_2['phase'],
                # obs2['db_scale']
            ], dim=1).float()  # Shape: (batch, 3, 1025, 123)
            
            obs3_stacked = torch.stack([
                noisy_audio_3['magnitude'],
                noisy_audio_3['phase'],
                # obs1['db_scale']
            ], dim=1).float()  # Shape: (batch, 3, 1025, 123)
            
            obs4_stacked = torch.stack([
                noisy_audio_4['magnitude'],
                noisy_audio_4['phase'],
                # obs2['db_scale']
            ], dim=1).float()  # Shape: (batch, 3, 1025, 123)
            
            
            ResE2D1_obs_1 = torch.cat((obs1_stacked, obs2_stacked), dim = 1)
            ResE2D1_obs_2 = torch.cat((obs3_stacked, obs4_stacked), dim = 1)
            
            
            ResE1D1_obs = torch.cat((obs1_stacked, obs2_stacked, obs3_stacked, obs4_stacked), dim = 1)
            
            # new_obs_1 = new_obs_1[:, :,:1024,:592 ]
            # new_obs_2 = new_obs_2[:, :,:1024,:592 ]
            
            # print(ResE2D1_obs_1.shape)
            decoded, mse_loss, nuc_loss, _, cos_loss, spec_loss_dict, total_spec_snr, psnr = model(
                # obs1_stacked,
                # obs2_stacked,
                # obs3_stacked,
                # obs4_stacked,
                # ResE2D1_obs_1, 
                # ResE2D1_obs_2, 
                ResE1D1_obs,
                # True,
            )
            
            # Calculate total loss
            # loss = beta_rec * mse_loss 
            loss = (beta_rec * mse_loss + 
                   beta_kl * nuc_loss + 
                   weight_cross_penalty * cos_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    # Store losses for averaging
            # Append losses to their respective lists, checking if they are tensors
            epoch_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            mse_losses.append(mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss)
            nuc_losses.append(nuc_loss.item() if isinstance(nuc_loss, torch.Tensor) else nuc_loss)
            cos_losses.append(cos_loss.item() if isinstance(cos_loss, torch.Tensor) else cos_loss)
            mag_losses.append(spec_loss_dict["magnitude_loss"].item() if isinstance(spec_loss_dict["magnitude_loss"], torch.Tensor) else spec_loss_dict["magnitude_loss"])
            phase_losses.append(spec_loss_dict["phase_loss"].item() if isinstance(spec_loss_dict["phase_loss"], torch.Tensor) else spec_loss_dict["phase_loss"])
            total_losses.append(spec_loss_dict["total_loss"].item() if isinstance(spec_loss_dict["total_loss"], torch.Tensor) else spec_loss_dict["total_loss"])
            total_psnr.append(psnr.item() if isinstance(psnr, torch.Tensor) else psnr)
            spec_snrs.append(total_spec_snr.item() if isinstance(total_spec_snr, torch.Tensor) else total_spec_snr)

            # If batch_idx is divisible by 10, print the losses
            if batch_idx % 10 == 0:
                print(f"\nBatch {batch_idx}")
                print(f"MSE Loss: {mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss:.4f}")
                print(f"Nuclear Loss: {nuc_loss.item() if isinstance(nuc_loss, torch.Tensor) else nuc_loss:.4f}")
                print(f"Cosine Loss: {cos_loss.item() if isinstance(cos_loss, torch.Tensor) else cos_loss:.4f}")
                print(f"Spectral Magnitude Loss: {spec_loss_dict['magnitude_loss'] if isinstance(spec_loss_dict['magnitude_loss'], torch.Tensor) else spec_loss_dict['magnitude_loss']:.4f}")
                print(f"Spectral Phase Loss: {spec_loss_dict['phase_loss'] if isinstance(spec_loss_dict['phase_loss'], torch.Tensor) else spec_loss_dict['phase_loss']:.4f}")
                print(f"Spectral Total Loss: {spec_loss_dict['total_loss'] if isinstance(spec_loss_dict['total_loss'], torch.Tensor) else spec_loss_dict['total_loss']:.4f}")
                print(f"Spectral SNR: {total_spec_snr.item() if isinstance(total_spec_snr, torch.Tensor) else total_spec_snr:.2f} dB")
                print(f"PSNR: {psnr.item() if isinstance(psnr, torch.Tensor) else psnr:.2f}")
            break


        avg_loss = np.mean(epoch_losses)
        avg_mse_loss = np.mean(mse_losses)
        avg_nuc_loss = np.mean(nuc_losses)
        avg_cos_loss = np.mean(cos_losses)
        avg_mag_loss = np.mean(mag_losses)
        avg_phase_loss = np.mean(phase_losses)
        avg_total_loss = np.mean(total_losses) 
        avg_psnr = np.mean(total_psnr) 
        avg_spec_snr = np.mean(spec_snrs)
            # Save metrics for the current epoch to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1, avg_loss, avg_mse_loss, avg_nuc_loss, avg_cos_loss,
                avg_mag_loss, avg_phase_loss, avg_total_loss, avg_psnr, avg_spec_snr
            ])
        
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            base_dir = f"./models/{model_name}"
            os.makedirs(base_dir, exist_ok=True)
            checkpoint_path = f"{base_dir}/model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spectral Auto-Encoder")
    parser.add_argument("-n", "--num_epochs", type=int, default=100)
    parser.add_argument("-z", "--z_dim", type=int, default=32)
    parser.add_argument("-l", "--lr", type=float, default=2e-4)
    parser.add_argument("-bs", "--batch_size", type=int, default=2)
    parser.add_argument("-r", "--beta_rec", type=float, default=1.0)
    parser.add_argument("-k", "--beta_kl", type=float, default=0.1)
    parser.add_argument("-w", "--weight_cross_penalty", type=float, default=0.1)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-p", "--randpca", type=bool, default=True)
    parser.add_argument("-tf", "--total_feature_after", type=int, default=128)

    args = parser.parse_args()
    
    train_spectral_ae(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        beta_kl=args.beta_kl,
        beta_rec=args.beta_rec,
        weight_cross_penalty=args.weight_cross_penalty,
        device=args.device,
        lr=args.lr,
        seed=args.seed,
        randpca=True,
        z_dim=args.z_dim,
        total_feature_after=args.total_feature_after
        
    )