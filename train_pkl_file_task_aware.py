import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import librosa
import os
from speechbrain.pretrained import SpectralMaskEnhancement
from tempfile import TemporaryDirectory
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
from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics import SignalNoiseRatio as SNR
from speechbrain.processing.features import STFT
from msstftd import MultiScaleSTFTDiscriminator
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
        self.target_shape = (1025, 600)
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


# Step 1: Reconstruct the waveform from magnitude and phase for both clean and enhanced audio

# def reconstruct_waveform(magnitude, phase, sample_rate, n_fft=2048, hop_length=512, win_length=2048, window='hann'):
#     """
#     Reconstruct waveform from magnitude and phase using ISTFT.
#     """
#     magnitude = magnitude.detach().cpu().numpy()
#     phase = phase.detach().cpu().numpy()
#     # Combine magnitude and phase into a complex spectrogram
#     complex_spectrogram = magnitude * np.exp(1j * phase)
    
#     # Perform inverse STFT to get the time-domain waveform
#     y_reconstructed = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=win_length, window=window)
#     return y_reconstructed


def batch_reconstruct_waveform(magnitude_batch, phase_batch, sample_rate, n_fft=2048, hop_length=512, win_length=2048, window='hann'):
    """
    Reconstruct waveforms from batched magnitude and phase using ISTFT in PyTorch.
    """
    # Check if the batch dimensions of magnitude and phase match
    if magnitude_batch.shape != phase_batch.shape:
        raise ValueError("The shapes of magnitude and phase must match for batched processing.")
    
    batch_size = magnitude_batch.shape[0]  # Assuming shape is (batch_size, freq_bins, time_steps)
    
    # Combine magnitude and phase into a complex spectrogram for the entire batch
    complex_spectrogram_batch = magnitude_batch * torch.exp(1j * phase_batch)  # Shape: (batch_size, freq_bins, time_steps)

    # Create the window tensor based on the window type
    if window == 'hann':
        window_tensor = torch.hann_window(win_length).to(device="cuda:0")
    else:
        raise ValueError(f"Window type '{window}' is not supported. Use 'hann'.")

    # Perform inverse STFT on each item in the batch
    # Use list comprehension to apply torch.istft on each batch element
    reconstructed_waveforms = [
        torch.istft(
            complex_spectrogram_batch[i],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window_tensor,
            return_complex=False
        )
        for i in range(batch_size)
    ]

    # Stack reconstructed waveforms along the batch dimension and return
    return torch.stack(reconstructed_waveforms, dim=0)  # Shape: (batch_size, time)

def reconstruct_waveform(magnitude, phase, sample_rate, n_fft=2048, hop_length=512, win_length=2048, window='hann'):
    """
    Reconstruct waveform from magnitude and phase using ISTFT in PyTorch.
    """
    # Combine magnitude and phase into a complex spectrogram
    complex_spectrogram = magnitude * torch.exp(1j * phase)

    # Create the window tensor based on the window type
    if window == 'hann':
        window_tensor = torch.hann_window(win_length).to(device="cuda:0")
    else:
        raise ValueError(f"Window type '{window}' is not supported. Use 'hann'.")

    # Perform inverse STFT to get the time-domain waveform
    y_reconstructed = torch.istft(complex_spectrogram, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_tensor, return_complex=False)

    return y_reconstructed.unsqueeze(0)

# def reconstruct_waveform(magnitude, phase, sample_rate, n_fft=2048, hop_length=512, win_length=2048, window='hann'):
#     """
#     Reconstruct waveform from magnitude and phase using ISTFT.
#     """
#     # Convert magnitude and phase to complex form
#     complex_spectrogram = magnitude * torch.exp(1j * phase)

#     # If you're using librosa for inverse STFT (with numpy), you need to convert to numpy
#     complex_spectrogram = complex_spectrogram.detach().cpu().numpy()

#     # Perform inverse STFT to get the time-domain waveform
#     y_reconstructed = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=win_length, window=window)

#     return y_reconstructed

def save_waveform(y, sr, file_path):
    """
    Save the waveform as a .wav file.
    """
    # Convert waveform to tensor and add a batch dimension for saving
    waveform_tensor = torch.tensor(y).unsqueeze(0)
    torchaudio.save(file_path, waveform_tensor, sample_rate=sr)


SAMPLE_RATE = 8000  # Use the appropriate sample rate for your data
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", sampling_rate = SAMPLE_RATE)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model.to("cuda:0")
for param in model.parameters():
    param.requires_grad = False  # Freeze parameters

disc = MultiScaleSTFTDiscriminator(filters=8)
disc.to("cuda:0")


# speech_branin_model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir='pretrained_models/sepformer-wham-enhancement')
# for param in speech_branin_model.parameters():
#     param.requires_grad = False  # Freeze the model parameters


def task_aware(noisy_audio_batch, clean_audio_batch):
    # noisy_audio_batch and clean_audio_batch are assumed to be batches of data
    # sample_rate = 16000  # Use the appropriate sample rate for your data

    # Ensure both noisy and clean batches are the same size
    batch_size = noisy_audio_batch.size(0)
    
    # Assuming both `noisy_audio_batch` and `clean_audio_batch` are batches of tensors
    mse_loss_total = 0.0
    pesq_loss_total = 0.0
    snr_loss_total = 0.0

    # Initialize SpeechBrain model for source separation
    # speech_brain_model = speech_branin_model  # Replace with your actual model loading

    # Load and process the clean and noisy audio for the entire batch
    clean_mag, clean_phase = clean_audio_batch["magnitude"], clean_audio_batch["phase"]  # Shape: (batch_size, ...)
    noisy_mag, noisy_phase = noisy_audio_batch[:, 0], noisy_audio_batch[:, 1]  # Shape: (batch_size, ...)
    # print(clean_mag.shape)
    # Reconstruct waveforms for the entire batch
    reconstructed_clean_waveforms = batch_reconstruct_waveform(clean_mag, clean_phase, SAMPLE_RATE)  # Shape: (batch_size, time)
    reconstructed_noisy_waveforms = batch_reconstruct_waveform(noisy_mag, noisy_phase, SAMPLE_RATE)  # Shape: (batch_size, time)
    # Process noisy waveforms through the model
    inputs = processor(reconstructed_noisy_waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs_values = inputs.input_values.squeeze()  # Shape: (batch_size, time)
    inputs_values = inputs_values.to(device="cuda:0")

    # Forward pass through the model
    with torch.no_grad():
        logits = model(inputs_values).logits  # Shape: (batch_size, time, classes)

    # Obtain enhanced audio by taking the argmax over logits
    enhanced_audio = torch.argmax(logits, dim=-1).to(device="cuda:0")  # Shape: (batch_size, time)
    # print(reconstructed_clean_waveforms.shape, reconstructed_noisy_waveforms.shape, enhanced_audio.shape, logits.shape)

    # Ensure enhanced audio and reconstructed clean waveforms have compatible lengths
    # ----------------------------------------------------------
    # min_length = min(reconstructed_clean_waveforms.shape[-1], enhanced_audio.shape[-1])  (2, 958)
    # enhanced_audio = enhanced_audio[..., :min_length]
    # reconstructed_clean_waveforms = reconstructed_clean_waveforms[..., :min_length]
    # ----------------------------------------------------------
    # Get the lengths of both tensors
    enhanced_length = enhanced_audio.shape[-1]
    reconstructed_length = reconstructed_clean_waveforms.shape[-1]
    enhanced_audio, reconstructed_clean_waveforms = enhanced_audio.float(), reconstructed_clean_waveforms.float()
    # Check which tensor is shorter and interpolate it to match the longer one
    if enhanced_length < reconstructed_length:
        # Interpolate enhanced_audio to match reconstructed_clean_waveforms
        enhanced_audio = F.interpolate(enhanced_audio.unsqueeze(1), size=reconstructed_length, mode='linear', align_corners=False).squeeze(1)
    elif reconstructed_length < enhanced_length:
        # Interpolate reconstructed_clean_waveforms to match enhanced_audio
        reconstructed_clean_waveforms = F.interpolate(reconstructed_clean_waveforms.unsqueeze(1), size=enhanced_length, mode='linear', align_corners=False).squeeze(1)
    # ----------------------------------------------------------
    # Convert numpy arrays to PyTorch tensors if needed
    if isinstance(reconstructed_clean_waveforms, np.ndarray):
        reconstructed_clean_waveforms = torch.tensor(reconstructed_clean_waveforms)

    if reconstructed_clean_waveforms.ndimension() == 2:  # Shape: (batch_size, time)
        reconstructed_clean_waveforms = reconstructed_clean_waveforms.to(device="cuda:0")


    # print(enhanced_audio.shape, reconstructed_clean_waveforms.shape)
    # Step 3: Compute losses for the entire batch
    mse_loss = F.mse_loss(enhanced_audio, reconstructed_clean_waveforms, reduction='mean')

    # Optional: Compute additional metrics if required
    # pesq_loss and snr_loss calculations can be added here if PESQ and SNR functions support batched input

    # Output the average losses
    avg_mse_loss_enc_clean = mse_loss.item()
    # avg_pesq_loss and avg_snr_loss can be calculated if using PESQ and SNR metrics

    # ----------------To be checked by professional :)------------------------------------
    discriminator_loss = 0.0

    # Assuming enhanced_audio and ground_truth_clean_waveforms are your inputs with shape [2, 958]
    # They need to be reshaped to [batch_size, channels, length], e.g., [2, 1, 958]
    enhanced_audio = enhanced_audio.unsqueeze(1)  # Shape becomes [2, 1, 958]
    ground_truth_clean_waveforms = reconstructed_clean_waveforms.unsqueeze(1)  # Shape becomes [2, 1, 958]

    # print(enhanced_audio.shape)
    # Forward pass through the discriminator
    y_disc_enhanced, fmap_enhanced = disc(enhanced_audio)
    y_disc_ground_truth, fmap_ground_truth = disc(ground_truth_clean_waveforms)

    
    # Calculate L1 loss between feature maps for feature matching
    for fmap_enh, fmap_gt in zip(fmap_enhanced, fmap_ground_truth):
        for feat_enh, feat_gt in zip(fmap_enh, fmap_gt):
            discriminator_loss += F.l1_loss(feat_enh, feat_gt)

    # Calculate the adversarial loss for logits (optional, depending on your use case)
    for y_enh, y_gt in zip(y_disc_enhanced, y_disc_ground_truth):
        discriminator_loss += F.mse_loss(y_enh, torch.ones_like(y_enh))  # adversarial loss for enhanced
        discriminator_loss += F.mse_loss(y_gt, torch.zeros_like(y_gt))   # adversarial loss for ground truth

    # print(avg_mse_loss_enc_clean, discriminator_loss.item())
    return avg_mse_loss_enc_clean, discriminator_loss.item()  # , avg_pesq_loss, avg_snr_loss if implemented
    # for num_index in range(batch_size):
    #     # Load and process the clean and noisy audio for the current batch index
    #     # Process audio batch
    #     clean_mag, clean_phase = clean_audio_batch["magnitude"][num_index], clean_audio_batch["phase"][num_index]
    #     noisy_mag, noisy_phase = noisy_audio_batch[num_index][0], noisy_audio_batch[num_index][1]

    #     # Reconstruct waveforms
    #     reconstructed_clean_waveform = reconstruct_waveform(clean_mag, clean_phase, sample_rate)
    #     reconstructed_noisy_waveform = reconstruct_waveform(noisy_mag, noisy_phase, sample_rate)
    #     # print(reconstructed_noisy_waveform.shape)
    #     # Step 2: Use SpeechBrain's model to enhance the noisy audio
    #     # est_sources = speech_branin_model.separate_file(path=None, wav=reconstructed_noisy_waveform)
    #     print("-----------------------")
    #     # est_sources = speech_branin_model.separate_batch(mix=reconstructed_noisy_waveform)
    #     # enhanced_audio = est_sources[:, :, 0].detach().cpu()  # First separated source as enhanced
    #     # Enhance audio using the denoiser
    #     # Process and pass through the model
    #     inputs = processor(reconstructed_noisy_waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    #     inputs_values = inputs.input_values.squeeze().unsqueeze(0)  # Remove any extra dimensions
        
    #     # print(inputs_values.shape)
    #     with torch.no_grad():
    #         logits = model(inputs_values).logits
    #     # enhanced_audio = torch.argmax(logits, dim=-1).cpu()
    #     enhanced_audio = torch.argmax(logits, dim=-1).to(device="cuda:0")
    #     # enhanced_audio = torch.argmax(logits, dim=-1)

    #     print("-----------------------")

        
    #     # with TemporaryDirectory() as tempdir:
    #     #     temp_clean_path = os.path.join(tempdir, f"reconstructed_clean_{num_index}.wav")
    #     #     temp_noisy_path = os.path.join(tempdir, f"reconstructed_noisy_{num_index}.wav")
            
    #     #     # Reconstruct clean and noisy waveforms for the current sample
    #     #     reconstructed_clean_waveform = reconstruct_waveform(clean_audio_batch["magnitude"][num_index], clean_audio_batch["phase"][num_index], sample_rate)
    #     #     save_waveform(reconstructed_clean_waveform, sample_rate, temp_clean_path)

    #     #     reconstructed_noisy_waveform = reconstruct_waveform(noisy_audio_batch[num_index][0], noisy_audio_batch[num_index][1], sample_rate)
    #     #     save_waveform(reconstructed_noisy_waveform, sample_rate, temp_noisy_path)

    #     #     # Step 2: Use SpeechBrain's model to enhance the noisy audio
    #     #     est_sources = speech_brain_model.separate_file(path=temp_noisy_path)
    #     #     enhanced_audio = est_sources[:, :, 0].detach().cpu()  # Take the first separated source as the enhanced audio

    #     # Ensure enhanced audio and reconstructed clean waveform have compatible lengths
    #     min_length = min(reconstructed_clean_waveform.shape[-1], enhanced_audio.shape[-1])
    #     enhanced_audio = enhanced_audio[..., :min_length]
    #     reconstructed_clean_waveform = reconstructed_clean_waveform[..., :min_length]

    #     # Convert numpy arrays to PyTorch tensors if needed
    #     if isinstance(reconstructed_clean_waveform, np.ndarray):
    #         reconstructed_clean_waveform = torch.tensor(reconstructed_clean_waveform)

    #     if reconstructed_clean_waveform.ndimension() == 1:
    #         reconstructed_clean_waveform = reconstructed_clean_waveform.unsqueeze(0)  # Add batch dimension

    #     # Step 3: Compute losses for the current batch entry
    #     mse_loss = F.mse_loss(enhanced_audio, reconstructed_clean_waveform)
    #     # pesq_metric = PESQ(fs=sample_rate, mode='wb')
    #     # pesq_loss = pesq_metric(enhanced_audio, reconstructed_clean_waveform)

    #     # snr_metric = SNR()
    #     # snr_loss = snr_metric(enhanced_audio, reconstructed_clean_waveform)

    #     # Add the current losses to the totals
    #     mse_loss_total += mse_loss.item()
    #     # pesq_loss_total += pesq_loss.item()
    #     # snr_loss_total += snr_loss.item()

    #     # # Output the losses for the current batch entry
    #     # print(f"Entry {num_index}:")
    #     # print("MSE Loss:", mse_loss.item())
    #     # print("Perceptual Loss (PESQ):", pesq_loss.item())
    #     # print("SNR Loss:", snr_loss.item())
    
    # # Step 4: Calculate the average losses over the whole batch
    # avg_mse_loss_enc_clean = mse_loss_total / batch_size
    # avg_pesq_loss = pesq_loss_total / batch_size
    # avg_snr_loss = snr_loss_total / batch_size
    
    # return avg_mse_loss_enc_clean, avg_pesq_loss, avg_snr_loss
# def task_aware(noisy_audio, clean_audio):
#     # noisy_audio, clean_audio = noisy_audio[0], clean_audio.squeeze()
#     # print(clean_audio)
#     num_index = 0
#     # reconstructed_clean_waveform = reconstruct_waveform(clean_audio["magnitude"][num_index], clean_audio["phase"][num_index])
#     # reconstructed_noisy_waveform = reconstruct_waveform(noisy_audio[num_index][0], noisy_audio[num_index][1])
    
    
#     sample_rate = 16000  # Use the appropriate sample rate for your data
#     # Load SpeechBrain model for source separation and freeze its parameters

#     # Save the reconstructed waveform in a temporary directory
#     with TemporaryDirectory() as tempdir:
#         temp_clean_path = os.path.join(tempdir, "reconstructed_clean.wav")
#         temp_noisy_path = os.path.join(tempdir, "reconstructed_noisy.wav")
#         # ---------------------------------
#         reconstructed_clean_waveform = reconstruct_waveform(clean_audio["magnitude"][num_index], clean_audio["phase"][num_index], sample_rate)
#         save_waveform(reconstructed_clean_waveform, sample_rate, temp_clean_path)
#         # ---------------------------------
#         reconstructed_noisy_waveform = reconstruct_waveform(noisy_audio[num_index][0], noisy_audio[num_index][1], sample_rate)
#         save_waveform(reconstructed_noisy_waveform, sample_rate, temp_noisy_path)
#         # ---------------------------------
#         # torchaudio.save(temp_clean_path, reconstructed_clean_waveform.unsqueeze(0), sample_rate=sample_rate)
#         # torchaudio.save(temp_noisy_path, reconstructed_noisy_waveform.unsqueeze(0), sample_rate=sample_rate)

#         # Step 2: Use Hugging Face's SpeechBrain model to enhance the saved clean audio file
#         # model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir='pretrained_models/sepformer-wham-enhancement')
#         est_sources = speech_branin_model.separate_file(path=temp_noisy_path)
#         enhanced_audio = est_sources[:, :, 0].detach().cpu()  # Take the first separated source as the enhanced audio

#     # Ensure enhanced audio and reconstructed clean waveform have compatible lengths
#     min_length = min(reconstructed_clean_waveform.shape[-1], enhanced_audio.shape[-1])
#     enhanced_audio = enhanced_audio[..., :min_length]
#     reconstructed_clean_waveform = reconstructed_clean_waveform[..., :min_length]

#     # Convert the numpy array to a PyTorch tensor
#     if isinstance(reconstructed_clean_waveform, np.ndarray):
#         reconstructed_clean_waveform = torch.tensor(reconstructed_clean_waveform)

#     # Reshape the 1D tensor to match the shape of the 2D tensor
#     if reconstructed_clean_waveform.ndimension() == 1:
#         reconstructed_clean_waveform = reconstructed_clean_waveform.unsqueeze(0)  # Add batch dimension
#     # Step 3: Compute losses

#     # Mean Squared Error (MSE) Loss
#     print("--------------------------------")
#     print(enhanced_audio.shape)
#     print(reconstructed_clean_waveform.shape)
#     print("--------------------------------")
#     mse_loss = F.mse_loss(enhanced_audio, reconstructed_clean_waveform)

#     # Perceptual Evaluation of Speech Quality (PESQ) - perceptual loss
#     pesq_metric = PESQ(fs=sample_rate, mode='wb')
#     pesq_loss = pesq_metric(enhanced_audio, reconstructed_clean_waveform)

#     # Signal-to-Noise Ratio (SNR) Loss
#     snr_metric = SNR()
#     snr_loss = snr_metric(enhanced_audio, reconstructed_clean_waveform)
#     # Output the losses
#     print("MSE Loss:", mse_loss.item())
#     print("Perceptual Loss (PESQ):", pesq_loss.item())
#     print("SNR Loss:", snr_loss.item())


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
    
    
    model = SpectralResE2D2(z_dim1=int(z_dim/2), z_dim2=int(z_dim/2), n_res_blocks=3, total_features_after=total_feature_after).to(device)
    # model = SpectralResE4D1(z_dim1=int(z_dim/2), z_dim2=int(z_dim/2), z_dim3=int(z_dim/2), z_dim4=int(z_dim/2), n_res_blocks=3, random_bottle_neck=True, total_features_after=total_feature_after).to(device)
    # model = SpectralResE2D1(z_dim1=int(z_dim/2), z_dim2=int(z_dim/2), n_res_blocks=3, total_features_after=total_feature_after).to(device)
    # model = SpectralResE1D1(z_dim=int(z_dim), n_res_blocks=3, total_features_after=total_feature_after).to(device)
    # model_name = f"SpecResE2D1_z_dim_{int(z_dim/2)}"
    model_name = model.get_model_name()
    model.train()
    # Create a CSV file and write the header
    csv_file = f'{model_name}.csv'
    # Assuming consistent keys in dim_info, initialize them once
    dim_keys = []  # Set of unique keys in dim_info, assumed to be static
    # # Sample batch to retrieve `dim_info` keys
    # sample_batch = next(iter(train_loader))

    # _, _, _, _, _, _, _, _, dim_info_sample = model(
    #     sample_batch["noisy_audio_1"],
    #     sample_batch["noisy_audio_2"],
    #     sample_batch["clean_audio"],
    #     random_bottle_neck=randpca
    # )
    dim_keys = model.get_dim_info()  # Store sorted keys for consistent order

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Initialize CSV header
    header = [
        "Epoch", "Avg_MSE_Loss", "Avg_Nuclear_Loss", "Avg_Cosine_Loss", 
        "Avg_Spectral_Loss", "Avg_Spectral_SNR", 
        "Avg_Magnitude_Loss", "Avg_Phase_Loss", "Avg_Total_Loss", "psnr_obs", "psnr_clean"
    ] + dim_keys

    # Write the header once
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    for epoch in range(num_epochs):
        epoch_losses = []
        mse_losses, nuc_losses, cos_losses, spec_losses, spec_snrs = [], [], [], [], []
        mag_losses, phase_losses, total_losses = [], [], []
        total_psnr_obs = []
        total_psnr_clean = []
        total_denoised_mse_loss = []
        epoch_dim_info = defaultdict(list)  # Collect dynamic dim_info per epoch
        batch_idx = 0
        for batch_idx, data in enumerate(track(train_loader, description=f"Epoch {epoch+1}/{num_epochs}:   Batch {batch_idx}/{len(train_loader)} ")):
            clean_audio = data["clean_audio"]
            noisy_audio_1 = data["noisy_audio_1"]
            noisy_audio_2 = data["noisy_audio_2"]
            noisy_audio_3 = data["noisy_audio_3"]
            noisy_audio_4 = data["noisy_audio_4"]
            if clean_audio["magnitude"].shape[0] != batch_size:
                continue
            # a =clean_audio["magnitude"]
            # print(f"Shape of  {a.shape}")

            # print(randpca)
            # Forward pass
            decoded, mse_loss, nuc_loss, _, cos_loss, spec_loss, spec_loss_dict, spec_snr,psnr_obs, psnr_clean, dim_info = model(
                noisy_audio_1, 
                noisy_audio_2, 
                # noisy_audio_3, 
                # noisy_audio_4, 
                clean_audio,
                True,
            )
            denoised_mse_loss = task_aware(decoded, clean_audio)
            # Calculate total loss
            # loss = beta_rec * mse_loss 
            loss = (beta_rec * mse_loss + 
                   beta_kl * nuc_loss + 
                   weight_cross_penalty * cos_loss + 
                   spec_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    # Store losses for averaging
            epoch_losses.append(loss.item())
            mse_losses.append(mse_loss.item())
            total_denoised_mse_loss.append(denoised_mse_loss)
            nuc_losses.append(nuc_loss.item())
            cos_losses.append(cos_loss.item())
            spec_losses.append(spec_loss.item())
            spec_snrs.append(spec_snr.item())
            mag_losses.append(spec_loss_dict["magnitude_loss"].item())
            phase_losses.append(spec_loss_dict["phase_loss"].item())
            total_losses.append(spec_loss_dict["total_loss"].item())
            total_psnr_clean.append(psnr_clean.item())
            total_psnr_obs.append(psnr_obs.item())
        

            # if batch_idx % 10 == 0:
            #     print(f"\nBatch {batch_idx}")
            #     print(f"MSE Loss: {mse_loss.item():.4f}")
            #     print(f"Nuclear Loss: {nuc_loss.item():.4f}")
            #     print(f"Cosine Loss: {cos_loss.item():.4f}")
            #     print(f"Spectral Loss: {spec_loss.item():.4f}")
            #     print(f"Spectral SNR: {spec_snr.item():.2f} dB")
            #     for key in sorted(dim_info.keys()):
            #         # avg_dim_value = np.mean(epoch_dim_info[key])  # Average value for this key
            #         print(f"{key}: {dim_info[key]}")

    
        avg_loss = np.mean(epoch_losses)
        avg_mse_loss = np.mean(mse_losses)
        avg_nuc_loss = np.mean(nuc_losses)
        avg_cos_loss = np.mean(cos_losses)
        avg_spec_loss = np.mean(spec_losses)
        avg_spec_snr = np.mean(spec_snrs)
        avg_mag_loss = np.mean(mag_losses)
        avg_phase_loss = np.mean(phase_losses)
        avg_denoised_mse_loss = np.mean(total_denoised_mse_loss)
        avg_total_loss = np.mean(total_losses) 
        avg_psnr_obs = np.mean(total_psnr_obs) 
        avg_psnr_clean = np.mean(total_psnr_clean) 
            # Print batch statistics
        
        
        # Prepare data for CSV row
        epoch_row = [
            epoch + 1, avg_mse_loss, avg_nuc_loss, avg_cos_loss, 
            avg_spec_loss, avg_spec_snr, avg_mag_loss, avg_phase_loss, avg_total_loss,avg_psnr_obs,avg_psnr_clean
        ]
        # print(dim_info)
        # Add averaged dim_info values to the row
        for key in dim_info.keys():
            # avg_dim_value = np.mean(epoch_dim_info[key])  # Average value for this key
            epoch_row.append(dim_info[key])
        
        # Save the row to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(epoch_row)



        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
        print(f"\nEpoch {epoch+1} Average Denoised MSE Loss: {avg_denoised_mse_loss:.4f}")
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
    parser.add_argument("-z", "--z_dim", type=int, default=256)
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