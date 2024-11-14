import torch
from argparse import ArgumentParser
from os import makedirs
from os.path import join, dirname
from scipy.io.wavfile import write
import glob
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname
from argparse import ArgumentParser
from librosa import resample
import os
# Set CUDA architecture list
print(os.getcwd())
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec

def enhance_audio_batch(y, sr, ckpt, device="cuda", sampler_type="pc", corrector="ald", corrector_steps=1, snr=0.5, N=30, t_eps=0.03):
    """
    Enhances a batch of audio data.

    Parameters:
        y (torch.Tensor): Tensor containing the audio data with shape (batch_size, num_samples).
        sr (int): Sample rate of the input audio.
        ckpt (str): Path to the model checkpoint.
        device (str): Device to perform inference on (e.g., "cuda" or "cpu").
        sampler_type (str): Sampler type for the PC sampler.
        corrector (str): Corrector class for the PC sampler.
        corrector_steps (int): Number of corrector steps.
        snr (float): Signal-to-noise ratio value for (annealed) Langevin dynamics.
        N (int): Number of reverse steps.
        t_eps (float): Minimum process time.

    Returns:
        torch.Tensor: Enhanced audio with the same shape as the input.
        int: Target sample rate.
    """
    # Load score model
    model = ScoreModel.load_from_checkpoint(ckpt, map_location=device)
    model.t_eps = t_eps
    model.eval()

    # Set the target sample rate and padding mode based on the model backbone
    if model.backbone == 'ncsnpp_48k':
        target_sr = 48000
        pad_mode = "reflection"
    elif model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    # Resample if necessary
    if sr != target_sr:
        y = torch.tensor([resample(ch.numpy(), orig_sr=sr, target_sr=target_sr) for ch in y])

    T_orig = y.size(1)

    # Normalize
    norm_factor = y.abs().max(dim=1, keepdim=True).values
    y = y / norm_factor

    # Prepare DNN input
    Y = torch.stack([torch.unsqueeze(model._forward_transform(model._stft(ch.to(device))), 0) for ch in y])
    Y = pad_spec(Y, mode=pad_mode)

    # Select sampler based on SDE type and sampler type
    if model.sde.__class__.__name__ == 'OUVESDE':
        if sampler_type == 'pc':
            sampler = model.get_pc_sampler('reverse_diffusion', corrector, Y.to(device), N=N, corrector_steps=corrector_steps, snr=snr)
        elif sampler_type == 'ode':
            sampler = model.get_ode_sampler(Y.to(device), N=N)
        else:
            raise ValueError(f"Sampler type {sampler_type} not supported")
    elif model.sde.__class__.__name__ == 'SBVESDE':
        sampler_type = 'ode' if sampler_type == 'pc' else sampler_type
        sampler = model.get_sb_sampler(sde=model.sde, y=Y.cuda(), sampler_type=sampler_type)
    else:
        raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")

    # Enhance each audio channel in the batch
    enhanced_audio = []
    for ch in range(y.size(0)):
        sample, _ = sampler()
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor[ch].to(device)  # Renormalize
        enhanced_audio.append(x_hat)

    # Stack the batch back together
    enhanced_audio = torch.stack(enhanced_audio)

    return enhanced_audio, target_sr


def enhance_audio(model, y, sr, device="cuda", sampler_type="pc", corrector="ald", corrector_steps=1, snr=0.5, N=30, t_eps=0.03):
    """
    Enhances a batch of audio data using the provided model.

    Parameters:
        model (ScoreModel): The preloaded score model for enhancement.
        y (torch.Tensor): Tensor containing the audio data with shape (batch_size, num_samples).
        sr (int): Sample rate of the input audio.
        device (str): Device to perform inference on (e.g., "cuda" or "cpu").
        sampler_type (str): Sampler type for the PC sampler.
        corrector (str): Corrector class for the PC sampler.
        corrector_steps (int): Number of corrector steps.
        snr (float): Signal-to-noise ratio value for (annealed) Langevin dynamics.
        N (int): Number of reverse steps.
        t_eps (float): Minimum process time.

    Returns:
        torch.Tensor: Enhanced audio with the same shape as the input.
    """
    # Set model parameters
    model.t_eps = t_eps
    model.eval()

    # Set the target sample rate and padding mode based on the model backbone
    if model.backbone == 'ncsnpp_48k':
        target_sr = 48000
        pad_mode = "reflection"
    elif model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    # print("------1-----------")
    # Resample if necessary
    if sr != target_sr:
        y = torch.tensor([resample(ch.detach().cpu().numpy(), orig_sr=sr, target_sr=target_sr) for ch in y])

    # print("---------2--------")
    T_orig = y.size(1)

    # Normalize
    norm_factor = y.abs().max(dim=1, keepdim=True).values
    y = y / norm_factor

    # Prepare DNN input
    Y = torch.stack([torch.unsqueeze(model._forward_transform(model._stft(ch.to(device))), 0) for ch in y])
    Y = pad_spec(Y, mode=pad_mode)
    # print("----------3-------")

    # Select sampler based on SDE type and sampler type
    if model.sde.__class__.__name__ == 'OUVESDE':
        if sampler_type == 'pc':
            sampler = model.get_pc_sampler('reverse_diffusion', corrector, Y.to(device), N=N, corrector_steps=corrector_steps, snr=snr)
        elif sampler_type == 'ode':
            sampler = model.get_ode_sampler(Y.to(device), N=N)
        else:
            raise ValueError(f"Sampler type {sampler_type} not supported")
    elif model.sde.__class__.__name__ == 'SBVESDE':
        sampler_type = 'ode' if sampler_type == 'pc' else sampler_type
        sampler = model.get_sb_sampler(sde=model.sde, y=Y.cuda(), sampler_type=sampler_type)
    else:
        raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")

    # print("-----------4------")
    # Enhance each audio channel in the batch
    # enhanced_audio = []
    # for ch in range(y.size(0)):
    #     sample, _ = sampler()
    #     x_hat = model.to_audio(sample.squeeze(), T_orig)
    #     x_hat = x_hat * norm_factor[ch].to(device)  # Renormalize
    #     enhanced_audio.append(x_hat)

    # print("------------5-----")
    # # Stack the batch back together
    # y_hat = torch.stack(enhanced_audio)

    # return y_hat
    # Enhance the entire batch at once
    samples, _ = sampler()
    x_hat = model.to_audio(samples.squeeze(), T_orig)  # Assuming model.to_audio handles batch processing
    x_hat = x_hat * norm_factor.to(device)  # Renormalize
    enhanced_audio = x_hat

    return enhanced_audio

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("--ckpt", type=str,default="/home/ahsan/Downloads/train_wsj0_2cta4cov_epoch=159.ckpt", required=False, help='Path to model checkpoint')
#     parser.add_argument("--output_dir", type=str, required=False, help='Directory to save the enhanced files')
#     parser.add_argument("--sampler_type", type=str, default="pc", help="Sampler type for the PC sampler.")
#     parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
#     parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
#     parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics")
#     parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
#     parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
#     parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
#     args = parser.parse_args()

#     # Example batch input (y and sr need to be defined elsewhere or loaded as required)
#     y = torch.randn(2, 306688)  # Example tensor for batch size of 2
#     sr = 48000  # Sample rate

#     # Enhance the batch of audio
#     enhanced_audio, sample_rate = enhance_audio_batch(
#         y=y,
#         sr=sr,
#         ckpt=args.ckpt,
#         device=args.device,
#         sampler_type=args.sampler_type,
#         corrector=args.corrector,
#         corrector_steps=args.corrector_steps,
#         snr=args.snr,
#         N=args.N,
#         t_eps=args.t_eps
#     )

#     print(enhanced_audio.shape)
#     print(sample_rate)
    # # Save each enhanced audio file
    # for i, enhanced in enumerate(enhanced_audio):
    #     output_path = join(args.output_dir, f"enhanced_{i}.wav")
    #     makedirs(dirname(output_path), exist_ok=True)
    #     write(output_path, sample_rate, enhanced.cpu().numpy())
