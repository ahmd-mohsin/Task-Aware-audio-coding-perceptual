# Distributed Task-Aware Source Coding for Correlated Audio Signals with Perceptual Loss

This repository implements a distributed task-aware source coding framework focused on encoding and decoding correlated audio signals using perceptual loss. The main approach involves encoding audio files as spectrograms, using an autoencoder model for compact representation, and applying audio denoising for improved reconstruction.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [1. Audio Encoding and Denoising](#1-audio-encoding-and-denoising)
  - [2. Task-Aware Source Coding](#2-task-aware-source-coding)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project leverages task-aware source coding techniques to efficiently encode audio signals with high correlation. The process includes converting audio into spectrograms, applying perceptual loss for enhanced quality, and using an autoencoder to achieve low-dimensional representations of audio data, allowing for more efficient storage and transmission. Additionally, audio denoising is performed post-decoding to enhance clarity and quality.

## Repository Structure

```plaintext
.
├── data/                               # Folder for input audio files
├── data_loaders/                       # Dataloading pipeline
├── dtac/                               # Core distributed task-aware coding functions
├── summary/                            # Summary files and experiment logs
├── audio_DAE.py                        # Script for audio denoising autoencoder
├── spectrogram.ipynb                   # Notebook for spectrogram conversion
├── testing_script.py                   # Script for encoder-decoder testing
├── testing_scripts.ipynb               # Notebook for model testing and evaluation
├── train_de_noising_audio.py           # Training script for denoising audio
├── train_de_noising_images.py          # Training script for denoising images
├── requirements.txt                    # Project dependencies
├── README.md                           # This readme file
└── LICENSE                             # License information
```

## Methodology

### 1. Audio Encoding and Denoising

- **Spectrogram Generation:** The `spectrogram.ipynb` notebook converts audio files to spectrograms for feature extraction.
- **Denoising Autoencoder:** The `audio_DAE.py` script and the `train_de_noising_audio.py` file work together to denoise audio, minimizing noise while retaining essential information.

### 2. Task-Aware Source Coding

The core coding pipeline (`dtac`) encodes correlated audio signals and leverages perceptual loss to ensure the reconstructed audio maintains high intelligibility. The process:
1. Encodes audio to spectrograms, with perceptual loss guiding model focus on important features.
2. Enhances decoded audio using speech enhancement techniques.

## Getting Started

### Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/ahmd-mohsin/distributed-task-aware-source-coding.git
cd distributed-task-aware-source-coding
pip install -r requirements.txt
```

### Usage

#### Preprocessing

Convert audio files into spectrograms for encoding:

```bash
python spectrogram.ipynb --input data/ --output data/spectrograms
```

#### Training

To train the denoising autoencoder, use:

```bash
python train_de_noising_audio.py --data data/spectrograms --output models/
```

#### Testing and Evaluation

Run the testing script to evaluate the autoencoder’s performance:

```bash
python testing_script.py --input data/spectrograms --output data/enhanced_audio
```

## Results

Generated audio and visual results are saved in the `data/` and `summary/` folders. Key files include:
- **Reconstructed Audio**: `reconstructed_audio.wav`
- **Visual Representations**: Images such as `original_clean_image.png`, `transformed_noisy_image_1.png`

## Contributing

Contributions to improve the functionality and efficiency of this project are welcome. Please submit issues or pull requests with improvements.

## License

This project is licensed under the MIT License.

--- 


