import unittest
import torch
import os
import numpy as np
import pickle
from pathlib import Path
from train_pkl_file import *
from pkl_file_models import *   



class TestSpectralAutoEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        # Define directories
        cls.test_clean_dir = Path("./test_data/complex/complex_specs_S02_P08/Test")
        cls.test_noisy_dirs = [
            Path("./test_data/complex/complex_specs_S02_P08_U02.CH3/Test"),
            Path("./test_data/complex/complex_specs_S02_P08_U03.CH3/Test"),
            Path("./test_data/complex/complex_specs_S02_P08_U04.CH3/Test"),
            Path("./test_data/complex/complex_specs_S02_P08_U05.CH3/Test")
        ]
        cls.model_path = "./models/SpecResE4D1/model_epoch_100.pth"

        # Load test data
        cls.test_dataset = SpectralDataset(
            clean_data_dir="./test_data/complex/complex_specs_S02_P08",
            noisy_data_dir="./test_data/complex",
            file_type='Test',
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load trained model
        cls.model = SpectralResE4D1(z_dim1=32, z_dim2=32, z_dim3=32, z_dim4=32, n_res_blocks=3)
        checkpoint = torch.load(cls.model_path, map_location=torch.device("cpu"))
        cls.model.load_state_dict(checkpoint['model_state_dict'])
        cls.model.eval()

    def test_model_performance(self):
        """Test the model's performance on the test dataset"""
        total_mse_loss = 0
        total_nuc_loss = 0
        total_cos_loss = 0
        total_spec_loss = 0
        total_spec_snr = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_dataset):
                clean_audio = data["clean_audio"]
                noisy_audio_1 = data["noisy_audio_1"]
                noisy_audio_2 = data["noisy_audio_2"]
                noisy_audio_3 = data["noisy_audio_3"]
                noisy_audio_4 = data["noisy_audio_4"]

                decoded, mse_loss, nuc_loss, _, cos_loss, spec_loss, spec_loss_dict, spec_snr, _ = self.model(
                    noisy_audio_1, 
                    noisy_audio_2, 
                    noisy_audio_3, 
                    noisy_audio_4, 
                    clean_audio,
                    True,
                )

                total_mse_loss += mse_loss.item()
                total_nuc_loss += nuc_loss.item()
                total_cos_loss += cos_loss.item()
                total_spec_loss += spec_loss.item()
                total_spec_snr += spec_snr.item()

        avg_mse_loss = total_mse_loss / len(self.test_dataset)
        avg_nuc_loss = total_nuc_loss / len(self.test_dataset)
        avg_cos_loss = total_cos_loss / len(self.test_dataset)
        avg_spec_loss = total_spec_loss / len(self.test_dataset)
        avg_spec_snr = total_spec_snr / len(self.test_dataset)

        print(f"Average MSE Loss: {avg_mse_loss:.4f}")
        print(f"Average Nuclear Loss: {avg_nuc_loss:.4f}")
        print(f"Average Cosine Loss: {avg_cos_loss:.4f}")
        print(f"Average Spectral Loss: {avg_spec_loss:.4f}")
        print(f"Average Spectral SNR: {avg_spec_snr:.2f} dB")

        # Add your own assertions based on expected performance
        self.assertLess(avg_mse_loss, 0.1)
        self.assertLess(avg_nuc_loss, 0.5)
        self.assertLess(avg_cos_loss, 0.2)
        self.assertGreater(avg_spec_snr, 10.0)

def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()