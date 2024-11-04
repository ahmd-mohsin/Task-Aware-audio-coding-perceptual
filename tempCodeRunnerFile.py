class TestSpectralAutoEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        # Define directories
        cls.test_clean_dir = Path("./test_data/complex/complex_specs_S02_P08/Test")
        cls.test_noisy_dirs = [
            Path(".Data/complex/complex_specs_S02_P08_U02.CH3/Test"),
            Path(".Data/complex/complex_specs_S02_P08_U03.CH3/Test"),
            Path(".Data/complex/complex_specs_S02_P08_U04.CH3/Test"),
            Path(".Data/complex/complex_specs_S02_P08_U05.CH3/Test")
        ]
        cls.model_path = "./models/SpecResE4D1/model_epoch_100.pth"

        # Load test data
        cls.test_dataset = SpectralDataset(
            clean_data_dir=".Data/complex/complex_specs_S02_P08/Test",
            noisy_data_dir=".Data/complex",
            file_type='Test',
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
