# dr/config/paths.py
import os
from pathlib import Path

class Paths:
    def __init__(self, base_path=None):
       
        if base_path is None:
         
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)
        
    
        self.data_dir = self.base_path / "data"
        self.images_dir = self.data_dir / "APTOS 2019" / "train_images"
        self.main_csv = self.data_dir / "APTOS 2019" / "train.csv"
        self.saved_models = self.base_path / "experiments" / "saved_models"
        self.results = self.base_path / "results"
        self.outputs = self.base_path / "outputs"
        self.cache_dir = self.base_path / "cache"
        
        # Create directories
        self.saved_models.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

paths = Paths()
