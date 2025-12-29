# utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm


from dr_app.config.paths import paths
from dr_app.config.params import ModelParams

class DRDatasetCached(Dataset):



    def __init__(self, dataframe, images_dir, transform=None, cache_file=None, dataset_type="train"):
        self.dataframe = dataframe
        self.transform = transform
        self.images_dir = images_dir
        self.dataset_type = dataset_type
        self.skipped = []
        self.loaded_count = 0
        
        # Generate cache 
        if cache_file is None:
            cache_file = f"cache_{dataset_type}_{len(dataframe)}.pkl"
        
        
        self.cache_file = os.path.join(paths.cache_dir, cache_file)
        
        print(f" Initializing {dataset_type} cached dataset...")
        print(f" Cache file: {self.cache_file}")
        print(f" Dataset size: {len(dataframe)} images")
        
        # try to load cache
        if os.path.exists(self.cache_file):
            print(" Loading existing cache...")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.image_cache = cache_data['images']
                self.labels = cache_data['labels']
            print(f" Loaded {len(self.image_cache)} images from cache")
        else:
            print(" Creating new cache (this may take a few minutes)...")
            self.image_cache = {}
            self.labels = []
            
            # progress bar
            pbar = tqdm(total=len(dataframe), desc=f" Caching {dataset_type} images")
            
            for idx in range(len(dataframe)):
                img_name = dataframe.iloc[idx]['id_code']
                label = dataframe.iloc[idx]['diagnosis']
                
                # Find and load image
                try:
                    img_path = self._find_image_path(img_name)
                    image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    
                    self.skipped.append((img_name, str(e)))
                    pbar.update(1)
                    continue
                
                self.loaded_count += 1

                # Resize
                image = image.resize(ModelParams.IMAGE_SIZE)
                
                # Convert to numpy array and store
                self.image_cache[idx] = np.array(image, dtype=np.uint8)
                self.labels.append(label)
                
                pbar.update(1)
                if idx % 100 == 0:
                    pbar.set_postfix({'Current': img_name})
            
            pbar.close()
            
            # Save cache
            print(" Saving cache to disk...")
            cache_data = {'images': self.image_cache, 'labels': self.labels}
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f" Cache created with {len(self.image_cache)} images")

            print(f" Cached images: {self.loaded_count} / {len(dataframe)}")
            if self.skipped:
                print(f" Skipped {len(self.skipped)} file(s). Examples:")
                for pair in self.skipped[:10]:
                    print("   -", pair)

    
    def _find_image_path(self, img_name):
        paths = [
            os.path.join(self.images_dir, img_name + '.png'),
            os.path.join(self.images_dir, img_name + '.jpeg'), 
            os.path.join(self.images_dir, img_name + '.jpg')
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image not found: {img_name}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get image array from cache and convert back to PIL
        image_array = self.image_cache[idx]
        image = Image.fromarray(image_array)
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms(image_size=ModelParams.IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(train_df, val_df, test_df, images_dir, batch_size=32, image_size=ModelParams.IMAGE_SIZE, use_cache=True):
    
    train_transform, val_transform = get_data_transforms(image_size)
    
    if use_cache:
        print("Using cached datasets for faster training")
        print("=" * 50)
        
        train_dataset = DRDatasetCached(train_df, images_dir, transform=train_transform, 
                                      cache_file='cache_train.pkl', dataset_type="train")
        print("-" * 30)
        
        val_dataset = DRDatasetCached(val_df, images_dir, transform=val_transform, 
                                    cache_file='cache_val.pkl', dataset_type="val")
        print("-" * 30)
        
        test_dataset = DRDatasetCached(test_df, images_dir, transform=val_transform, 
                                     cache_file='cache_test.pkl', dataset_type="test")
        print("=" * 50)
        print("ALL CACHES READY!")
    else:
        # Non-cached version fallback
        train_dataset = DRDatasetCached(train_df, images_dir, transform=train_transform, dataset_type="train")
        val_dataset = DRDatasetCached(val_df, images_dir, transform=val_transform, dataset_type="val")
        test_dataset = DRDatasetCached(test_df, images_dir, transform=val_transform, dataset_type="test")
    

    labels = np.array(train_dataset.labels)
    class_sample_counts = np.bincount(labels)
    class_weights = 1.0 / class_sample_counts
    sample_weights = class_weights[labels] 
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader( train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("[Dataset sizes] train:", len(train_dataset),
      "val:", len(val_dataset), "test:", len(test_dataset))
    
    assert len(test_dataset) == len(test_df), \
    f"Test dataset size {len(test_dataset)} != CSV {len(test_df)}"

    
    return train_loader, val_loader, test_loader

def get_class_weights(dataframe):
    """Calculate class weights for imbalanced dataset"""
    class_counts = dataframe['diagnosis'].value_counts().sort_index()
    total_samples = len(dataframe)
    num_classes = len(class_counts)
    
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} samples (weight: {class_weights[i]:.2f})")
    
    return class_weights_tensor

