"""
Dataset utilities pour classification véhicules
Gestion Stanford Cars et conversions YOLO
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.io
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_stanford_annotations(annos_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse fichier .mat annotations Stanford Cars
    
    Args:
        annos_path: Chemin vers dossier contenant devkit
        
    Returns:
        df: DataFrame avec colonnes [image_name, class_id, class_name, bbox]
        class_names: Liste des noms de classes
    """
    mat = scipy.io.loadmat(annos_path / 'devkit' / 'cars_train_annos.mat')
    meta = scipy.io.loadmat(annos_path / 'devkit' / 'cars_meta.mat')
    
    # Extraction noms classes
    class_names = [name[0] for name in meta['class_names'][0]]
    
    # Extraction annotations
    annotations = []
    for annotation in mat['annotations'][0]:
        bbox = annotation['bbox'][0][0].tolist()
        class_id = int(annotation['class'][0][0]) - 1  # 0-indexed
        fname = annotation['fname'][0]
        
        annotations.append({
            'image_name': fname,
            'class_id': class_id,
            'class_name': class_names[class_id],
            'bbox': bbox,  # [x1, y1, x2, y2]
        })
    
    df = pd.DataFrame(annotations)
    print(f"✅ {len(df)} annotations parsées")
    print(f"📊 {df['class_id'].nunique()} classes uniques")
    
    return df, class_names


def convert_bbox_to_yolo(
    bbox: List[int], 
    img_width: int, 
    img_height: int
) -> List[float]:
    """
    Convertit bbox [x1, y1, x2, y2] en format YOLO [x_center, y_center, width, height]
    Valeurs normalisées entre 0 et 1
    """
    x1, y1, x2, y2 = bbox
    
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return [x_center, y_center, width, height]


def prepare_yolo_dataset(
    df: pd.DataFrame, 
    images_dir: Path, 
    output_dir: Path, 
    class_names: List[str],
    test_size: float = 0.15
) -> Tuple[Path, Path]:
    """
    Prépare dataset au format YOLO pour détection véhicule
    Classe unique: 'vehicle' (fusion toutes marques/modèles)
    
    Returns:
        yolo_dir: Chemin vers dataset YOLO
        yaml_path: Chemin vers fichier data.yaml
    """
    yolo_dir = output_dir / 'yolo_vehicle_detection'
    
    # Créer structure YOLO
    for split in ['train', 'val']:
        (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Split train/val
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['class_id'], 
        random_state=42
    )
    
    for split, split_df in [('train', train_df), ('val', val_df)]:
        print(f"\n📦 Préparation split '{split}': {len(split_df)} images")
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split}"):
            img_path = images_dir / row['image_name']
            if not img_path.exists():
                continue
            
            # Lire image pour dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Convertir bbox en YOLO format
            yolo_bbox = convert_bbox_to_yolo(row['bbox'], w, h)
            
            # Copier image
            dest_img = yolo_dir / split / 'images' / row['image_name']
            shutil.copy(img_path, dest_img)
            
            # Créer fichier label (classe 0 = vehicle)
            label_file = yolo_dir / split / 'labels' / row['image_name'].replace('.jpg', '.txt')
            with open(label_file, 'w') as f:
                f.write(f"0 {' '.join(map(str, yolo_bbox))}\n")
    
    # Créer fichier data.yaml
    yaml_content = f"""# YOLOv8 Vehicle Detection Dataset
path: {yolo_dir.absolute()}
train: train/images
val: val/images

# Classes
nc: 1
names: ['vehicle']
"""
    
    yaml_path = yolo_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset YOLO créé: {yolo_dir}")
    print(f"📄 Configuration: {yaml_path}")
    
    return yolo_dir, yaml_path


class VehicleClassificationDataset(Dataset):
    """
    Dataset PyTorch pour classification marque/modèle véhicule
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        images_dir: Path,
        class_to_idx: Dict[str, int],
        transform=None,
        use_bbox_crop: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_bbox_crop = use_bbox_crop
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Charger image
        img_path = self.images_dir / row['image_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop véhicule si bbox disponible
        if self.use_bbox_crop and 'bbox' in row:
            x1, y1, x2, y2 = map(int, row['bbox'])
            image = image[y1:y2, x1:x2]
        
        # Appliquer transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Label
        label = self.class_to_idx[row['class_name']]
        
        return image, label


def get_transforms(img_size: int = 224, is_train: bool = True):
    """
    Retourne transformations Albumentations pour classification
    
    Args:
        img_size: Taille image carrée
        is_train: Mode train (avec augmentations) ou validation
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=15, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
