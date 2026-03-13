"""
Vehicle Classification Project
ALPR Extension avec détection et classification de véhicules
"""

__version__ = "1.0.0"
__author__ = "Mickael - JEDHA Final Project"

from .dataset import (
    parse_stanford_annotations,
    prepare_yolo_dataset,
    VehicleClassificationDataset,
    get_transforms
)

from .model import (
    VehicleClassifier,
    load_classifier
)

from .train import (
    train_one_epoch,
    validate,
    train_vehicle_classifier
)

# from .inference import (
#     ALPRVehicleSystem,
#     validate_vehicle_plate_coupling
# )

__all__ = [
    # Dataset
    'parse_stanford_annotations',
    'prepare_yolo_dataset',
    'VehicleClassificationDataset',
    'get_transforms',
    
    # Model
    'VehicleClassifier',
    'load_classifier',
    
    # Training
    'train_one_epoch',
    'validate',
    'train_vehicle_classifier',
    
    # # Inference
    # 'ALPRVehicleSystem',
    # 'validate_vehicle_plate_coupling',
]
