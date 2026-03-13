"""
Modèle EfficientNet pour classification véhicules
Architecture avec classification head personnalisable
"""

import torch
import torch.nn as nn
import timm


class VehicleClassifier(nn.Module):
    """
    EfficientNet-B4 pour classification marque/modèle véhicule
    """
    def __init__(
        self, 
        num_classes: int, 
        pretrained: bool = True, 
        dropout: float = 0.3,
        model_name: str = 'efficientnet_b4'
    ):
        """
        Args:
            num_classes: Nombre de classes (marques/modèles)
            pretrained: Charger poids ImageNet
            dropout: Taux dropout pour régularisation
            model_name: Nom modèle timm (efficientnet_b4, efficientnet_b3, etc.)
        """
        super(VehicleClassifier, self).__init__()
        
        # Charger EfficientNet pré-entraîné
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Retirer classification head
            global_pool='avg'
        )
        
        # Feature dimension
        num_features = self.backbone.num_features  # 1792 pour B4, 1536 pour B3
        
        # Classification head custom
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
        self.model_name = model_name
        
    def forward(self, x):
        """Forward pass"""
        # Extract features
        features = self.backbone(x)
        # Classification
        out = self.classifier(features)
        return out
    
    def freeze_backbone(self):
        """Gèle les poids du backbone pour fine-tuning progressif"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone gelé")
    
    def unfreeze_backbone(self):
        """Dégèle backbone pour fine-tuning complet"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone dégelé")
    
    def get_num_trainable_params(self) -> int:
        """Retourne nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Retourne nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())


def load_classifier(
    checkpoint_path: str,
    num_classes: int,
    device: str = 'cuda',
    model_name: str = 'efficientnet_b4'
) -> VehicleClassifier:
    """
    Charge modèle depuis checkpoint
    
    Args:
        checkpoint_path: Chemin vers fichier .pth
        num_classes: Nombre de classes
        device: Device (cuda/cpu)
        model_name: Architecture modèle
        
    Returns:
        model: Modèle chargé en mode eval
    """
    model = VehicleClassifier(
        num_classes=num_classes,
        pretrained=False,
        model_name=model_name
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modèle chargé depuis {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"📊 Val Accuracy checkpoint: {checkpoint['val_acc']:.2f}%")
    
    return model
