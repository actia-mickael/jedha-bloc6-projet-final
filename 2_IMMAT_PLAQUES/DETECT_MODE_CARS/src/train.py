"""
Module entraînement pour classification véhicules
Fine-tuning progressif en 2 phases
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple
from tqdm.auto import tqdm
import json


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Entraînement pour une epoch
    
    Returns:
        epoch_loss: Loss moyenne
        epoch_acc: Accuracy moyenne
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Métriques
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validation du modèle
    
    Returns:
        epoch_loss: Loss moyenne
        epoch_acc: Accuracy moyenne
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_vehicle_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    save_dir: Path
):
    """
    Entraînement complet en 2 phases:
    - Phase 1: Backbone gelé (head uniquement)
    - Phase 2: Backbone dégelé (fine-tuning complet)
    
    Args:
        model: Modèle VehicleClassifier
        train_loader: DataLoader train
        val_loader: DataLoader validation
        config: Configuration entraînement
        device: Device (cuda/cpu)
        save_dir: Dossier sauvegarde modèles
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Critère
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': []
    }
    
    # ========== PHASE 1: Backbone gelé ==========
    print("\n" + "="*60)
    print("🔒 PHASE 1: Entraînement Classification Head (Backbone gelé)")
    print("="*60)
    
    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate_phase1'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    for epoch in range(config['phase1_epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{config['phase1_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Sauvegarde meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'phase': 1
            }, save_dir / 'best_model_phase1.pth')
            print(f"  ✅ Nouveau meilleur modèle sauvegardé (Val Acc: {val_acc:.2f}%)")
    
    # ========== PHASE 2: Backbone dégelé ==========
    print("\n" + "="*60)
    print("🔓 PHASE 2: Fine-tuning complet (Backbone dégelé)")
    print("="*60)
    
    model.unfreeze_backbone()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate_phase2'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    patience_counter = 0
    
    for epoch in range(config['phase2_epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{config['phase2_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Sauvegarde meilleur modèle
        if val_acc > best_val_acc + config['min_delta']:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'phase': 2
            }, save_dir / 'best_model_final.pth')
            print(f"  ✅ Nouveau meilleur modèle sauvegardé (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⛔ Early stopping déclenché après {epoch+1} epochs")
            break
    
    # Sauvegarde historique
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Entraînement terminé")
    print(f"🏆 Meilleure Val Accuracy: {best_val_acc:.2f}%")
    
    return model, history
