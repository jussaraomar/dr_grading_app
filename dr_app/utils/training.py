# utils/training.py
import torch
import time
import numpy as np
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # progress bar 
    pbar = tqdm(dataloader, desc=" Training", unit="batch", 
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
   
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch with clean progress bar"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    

    pbar = tqdm(dataloader, desc=" Validating", unit="batch",
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def calculate_kappa(model, dataloader, device):
    """Calculate quadratic weighted kappa with clean progress bar"""
    model.eval()
    all_preds = []
    all_labels = []
    

    pbar = tqdm(dataloader, desc=" Calculating Kappa", unit="batch",
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return cohen_kappa_score(all_labels, all_preds, weights='quadratic')
