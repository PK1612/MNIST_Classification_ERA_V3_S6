import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from model import Net
from datetime import datetime
import os
import json
import math

def create_rotated_dataset(original_dataset, angles=[5, -5, 10, -10]):
    """Create additional datasets with rotated images"""
    rotated_datasets = []
    for angle in angles:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(angle, angle)),  # Fixed angle rotation
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        rotated_datasets.append(dataset)
    return rotated_datasets

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Base transformations for original dataset
    base_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create original and augmented datasets
    original_dataset = datasets.MNIST('../data', train=True, download=True, transform=base_transform)
    rotated_datasets = create_rotated_dataset(original_dataset)
    
    # Combine all datasets
    all_datasets = [original_dataset] + rotated_datasets
    combined_dataset = ConcatDataset(all_datasets)
    
    # Split combined dataset
    train_size = 50000 * (len(all_datasets))
    train_dataset, _ = torch.utils.data.random_split(
        combined_dataset, 
        [train_size, len(combined_dataset) - train_size]
    )
    
    # Test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=test_transform)
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize model
    model = Net().to(device)
    
    # Optimizer setup with SGD + Momentum
    optimizer = optim.SGD(model.parameters(), 
                         lr=0.015,  # Slightly higher initial learning rate
                         momentum=0.9,
                         nesterov=True,
                         weight_decay=1e-4)
    
    # Cosine Annealing learning rate scheduler with warmup
    def warmup_cosine_schedule(epoch):
        warmup_epochs = 3  # Reduced warmup period
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (20 - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    # Training loop
    best_accuracy = 0
    best_model_path = None
    training_history = []
    patience = 4  # Reduced patience for early stopping
    no_improvement_count = 0
    
    for epoch in range(1, 21):  # Maximum 20 epochs
        print(f"\nEpoch {epoch}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Training
        train(model, device, train_loader, optimizer, epoch)
        
        # Testing
        accuracy = test(model, device, test_loader)
        
        # Save training history
        training_history.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'lr': current_lr
        })
        
        # Save model if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_count = 0
            
            # Remove previous best model
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            # Save new best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f'mnist_model_best_{timestamp}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, best_model_path)
            print(f"New best model saved with accuracy: {accuracy:.2f}%")
        else:
            no_improvement_count += 1
        
        # Early stopping check with reduced patience
        if no_improvement_count >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break
        
        # Step the scheduler
        scheduler.step()
        
        print(f"Best accuracy so far: {best_accuracy:.2f}%")
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=4)
    
    print(f"\nTraining completed. Best accuracy: {best_accuracy:.2f}%")
    print(f"Best model saved as: {best_model_path}")

if __name__ == '__main__':
    main()