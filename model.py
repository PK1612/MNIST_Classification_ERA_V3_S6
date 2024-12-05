import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block - reduced channels
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.dropout1 = nn.Dropout2d(0.1)
        
        # Second block - reduced channels
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.dropout2 = nn.Dropout2d(0.2)
        
        # Third block - maintain channels
        self.conv3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Fourth block - channel reduction
        self.conv4 = nn.Conv2d(24, 12, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(12)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # Fully connected layers - adjusted sizes
        self.fc1 = nn.Linear(12 * 3 * 3, 72)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(72, 10)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Fully connected layers
        x = x.view(-1, 12 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Test the model architecture
if __name__ == "__main__":
    model = Net()
    print(f"Model Architecture:")
    print(model)
    total_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    # Test with random input
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"\nOutput shape for input (1, 1, 28, 28): {output.shape}")
    
    print("\nParameter count by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,}")
    
    # Verify parameter count is within limit
    assert total_params < 20000, f"Model has {total_params} parameters, exceeding limit of 20,000"