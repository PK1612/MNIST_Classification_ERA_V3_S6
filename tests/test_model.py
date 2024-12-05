"""Test module for MNIST model."""
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import Net
from utils import evaluate_model

@pytest.fixture(scope="session")
def device():
    """Fixture for PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def model(device):
    """Fixture for model instance."""
    model = Net().to(device)
    # If there's a trained model, load it
    model_files = [f for f in os.listdir('.') if f.startswith('mnist_model_best_') and f.endswith('.pth')]
    if model_files:
        latest_model = max(model_files)  # Get the most recent model
        checkpoint = torch.load(latest_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

@pytest.fixture(scope="session")
def test_loader():
    """Fixture for test data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=1000, shuffle=False)

def test_parameter_count(model):
    """Test the number of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

def test_input_output_shape(model, device):
    """Test model's input-output shapes."""
    x = torch.randn(1, 1, 28, 28).to(device)
    output = model(x)
    print(f"\nOutput shape: {output.shape}")
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_forward(model, device):
    """Test model's forward pass."""
    x = torch.randn(1, 1, 28, 28).to(device)
    output = model(x)
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains infinite values"

def test_accuracy(model, device, test_loader):
    """Test model's accuracy."""
    accuracy = evaluate_model(model, device, test_loader)
    print(f"\nModel Accuracy: {accuracy:.2f}%")
    assert accuracy >= 99.4, f"Model accuracy {accuracy:.2f}% is below target of 99.4%"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])