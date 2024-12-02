import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import Net
from src.utils import evaluate_model

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def model(device):
    return Net().to(device)

@pytest.fixture
def test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=1000, shuffle=False)

def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

def test_input_output_shape(model, device):
    x = torch.randn(1, 1, 28, 28).to(device)
    output = model(x)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_forward(model, device):
    x = torch.randn(1, 1, 28, 28).to(device)
    output = model(x)
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains infinite values"

def test_accuracy(model, device, test_loader):
    accuracy = evaluate_model(model, device, test_loader)
    assert accuracy >= 99.4, f"Model accuracy {accuracy:.2f}% is below target of 99.4%"