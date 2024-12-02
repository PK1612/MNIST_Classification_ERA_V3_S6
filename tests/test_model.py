import pytest
import torch
from src.model import Net

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def model(device):
    return Net().to(device)

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