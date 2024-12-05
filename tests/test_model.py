"""Test module for MNIST model."""

import pytest
import torch
import os
import sys

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.model import Net
from torchsummary import summary

@pytest.fixture
def model():
    """Fixture to create and return model instance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Net().to(device)


def test_parameter_count(model):
    """Test that model has less than 20k parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params < 20000
    ), f"Model has {total_params} parameters, which exceeds limit of 20000"

def test_has_batch_norm(model):
    """Test that model contains batch normalization layers"""
    has_batch_norm = any(
        isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()
    )
    assert has_batch_norm, "Model does not contain batch normalization layers"

def test_has_dropout(model):
    """Test that model contains dropout layers"""
    has_dropout = any(
        isinstance(module, torch.nn.Dropout) for module in model.modules()
    )
    assert has_dropout, "Model does not contain dropout layers"

def test_has_fully_connected(model):
    """Test that model contains fully connected (Linear) layers"""
    has_linear = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    assert has_linear, "Model does not contain fully connected layers"