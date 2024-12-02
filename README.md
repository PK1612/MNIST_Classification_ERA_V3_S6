# MNIST Digit Classification with CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The model achieves >99% accuracy while maintaining less than 20,000 parameters through efficient architecture design and training strategies.

## Model Architecture
The model uses a combination of convolutional layers, batch normalization, and dropout for effective feature extraction and regularization. The architecture progresses from local feature detection to global pattern recognition through multiple stages of processing.

### Architecture Overview
- Input: 28x28 grayscale images
- 4 Convolutional blocks with batch normalization and dropout
- Adaptive pooling to reduce spatial dimensions
- 2 Fully connected layers with dropout
- Output: 10 classes (digits 0-9)

### Layer-wise Details

| Layer | Input Size | Output Size | RF | Jump In | Jump Out | Parameters |
|-------|------------|-------------|-----|----------|-----------|------------|
| Input | 28x28x1 | 28x28x1 | 1 | 1 | 1 | 0 |
| Conv1 | 28x28x1 | 28x28x12 | 3 | 1 | 1 | 108 + 12 |
| BatchNorm1 | 28x28x12 | 28x28x12 | 3 | 1 | 1 | 24 |
| MaxPool1 | 28x28x12 | 14x14x12 | 4 | 1 | 2 | 0 |
| Conv2 | 14x14x12 | 14x14x24 | 8 | 2 | 2 | 2,592 + 24 |
| BatchNorm2 | 14x14x24 | 14x14x24 | 8 | 2 | 2 | 48 |
| MaxPool2 | 14x14x24 | 7x7x24 | 10 | 2 | 4 | 0 |
| Conv3 | 7x7x24 | 7x7x24 | 14 | 4 | 4 | 5,184 + 24 |
| BatchNorm3 | 7x7x24 | 7x7x24 | 14 | 4 | 4 | 48 |
| Conv4 | 7x7x24 | 7x7x12 | 14 | 4 | 4 | 288 + 12 |
| BatchNorm4 | 7x7x12 | 7x7x12 | 14 | 4 | 4 | 24 |
| AdaptivePool | 7x7x12 | 3x3x12 | 20 | 4 | - | 0 |
| FC1 | 108 | 72 | - | - | - | 7,848 |
| FC2 | 72 | 10 | - | - | - | 730 |

Total Parameters: 16,966

### Key Features
- Batch Normalization after each convolution
- Progressive dropout (0.1 to 0.4)
- Adaptive pooling for fixed output size
- Efficient parameter utilization

## Training Details

### Hyperparameters
- Optimizer: SGD with Nesterov momentum
- Initial Learning Rate: 0.015
- Momentum: 0.9
- Weight Decay: 1e-4
- Batch Size: 64
- Epochs: 20 (max)

### Data Augmentation
- Random rotation (±5°, ±10°)
- Random affine transforms
- Random scaling (0.9-1.1)

### Training Logs

Epoch 1
Learning rate: 0.005000
Train Loss: 0.2145 | Accuracy: 93.45%
Epoch 5
Learning rate: 0.003500
Train Loss: 0.0845 | Accuracy: 98.12%
Epoch 10
Learning rate: 0.001500
Train Loss: 0.0456 | Accuracy: 98.89%
Epoch 15
Learning rate: 0.000750
Train Loss: 0.0334 | Accuracy: 99.23%
Final Test Accuracy: 99.42%


## Results
- Test Accuracy: 99.42%
- Parameters: 16,966
- Training Time: ~15 minutes (GPU)

## Requirements
- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib

## Usage

bash
Train the model
python train.py
Test the model
python test_model.py
Visualize misclassified examples
python visualize_errors.py


## Model Performance Analysis
The model achieves high accuracy through:
1. Efficient feature extraction with progressive channel expansion
2. Strong regularization with dropout and batch normalization
3. Effective data augmentation
4. Carefully tuned learning rate schedule

## Future Improvements
1. Implement cross-validation
2. Explore knowledge distillation
3. Add model interpretability visualizations
4. Implement TensorBoard logging