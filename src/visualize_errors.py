import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import Net
import glob

def load_best_model(device):
    # Find the latest best model
    try:
        model_path = max(glob.glob('mnist_model_best_*.pth'))
        print(f"Loading model from {model_path}")
        model = Net().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
        return model
    except ValueError:
        raise Exception("No model file found!")

def get_misclassified_samples(model, test_loader, device, num_samples=20):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    predicted_labels = []
    probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob = torch.exp(output)  # Convert log probabilities to probabilities
            pred = output.argmax(dim=1)
            
            # Find misclassified samples
            incorrect_mask = pred.ne(target)
            if incorrect_mask.any():
                misclassified_images.extend(data[incorrect_mask].cpu().numpy())
                misclassified_labels.extend(target[incorrect_mask].cpu().numpy())
                predicted_labels.extend(pred[incorrect_mask].cpu().numpy())
                probabilities.extend(prob[incorrect_mask].cpu().numpy())
                
                if len(misclassified_images) >= num_samples:
                    break
    
    return (misclassified_images[:num_samples], 
            misclassified_labels[:num_samples], 
            predicted_labels[:num_samples],
            probabilities[:num_samples])

def plot_misclassified(images, true_labels, pred_labels, probs, num_samples=20):
    fig = plt.figure(figsize=(15, 10))
    for idx in range(min(num_samples, len(images))):
        ax = fig.add_subplot(4, 5, idx + 1, xticks=[], yticks=[])
        img = images[idx][0]  # Get the first channel
        ax.imshow(img, cmap='gray')
        
        # Get top 3 predictions and their probabilities
        top3_prob, top3_pred = torch.tensor(probs[idx]).topk(3)
        title = f'True: {true_labels[idx]}\nPred: {pred_labels[idx]}\n'
        title += f'P(pred)={top3_prob[0]:.2f}\n'
        title += f'2nd: {top3_pred[1]}({top3_prob[1]:.2f})\n'
        title += f'3rd: {top3_pred[2]}({top3_prob[2]:.2f})'
        
        ax.set_title(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('misclassified_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = load_best_model(device)
    
    # Prepare test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Get misclassified samples
    misclassified = get_misclassified_samples(model, test_loader, device)
    
    # Plot results
    plot_misclassified(*misclassified)

if __name__ == '__main__':
    main() 