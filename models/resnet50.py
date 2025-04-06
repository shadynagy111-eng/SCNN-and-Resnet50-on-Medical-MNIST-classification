import torch

import torch.nn as nn
import torchvision.models as models

class ResNet50MedicalMNIST(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50MedicalMNIST, self).__init__()
        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer to match the number of classes
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

# Example usage
if __name__ == "__main__":
    num_classes = 6 
    model = ResNet50MedicalMNIST(num_classes)
    print(model)

    # Example input tensor (batch_size=1, channels=1, height=64, width=64)
    input_tensor = torch.randn(1, 1, 64, 64)
    output = model(input_tensor)
    print(output)