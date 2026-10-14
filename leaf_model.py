import torch.nn as nn

class LeafCNN(nn.Module):
    """Compact convolutional classifier for labeled crop-leaf images."""
    def __init__(self, num_classes: int):
        super().__init__()
        def block(inp, out):
            return nn.Sequential(nn.Conv2d(inp, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3,32), block(32,64), block(64,128), block(128,256), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.35), nn.Linear(256,128), nn.ReLU(inplace=True), nn.Dropout(.2), nn.Linear(128,num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))
