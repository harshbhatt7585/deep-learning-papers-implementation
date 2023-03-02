# relu function after every conv layer
# [optional] max pooling over local neighbourhood 
# [optional] a local contrast operation that normalizes the responses across the features maps
# final layer is softmax classier
# loss function - cross-entropy
# dropout in the 6th and 7th layer with p = 0.5
# momentum with value 0.9
# learning rate = 10e-2
# batch size = 128

# first conv layer with filter size = 7 x 7 and stride = 2



import torch
import torch.nn as nn


class ZFNet(nn.Module):
    def __init__(self):
        super().__init__(self, ZFNet)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, padding_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=1),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kenel_size=3, stride=2),
            nn.LocalResponseNorm(size=1),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=2, padding_size=1, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=13*13*256, out_features=4096),
            nn.Dropout(0.5),
            nn.Lienar(in_features=4096, out_features=4096),
            nn.Softmax() 
        )

    def init_parameters(self):
        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.contant_(layer.weight, 0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.01)
                nn.init.contant_(layer.bias, 0)

    def forward(self, x):
        x = self.convs(x)
        out = self.classifier(x)
        return out
    

