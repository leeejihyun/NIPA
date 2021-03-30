import torch
import torch.nn as nn
import torchvision.models as models


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            
            # 1 x 68 x 68
            
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            
            # 64 x 68 x 68
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
            
            # 128 x 68 x 68
        )
        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            
            # 128 x 34 x 34
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            
            # 128 x 34 x 34
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            
            # 256 x 34 x 34
            
            nn.MaxPool2d(2, stride=2, ceil_mode=False)
            
            # 256 x 17 x 17
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # 256 x 2 x 2
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout()
        )
        
        self.last_fc = nn.Sequential(nn.Linear(256, num_classes))

        self._initialize_weights()
        
        
    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)

        pool = self.avgpool(h)
        flatten = torch.flatten(pool, 1)
        classifier = self.classifier(flatten)
        classes = self.last_fc(classifier)

        return classes