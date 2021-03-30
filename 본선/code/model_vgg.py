import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            
            # 1 x 64 x 64
            
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 64 x 64 x 64
            
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),  
            
            # 64 x 64 x 64
            
            nn.MaxPool2d(2,2),
            
            # 64 x 32 x 32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 128 x 32 x 32
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 128 x 32 x 32
            
            nn.MaxPool2d(2,2),
            
            # 128 x 16 x 16
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 256 x 16 x 16
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 256 x 16 x 16
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 256 x 16 x 16
            
            nn.MaxPool2d(2,2),
            
            # 256 x 8 x 8
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 512 x 8 x 8
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 512 x 8 x 8
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 512 x 8 x 8
            
            nn.MaxPool2d(2,2),
            
            # 512 x 4 x 4
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 512 x 4 x 4
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 512 x 4 x 4
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            
            # 512 x 4 x 4
            
            nn.MaxPool2d(2,2),
            
            # 512 x 2 x 2
        )
        
        self.avg_pool = nn.AvgPool2d(2)
        
        # 512 x 1 x 1
        
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        
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
        features = self.conv(x)
        x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        x = self.classifier(x)
        return x