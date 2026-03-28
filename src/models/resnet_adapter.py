import torch
import torch.nn as nn
import torchvision.models as models
from .adapters import ConvParallelAdapter

class ResNet50WithConvParallelAdapters(nn.Module):
    
    def __init__(self, num_classes=102):
        
        super().__init__()

        # Loads ImageNet-pretrained weights
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")

     
        self.adapter1 = ConvParallelAdapter(256, rank=16)
        self.adapter2 = ConvParallelAdapter(512, rank=32)
        self.adapter3 = ConvParallelAdapter(1024, rank=32)
        self.adapter4 = ConvParallelAdapter(2048, rank=64)  

        # Adapts model to Oxford-102 Flowers dataset
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 
            num_classes
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        
        x = self.backbone.conv1(x) 
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  
        
        # ResNet Stage 1 + parallel adapter
        out = self.backbone.layer1(x)
        x = out + self.adapter1(out) # Parallel residual adaptation
        
        # ResNet Stage 2 + parallel adapter 
        out = self.backbone.layer2(x)
        x = out + self.adapter2(out)

        #  ResNet Stage 3 + parallel adapter
        out = self.backbone.layer3(x)
        x = out + self.adapter3(out)
 
        # ResNet Stage 4 + parallel adapter 
        out = self.backbone.layer4(x)
        x = out + self.adapter4(out)
        
        # Global Average Pooling + Classifier 
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x) # Final logits
       
        return x