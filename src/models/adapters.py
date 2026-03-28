import torch.nn as nn
import torch
class ConvParallelAdapter(nn.Module):
    
    def __init__(self, channels, rank=32):
        super().__init__()
        
        self.adapter = nn.Sequential(
          
            nn.Conv2d(channels, rank, kernel_size=1, bias=False),
            nn.ReLU(inplace=True), 
      
            nn.Conv2d(rank, rank, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True), 

            nn.Conv2d(rank, channels, kernel_size=1, bias=True)
        )
      
        self.alpha = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
    def forward(self, x):
        return self.alpha * self.adapter(x)
