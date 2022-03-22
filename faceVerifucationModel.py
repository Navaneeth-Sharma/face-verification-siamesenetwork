import torch
import torch.nn as nn
from torchvision.models import vgg11
# from torchsummary import summary


class SiameseModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseModel, self).__init__()
        self.features_extracter = vgg11(pretrained).features[:12]
        self.fc_layer = nn.Sequential(
                                      nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                                      nn.Flatten(),

                                      nn.Linear(in_features=6272, out_features=512, bias=False),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),

                                      nn.Linear(in_features=512, out_features=64, bias=False),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(64),

                                      nn.Linear(in_features=64, out_features=32),
                                  )
        
    def forward_once(self, x):
        x = self.features_extracter(x)
        x = self.fc_layer(x)
        return x

    def forward(self, x1, x2):

        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)

        return x1, x2
        


class SiameseModel2(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseModel2, self).__init__()
        self.features_extracter = vgg11(pretrained).features[:12]
        self.fc_layer = nn.Sequential(
                                      nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                                      nn.Flatten(),

                                      nn.Linear(in_features=6272, out_features=512, bias=False),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),

                                      nn.Linear(in_features=512, out_features=64, bias=False),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(64),

                                      nn.Linear(in_features=64, out_features=8),
                                  )
        
    def forward_once(self, x):
        x = self.features_extracter(x)
        x = self.fc_layer(x)
        return x

    def forward(self, x1, x2):

        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)

        return x1, x2
        

if __name__ == "__main__":
    pass
    # model = FaceNetModel()
    # print(summary(model, [(3, 224, 224), (3, 224, 224)]))