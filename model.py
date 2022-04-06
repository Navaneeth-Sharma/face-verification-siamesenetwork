import torch.nn as nn
from torchvision.models import resnet34
# from torchsummary import summary


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.features_extracter = resnet34(pretrained=True)
        self.features_extracter.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(in_features=512, out_features=512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),

            nn.Linear(in_features=128, out_features=128),
        )

    def forward(self, x):
        x = self.features_extracter(x)
        return x


if __name__ == "__main__":
    # summary()
    pass
