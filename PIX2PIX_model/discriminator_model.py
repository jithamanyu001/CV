import torch 
from torch import nn

class CNNBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=2) -> None:
        super(CNNBlock,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,4,stride=stride,bias=False,padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self,in_ch=3,features=[64,128,256,512]) -> None:
        super(Discriminator,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_ch*2,features[0],4,stride=2,padding=1,bias=False,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        layers=[]
        in_ch=features[0]
        for i in features[1:]:
            layers.append(CNNBlock(in_ch,i,stride=1 if i==features[-1] else 2))
            in_ch=i
        layers.append(
            nn.Conv2d(in_ch,1,4,1,1,padding_mode='reflect')
        )
        self.model=nn.Sequential(*layers)
    def forward(self,x,y):
        x=torch.cat((x,y),dim=1)
        x=self.conv1(x)
        return self.model(x)


def test():
    x=torch.randn((1,3,256,256))
    y=torch.randn((1,3,256,256))
    model=Discriminator()
    print(model(x,y).shape)

if __name__ == "__main__":
    test()