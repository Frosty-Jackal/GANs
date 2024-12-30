import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

os.makedirs("./images/gan/",exist_ok=True)
os.makedirs("./save/gan/",exist_ok=True)
os.makedirs("./datasets/mnist",exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")

opt = parser.parse_args()
print(opt)

img_shape=(opt.channels , opt.img_size, opt.img_size)
img_area = np.prod(img_shape)

device= torch.device('cuda') if torch.cuda.is_available() else 'cpu'

mnist=datasets.MNIST(
    root='./datasets/',train=True,download=False,transform=transforms.Compose(
    [transforms.Resize(opt.img_size),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])]
    ),
)

dataloader = DataLoader(
    mnist,
    batch_size=opt.batch_size,
    shuffle=True,
)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_area,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
    def forward(self,img):
        img_flat = img.view(img.size(0),-1)
        validity=self.model(img_flat)
        return validity

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        def block(in_feat,out_feat,normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.model=nn.Sequential(
            *block(opt.latent_dim,128,normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024,img_area),
            nn.Tanh()
        )
    def forward(self,z):
        imgs=self.model(z)
        imgs = imgs.view(imgs.size(0),*img_shape)
        return imgs
generator=Generator().to(device)
discriminator = Discriminator().to(device)
criterion = torch.nn.BCELoss().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))

for epoch in range(opt.n_epochs):
    for i,(imgs,_) in enumerate(dataloader):
        imgs = imgs.view(imgs.size(0),-1).to(device)
        real_img=Variable(imgs).to(device)
        real_label = Variable(torch.ones(imgs.size(0),1)).to(device)
        fake_label = Variable(torch.zeros(imgs.size(0),1)).to(device)
        
        real_out = discriminator(real_img)
        loss_real_D=criterion(real_out,real_label)
        real_scores = real_out

        z=Variable(torch.randn(imgs.size(0),opt.latent_dim)).to(device)
        fake_img = generator(z).detach()
        fake_out=discriminator(fake_img)
        loss_fake_D=criterion(fake_out,fake_label)
        fake_scores = fake_out
        loss_D=loss_real_D  + loss_fake_D
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        z=Variable(torch.randn(imgs.size(0),opt.latent_dim)).to(device)
        fake_img=generator(z)
        output = discriminator(fake_img)
        loss_G = criterion(output,real_label)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if (i+1) %100 ==0 :
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D real:%f] [D fake: %f]"
                % (epoch,opt.n_epochs,i,len(dataloader),loss_D.item(),loss_G.item(),real_scores.data.mean(),fake_scores.data.mean() )            
            )
        batches_done = epoch * len(dataloader)+ i
        if batches_done % opt.sample_interval==0:
            save_image(fake_img.data[:25],"./images/gan/%d.png"%batches_done,normalize=True)
torch.save(generator.state_dict,'./save/gan/generator.pth')
torch.save(discriminator.state_dict(),'./save/gan/discriminator.pth')