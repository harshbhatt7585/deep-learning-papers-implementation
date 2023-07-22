"""
    The first is the all convolutional net (Springenberg et al., 2014) which replaces deterministic spatial
    pooling functions (such as maxpooling) with strided convolutions,

    Use batchnorm in both the generator and the discriminator. not applying 
    batchnorm to the generator output layer and the discriminator input layer.

    Use ReLU activation in generator for all layers except for the output, which uses Tanh.

    Use LeakyReLU activation in the discriminator for all layers.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transform
import torchvision.utils as vutils
import os


CUDA = True
DATA_DIR = './data/mnist'
OUT_PATH = 'output'
LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 25
REAL_LABEL = 1
FAKE_LABEL = 0

lr = 2e-4
seed = 1

CUDA = CUDA and torch.cuda.is_available()

if CUDA:
    print(f'CUDA version: {torch.version.cuda}')

np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)

cudnn.benchmark = True

dataset = dset.MNIST(root=DATA_DIR, download=True,
                     transform=transform.Compose([
                        transform.Resize(X_DIM),
                        transform.ToTensor(),
                        transform.Normalize((0.5,), (0.5,)) 
                     ]))


assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

device = torch.device('cuda:0' if CUDA else 'cpu')

def weight_init(m):
    """
        Custom weight initialization
    """                                        
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=(4,4), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=(4,4), stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        o = self.sequential(x)
        return o
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4,4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(512, 1, kernel_size=(4,4), stride=1, padding=0),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        o = self.sequential(x).squeeze(1)
        return o 

gen = Generator().to(device)
gen.apply(weight_init)
print(gen)

dis = Discriminator().to(device)
dis.apply(weight_init)
print(dis)

criterion = nn.BCELoss()

viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

optimizerD = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(EPOCH_NUM):
    for i, data in enumerate(dataloader):
        x_real = data[0].to(device)
        real_label = torch.full((x_real.size(0),), REAL_LABEL, device=device).float()
        fake_label = torch.full((x_real.size(0), ), FAKE_LABEL, device=device).float()

        # update dis with real data
        dis.zero_grad()
        y_real = dis(x_real)
        loss_dis_real = criterion(y_real, real_label)
        loss_dis_real.backward()

        # update dis with fake data
        z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
        # generate using generator
        x_fake = gen(z_noise)
        y_fake = dis(x_fake.detach())
        loss_dis_fake = criterion(y_fake, fake_label)
        loss_dis_fake.backward()
        optimizerD.step()

        # update gen using fake data
        gen.zero_grad()
        y_fake_r = dis(x_fake)
        loss_gen = criterion(y_fake_r, real_label)
        loss_gen.backward()
        optimizerG.step()

        if i % 100 == 0:
            print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                epoch, i, len(dataloader),
                loss_dis_real.mean().item(),
                loss_dis_fake.mean().item(),
                loss_gen.mean().item()
            ))
            
            vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
            with torch.no_grad():
                viz_sample = gen(viz_noise)
                vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_samples_{}.png'.format(epoch)), normalize=True)
        
        torch.save(gen.state_dict(), os.path.join(OUT_PATH, 'gen_{}.pth'.format(epoch)))
        torch.save(dis.state_dict(), os.path.join(OUT_PATH, 'dis_{}.pth').format(epoch))

