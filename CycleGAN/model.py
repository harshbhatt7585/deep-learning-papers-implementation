import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image
import os
import random

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = 
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)

        return torch.autograd.Variable(torch.cat(to_return))


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        a = [os.path.join(os.path.join(data_dir, 'A'), data) for data in os.listdir(os.path.join(data_dir, 'A'))]
        b = [os.path.join(os.path.join(data_dir, 'B'), data) for data in os.listdir(os.path.join(data_dir, 'B'))]
        self.images = list(zip(a, b))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        imageA = Image.open(self.images[idx][0])
        imageB = Image.open(self.images[idx][1])

        if self.transform != None:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)
        
        return imageA, imageB


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.sequnetial = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        o = self.sequnetial(x)
        o += x
        return o
    

class Generator(nn.Module):
    def __init__(self, num_res_blocks=9):
        super(Generator, self).__init__()
        layers = []
        
        # intitial conv blocks
        self.initial_conv = [
            nn.Conv2d(in_channels=3, 
                      out_channels=64, 
                      kernel_size=7, 
                      stride=1, 
                      padding=3)
        ]
        layers += self.initial_conv

        for i in range(2):
            layers += [
                nn.Conv2d(in_channels=(64 * (i+1)),
                          out_channels=(128 * (i+1)),
                          kernel_size=3,
                          stride=2,
                          padding=1
                          )
            ]

        for i in range(num_res_blocks):
            layers += [
                ResidualBlock(256, 256)
            ]
        
        conv_tranposes = [nn.ConvTranspose2d(256//i, 128//i, 3, stride=2, padding=1, output_padding=1) for i in range(1, 3)]
        layers += conv_tranposes

        last_layer = [nn.Conv2d(64, 3, 7, stride=1, padding=3)]
        layers += last_layer

        self.model = nn.Sequential(*layers)    


    def forward(self, x):
        x = self.model(x)
        return x

    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.AvgPool2d(30)

        )

    def forward(self, x):
        return self.sequential(x)[:,0,0,0]
    
    

def cycle_consistency_loss(fake_x, real_x, fake_y, real_y):
    criterion = nn.L1Loss()
    loss = criterion(fake_x, real_x) + criterion(fake_y, real_y)
    return loss

def adversial_loss(fake, real):
    return torch.nn.MSELoss()(fake, real)

def discriminator_loss(dis_fake, dis_real):
    loss = ( (dis_real - 1) **2 ) +  (dis_fake **2 )
    return loss

def criterion_identity_loss(fake, real):
    return torch.nn.L1Loss()(fake, real)
    


OUT_PATH = 'outputs'
EPOCHS = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gen = Generator()
dis_x = Discriminator()
dis_y = Discriminator()

# rand = torch.rand((2, 3,256,256))
# ones = torch.ones((2, 1))
# r = gen(rand)
# r = dis_x(r)
# r = criterion_identity_loss(r, ones)
# print(r)

fake_x_buffer = ReplayBuffer()
fake_y_buffer = ReplayBuffer()

gen_optim = torch.optim.Adam(gen.parameters())
dis_x2y_optim = torch.optim.Adam(dis_x.parameters())
dis_y2x_optim = torch.optim.Adam(dis_y.parameters())

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dir = 'summer2winter_yosemite/Train'
test_dir = 'summer2winter_yosemite/Test'

dataset = MyDataset(train_dir, transform=transforms) 

test_dataset = MyDataset(test_dir, transform=transforms)

BATCH_SIZE = 1

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


real = torch.autograd.Variable(torch.Tensor(BATCH_SIZE).fill_(1.0), requires_grad=False)
fake = torch.autograd.Variable(torch.Tensor(BATCH_SIZE).fill_(0.0), requires_grad=False)

for epoch in range(EPOCHS):
    # a -> x, b -> y
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # Generator
        gen_optim.zero_grad()
        same_x = gen(x)
        g_loss_same_x = criterion_identity_loss(same_x, x)
        same_y = gen(y)
        g_loss_same_y = criterion_identity_loss(same_y, y)

        fake_x = gen(y)
        fake_y = gen(x)


        g_loss_y2x = adversial_loss(dis_x(fake_x), real)
        g_loss_x2y = adversial_loss(dis_y(fake_y), real)

        cycle_loss = cycle_consistency_loss(fake_x, same_x, fake_y, same_y)

        total_g_loss = cycle_loss + g_loss_y2x + g_loss_x2y + g_loss_same_x + g_loss_same_y
        total_g_loss.backward()
        gen_optim.step()

        # Discriminators  
        dis_real_y = dis_y(y)
        d_loss_real_y = adversial_loss(dis_real_y, real)

        # fake loss y
        fake_y = fake_y_buffer.push_and_pop(fake_y)
        dis_fake_y = dis_y(fake_y.detach())
        d_loss_fake_y = adversial_loss(dis_fake_y, fake)
        
        # total loss
        d_loss_y = d_loss_real_y + d_loss_fake_y
        d_loss_y.backward()

        dis_x2y_optim.step()

        # loss x
        dis_real_x = dis_x(x)
        d_loss_real_x = adversial_loss(dis_real_x, real)
        
        fake_x = fake_x_buffer.push_and_pop(fake_x)
        dis_fake_x = dis_x(fake_x.detach())
        d_loss_fake_x = adversial_loss(dis_fake_x, fake)

        # total loss
        d_loss_x = d_loss_real_x + d_loss_fake_x
        d_loss_x.backward()

        dis_y2x_optim.step()
        

        if idx % 100 == 0:
            print('Epoch {} [{}/{}] total_g_loss: {:.4f} d_loss_x: {:.4f} d_loss_y: {:.4f}'.format(
                epoch, idx, len(train_loader),
                total_g_loss.mean().item(),
                d_loss_x.mean().item(),
                d_loss_y.mean().item()
            ))
                
            with torch.no_grad():
                y_ = gen(x)
                torchvision.utils.save_image(x, os.path.join(OUT_PATH, 'fake_samples_x_{}.png'.format(epoch)), normalize=True)
                torchvision.utils.save_image(y_, os.path.join(OUT_PATH, 'fake_samples_y_{}.png'.format(epoch)), normalize=True)
        
    torch.save(gen.state_dict(), os.path.join(OUT_PATH, 'gen_{}.pth'.format(epoch)))
    torch.save(dis_x.state_dict(), os.path.join(OUT_PATH, 'dis_x_{}.pth').format(epoch))
    torch.save(dis_y.state_dict(), os.path.join(OUT_PATH, 'dis_y_{}.pth').format(epoch))


