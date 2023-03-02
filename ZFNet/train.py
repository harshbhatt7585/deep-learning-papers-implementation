import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import dataset, transform
from model import ZFNet
import os


GPU = [0]
EPOCHS = 90
NO_CLASSES = 1000
TRAIN_DIR = ''
VAL_DIR = ''
BATCH_SIZE = 128
IMG_DIM = 256
LR = 0.01
MOMENTUM = 0.9
CHECKPOINT_DIR = 'checkpoints/'

device = torch.device('mps')

# Input Image Dim = 256 x 256
# Transformations - 
# cropping the center
# substracting the per person mean (across all images)
# cropping - conrners + center with horizontal flip



data_transform = transform.Compose([
    transform.CenterCrop(IMG_DIM),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
])

train_dataset = dataset.ImageFolder(TRAIN_DIR, data_transform)
val_dataset = dataset.ImageFolder(VAL_DIR)


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# initalizing model
model = ZFNet().to(device)

# optimizer 
optim = torch.optim.Adam(
    model.parameters(),
    le=LR,
    momentum=MOMENTUM
)

# loss function
criterion = nn.CrossEntropyLoss()

# decay the learning rate
lr_scheduler = torch.optim.lr_scheduler(optim, step_size=50, gamma=0.1)

t_steps = 0
# training
for epoch in EPOCHS:
    for step, batch in enumerate(train_loader):
        x, y = batch.to(device)
        optim.zero_grad()
        pred = model(x)
        loss = criterion(pred, y).to(device)
        loss.backward()
        optim.step()

        if t_steps % 10 == 0:
            print(f'step: {t_steps} | Loss: {loss}')
            t_steps += 1

    
    # saving checkpoints
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_checkpoint{epoch+1}.pkl')
    state = {
        'epoch': EPOCHS + 1,
        'total_steps': t_steps,
        'optimizer': optim,
    }
    torch.save(state, checkpoint_path)

        


