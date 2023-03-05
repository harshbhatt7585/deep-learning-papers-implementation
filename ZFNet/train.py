import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from model import ZFNet
import os
from PIL import Image
import natsort


GPU = [0]
EPOCHS = 90
NO_CLASSES = 1000
TRAIN_DIR = 'dataset/imagenet-mini/train'
VAL_DIR = 'dataset/imagenet-mini/val'
BATCH_SIZE = 128
IMG_DIM = 256
LR = 0.01
MOMENTUM = 0.9
CHECKPOINT_DIR = 'checkpoints/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Input Image Dim = 256 x 256
# Transformations - 
# cropping the center
# substracting the per person mean (across all images)
# cropping - conrners + center with horizontal flip



data_transform = transforms.Compose([
    transforms.CenterCrop(IMG_DIM),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
])

# class CustomDataSet(Dataset):
#     def __init__(self, main_dir, transform):
#         self.main_dir = main_dir
#         self.transform = transform
#         all_imgs_dirs = os.listdir(main_dir)
#         all_imgs = os.listdir(all_imgs_dirs)
#         self.total_imgs = natsort.natsorted(all_imgs)

#     def __len__(self):
#         return len(self.total_imgs)

#     def __getitem__(self, idx):
#         img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
#         print(self.total_imgs[idx])
#         image = Image.open(img_loc).convert("RGB")
#         tensor_image = self.transform(image)
#         return tensor_image


train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, data_transform)


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# initalizing model
model = ZFNet().to(device)

# optimizer 
optim = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

# loss function
criterion = nn.CrossEntropyLoss()

# decay the learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)

t_steps = 0
# training
for epoch in tqdm(range(EPOCHS)):
    for step, batch in enumerate(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
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

        


