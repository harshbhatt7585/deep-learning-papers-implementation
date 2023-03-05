import torch
from model import ZFNet
import argparse
from train import data_transform
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def cal_accuracy(pred, y):
    com = torch.float(pred == y)
    acc = torch.sum(com)/com.size(0)
    return acc

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # laod pre-trained model
    model = ZFNet().to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    test_dataset = data_transform.ImageFolder(args.image_dir, data_transform)
    test_loader = DataLoader(test_dataset, batch_size=128)

    t_acc = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x, y = batch.to(device)
            pred = model(x)
            acc = cal_accuracy(pred, y)
            t_acc += acc
        print(t_acc/len(test_loader))

if __name__ == '__main__':


    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Checkpoint path")
    ap.add_argument("--image_dir", required=True, help="Directory of images to test")
    args = vars(ap.parse_args())




