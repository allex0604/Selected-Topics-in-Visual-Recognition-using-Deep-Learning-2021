import argparse
import pandas as pd
import torch
import os
from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision
from engine import*


# address of the hw2_folder
hw2 = os.getcwd()
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2,
                        help="size of image batch")
    parser.add_argument("--num_cpu", type=int, default=8,
                        help="decides the number of num_worker")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate setting")
    parser.add_argument("--epoch", type=int, default=10,
                        help="training times")
    param = parser.parse_args()

    # defines dataset and dataloader of train , validation
    train_df = pd.read_csv(os.path.join(hw2, "train1.csv"))
    DIR_TRAIN = os.path.join(hw2, "train")
    image_ids = train_df['img_name'].unique()

    valid_ids = image_ids[-2500:]
    train_ids = image_ids[:-2500]

    valid_df = train_df[train_df['img_name'].isin(valid_ids)]
    train_df = train_df[train_df['img_name'].isin(train_ids)]
    transform = transforms.ToTensor()

    train_dataset = Num_Dataset(train_df, DIR_TRAIN, transform)
    valid_dataset = Num_Dataset(valid_df, DIR_TRAIN, transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=param.batch,
                              shuffle=True,
                              num_workers=param.num_cpu,
                              pin_memory=True,
                              collate_fn=utils.collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=param.batch,
                              shuffle=False,
                              num_workers=param.num_cpu,
                              pin_memory=True,
                              collate_fn=utils.collate_fn)

    # model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            progress=True,
            num_classes=11,
            pretrained_backbone=True)
    model.to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=param.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.1)

    for epoch in range(param.epoch):
        # training for one epoch
        train_one_epoch(model, optimizer, train_loader, device,
                        epoch, print_freq=50)

        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)

    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn1.pth')
