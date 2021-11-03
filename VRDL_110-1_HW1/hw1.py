from torchvision import transforms
import torch
import torch.nn as nn
import geffnet
import numpy as np
import torch.optim as optim
from PIL import Image


def MyLoader(path):
    return Image.open(path).convert('RGB')


class birds_datasets():
    def __init__(self, txt, x, transform=None, loader=MyLoader):
        with open(txt, 'r') as fh:
            imgs = []
            # train_data
            if x == 0:
                i = 0
                for line in fh:
                    if i % 10 != 0:
                        line = line.strip('\n')
                        line = line.rstrip()
                        words = line.split()
                        num = words[1].split(".")
                        imgs.append((words[0], int(num[0])-1))
                    i += 1
            # validation_data
            elif x == 1:
                i = 0
                for line in fh:
                    if i % 8 == 0:
                        line = line.strip('\n')
                        line = line.rstrip()
                        words = line.split()
                        num = words[1].split(".")
                        imgs.append((words[0], int(num[0])-1))
                    i += 1
            # test_data
            elif x == 2:
                i = 0
                for line in fh:
                    if i % 10 == 0:
                        line = line.strip('\n')
                        line = line.rstrip()
                        words = line.split()
                        num = words[1].split(".")
                        imgs.append((words[0], int(num[0])-1))
                    i += 1
            # total_train
            elif x == 3:
                for line in fh:
                        line = line.strip('\n')
                        line = line.rstrip()
                        words = line.split()
                        num = words[1].split(".")
                        imgs.append((words[0], int(num[0])-1))
            # final_test
            elif x == 4:
                for line in fh:
                    line = line.strip('\n')
                    line = line.rstrip()
                    words = line.split()
                    imgs.append((words[0], words[0]))
        self.x = x
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        x = self.x

        if x == 0 or x == 1 or x == 2 or x == 3:
            path = root + '\\' + 'train' + '\\' + fn
        else:
            path = root + '\\' + 'test' + '\\' + fn
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root = r"C:\Users\user\Desktop\deep learning"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "gpu")
    net = geffnet.create_model('tf_efficientnet_b3_ns', pretrained=True)
    # change out_features to num_class
    net.classifier = nn.Linear(1536, 200)
    # run to GPU
    net.to(device)

    data_trans = {
        'train': transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomCrop((320, 320)),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.CenterCrop((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.CenterCrop((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.4559, 0.4425, 0.4395],
                                 [0.2623, 0.2608, 0.2648])
        ]),
    }

    total_train_set = birds_datasets(root + '\\' + 'training_labels.txt',
                                     3, transform=data_trans['train'])
    final_test_set = birds_datasets(root + '\\' + 'testing_img_order.txt',
                                    4, transform=data_trans['test'])
    total_train_loader = torch.utils.data.DataLoader(
                    total_train_set, batch_size=25, shuffle=True)
    final_test_loader = torch.utils.data.DataLoader(
                            final_test_set, batch_size=1, shuffle=False)

    # adjust hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # if OK! training all training image
    for epoch in range(20):
        net.train()
        running_loss = 0.0
        correct = 0
        for i, (X, labels) in enumerate(total_train_loader):
            X, labels = X.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat = net(X)

            # train accu compute (correct num)
            _, pred = y_hat.max(1)
            correct += pred.eq(labels).sum().item()
            loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Train:')
        print('[%d, %5d] loss: %.3f' % (epoch + 1,
                                        i + 1, running_loss / (i+1)))
        scheduler.step()

    # record our training
    path = './my_model.pth'
    torch.save(net.state_dict(), path)

    # load pretrain data
    net.load_state_dict(torch.load('./my_model.pth'))

    # predict test_data
    net.eval()
    preds = []
    for data in final_test_loader:
        X, image_name = data
        X = X.to(device)
        outputs = net(X)
        outputs = torch.softmax(outputs, dim=1)
        _, pred = outputs.max(1)
        preds += pred.tolist()

    size = 3033
    cla = np.empty(size, "S50")
    with open('classes.txt', 'r') as fh:
        i = 0
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            cla[i] = line
            i += 1

    # all the testing images
    with open('testing_img_order.txt') as f:
        test_images = [x.strip() for x in f.readlines()]

    submission = []
    i = 0
    for img in test_images:  # image order is important to your result
        predicted_class = cla[preds[i]]
        submission.append([img, predicted_class])
        i += 1

    np.savetxt('answer.txt', submission, fmt='%s')
