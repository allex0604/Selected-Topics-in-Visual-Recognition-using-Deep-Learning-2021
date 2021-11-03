from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import geffnet
from PIL import Image
root = r"C:\Users\user\Desktop\deep learning"
device = torch.device("cuda:0" if torch.cuda.is_available() else "gpu")


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
    # obtain model
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

    # create dataset and dataloader
    train_set = birds_datasets(root + '\\' + 'training_labels.txt',
                               0, transform=data_trans['train'])
    valid_set = birds_datasets(root + '\\' + 'training_labels.txt',
                               1, transform=data_trans['val'])
    test_set = birds_datasets(root + '\\' + 'training_labels.txt',
                              2, transform=data_trans['test'])
    train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=16, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
                    valid_set, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                    test_set, batch_size=16, shuffle=False)

    # adjust hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # training model(check overfit and accuracy)
    for epoch in range(20):
        net.train()
        running_loss = 0.0
        correct = 0
        for i, (X, labels) in enumerate(train_loader):
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
        print('train accuracy: %.4f%%' % (100.*correct/len(train_set)))
        print('train error: %.4f%%' % (100.*(1-correct/len(train_set))))

        # validation
        net.eval()
        running_loss = 0.0
        correct = 0
        l = 0
        with torch.no_grad():
            for data in valid_loader:
                X, labels = data
                X, labels = X.to(device), labels.to(device)
                outputs = net(X)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                l = l + 1
        running_loss = running_loss / l
        print('validation:')
        print('epoch: %d' % (epoch+1))
        print('validation accuracy: %.4f%%, loss: %.3f' % (
            100.*correct/len(valid_set), running_loss))
        print('validation error: %.4f%%' % (
            100.*(1-(correct/len(valid_set)))))
        print('--------------')

        # evaluation
        running_loss = 0.0
        correct = 0
        l = 0
        with torch.no_grad():
            for data in test_loader:
                X, labels = data
                X, labels = X.to(device), labels.to(device)
                outputs = net(X)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                l = l + 1
        running_loss = running_loss/l
        print('test:')
        print('epoch: %d' % (epoch+1))
        print('test accuracy: %.4f%%, loss: %.3f' % (
            100.*correct/len(test_set), running_loss))
        print('test error: %.4f%%' % (
            100.*(1-(correct/len(test_set)))))
        print('--------------')
