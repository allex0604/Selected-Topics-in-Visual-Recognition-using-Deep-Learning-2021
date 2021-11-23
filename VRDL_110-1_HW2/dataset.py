from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import glob
hw2 = os.getcwd()


class Prediction_Dataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.files = sorted(
            glob.glob("%s\*.*" % image_dir),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        # print(img.shape)
        return img_path, img

    def __len__(self):
        return len(self.files)


class Num_Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['img_name'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        data = self.df[self.df['img_name'] == image_id]
        # reading the images and converting them to correct size and color
        img_path = os.path.join(self.image_dir, image_id)
        img1 = Image.open(img_path).convert("RGB")
        boxes = data[['left', 'top', 'width', 'height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = data['label'].values
        labels = torch.as_tensor(labels, dtype=torch.int64)

        iscrowd = torch.zeros((data.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms:
            img = self.transforms(img1)

        return img, target

    def __len__(self):
        return self.image_ids.shape[0]
