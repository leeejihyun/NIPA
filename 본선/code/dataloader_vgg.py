import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image

def get_transform(method=Image.BILINEAR):
    transform_list = []

    transform_list.append(transforms.Resize((64,64)))
    transform_list.append(transforms.Grayscale())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5), (0.5)))
    
    return transforms.Compose(transform_list)


class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        
        self.label_path = os.path.join(root, self.phase, self.phase+'.csv')
        with open(self.label_path, 'r', encoding='utf-8-sig') as f:
            file_list = []
            label = []
            
            for line in f.readlines()[1:]:
                v = line.strip().split(',')
                file_list.append(v[0])
                if self.phase != 'test':
                    label.append(v[2])

        self.imgs = list(file_list)
        self.labels = list(label)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.phase, self.imgs[index])
        
        if self.phase != 'test':
            label = self.labels[index]
            label = torch.tensor(int(label))

        transform = get_transform()
        image = Image.open(image_path)
        image = transform(image)

        if self.phase != 'test' :
            return (image, label)
        elif self.phase == 'test' :
            dummy = ""
            return (image, dummy)

    def __len__(self):
        return len(self.imgs)

    def get_label_file(self):
        return self.label_path


def data_loader(root, phase='train', batch_size=16):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CustomDataset(root, phase)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_label_file()
