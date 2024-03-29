import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler
from latex2img import latex2img
import torchvision.transforms as transforms
from torchvision import utils as vutils
from math import *
import random

max_H = 512
max_W = 2560
class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        image = torch.Tensor(255-image) / 255
        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        
        return image, words
    
class HMERDatasetWithTwoSource(Dataset):
    def __init__(self, params, image_hand_path,image_standard_path, label_path, words, is_train=True):
        super(HMERDatasetWithTwoSource, self).__init__()
        if image_hand_path.endswith('.pkl'):
            with open(image_hand_path, 'rb') as f:
                self.images_hand = pkl.load(f)
        elif image_hand_path.endswith('.list'):
            with open(image_hand_path, 'r') as f:
                lines = f.readlines()
            self.images_hand = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images_hand.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')
        
        if image_standard_path.endswith('.pkl'):
            with open(image_standard_path, 'rb') as f:
                self.images_standard = pkl.load(f)
        elif image_standard_path.endswith('.list'):
            with open(image_standard_path, 'r') as f:
                lines = f.readlines()
            self.images_standard = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images_standard.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params
        self.len = len(self.labels)

    def __len__(self):
        assert len(self.images_hand) == len(self.images_standard)
        assert len(self.images_hand) == len(self.labels)
        return len(self.labels)
        #return 8

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image_hand = self.images_hand[name]
        #image_standard = self.images_standard[name]
        image_hand = torch.Tensor(255-image_hand) / 255
        image_hand = image_hand.unsqueeze(0)
        #image_standard = torch.Tensor(255-image_standard) / 255
        #image_standard = image_standard.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        
        if self.is_train:
            words_new = "$"+ self.labels[idx].split("\t")[1][:-1] + "$" 
        else:
            words_new = "$"+" ".join(self.labels[idx].split(" ")[1:])[:-1]+"$"
        #print(words_new)     
        words_new = words_new.replace(" _ ", "_")
        words_new = words_new.replace(" ^ ", "^")

        transf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor()]
        )
        try:
            a = latex2img(words_new)
            standard_image = a.get_img()
            standard_image = transf(standard_image)
            standard_image = (standard_image)/255

        except:
            #print("data error!")
            k = random.randint(0, self.len-1)
            return self.__getitem__(k)
        
        return image_hand, standard_image, words


class img_to_img(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(img_to_img, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        
        
        image = torch.Tensor(255-image) / 255
        image = image.unsqueeze(0)
        H, W = image.shape[1], image.shape[2]
        image = transforms.Pad([floor((max_W - W)/2), floor((max_H - H)/2), ceil((max_W - W)/2), ceil((max_H - H)/2)
            ],fill=(1),padding_mode='constant')(image)

        words = "$"+ self.labels[idx].split("\t")[1][:-1] + "$"       
        words = words.replace(" _ ", "_")
        words = words.replace(" ^ ", "^")
        
        transf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor()]
        )

        a = latex2img(words)      
        a.resize((max_H, max_W))
        target_image = a.get_img()
        
        target_tensor = transf(target_image)
        
        
        return image, target_tensor

def get_crohme_dataset_two_source(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"训练数据路径 hand written images: {params['train_image_hand_path']} standard images: {params['train_image_standard_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 hand written images: {params['eval_image_hand_path']} standard images: {params['eval_image_standard_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDatasetWithTwoSource(params, params['train_image_hand_path'], params['train_image_standard_path'], params['train_label_path'],words, is_train=True)
    eval_dataset = HMERDatasetWithTwoSource(params, params['eval_image_hand_path'],params['eval_image_standard_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader
        
def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader

def collate_fn_two_source(batch_images):
    # process hand images and standard images
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        if item[1].shape[1] * max_width > 1600 * 320 or item[1].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        
        max_height = item[1].shape[1] if item[1].shape[1] > max_height else max_height
        max_width = item[1].shape[2] if item[1].shape[2] > max_width else max_width
        
        max_length = item[2].shape[0] if item[2].shape[0] > max_length else max_length
        
        proper_items.append(item)
    
    images_hand, image_masks_hand = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    images_standard, image_masks_standard = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))
    
    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images_hand[i][:, :h, :w] = proper_items[i][0]
        image_masks_hand[i][:, :h, :w] = 1
        
        _, h, w = proper_items[i][1].shape
        images_standard[i][:, :h, :w] = proper_items[i][1]
        image_masks_standard[i][:, :h, :w] = 1
        
        l = proper_items[i][2].shape[0]
        labels[i][:l] = proper_items[i][2]
        labels_masks[i][:l] = 1
        
    return images_hand, image_masks_hand,images_standard,image_masks_standard, labels, labels_masks

def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn ,
    'collate_fn_two_source': collate_fn_two_source
}

