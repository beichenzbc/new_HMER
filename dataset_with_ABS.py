import torch
import time
import pickle as pkl
import sys
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from latex2img import latex2img
from counting_utils import gen_counting_label
import torchvision.transforms as transforms
from torchvision import utils as vutils
from math import *
import numpy as np
from memory_profiler import profile
import random
max_H = 512
max_W = 2560
class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True, preprocess=False):
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
        self.preprocessed = False
        if preprocess:
            self.out_channel = params['counting_decoder']['out_channel']
            self.preprocess()
            self.preprocessed = True

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if self.preprocessed:
            return self.data["images"][idx], self.data["words"][idx], self.data["counting_labels"][idx]
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        image = torch.Tensor(255-image) / 255
        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        
        return image, words
    
    def get_shape(self,idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        return image.shape
    
    def preprocess(self):
        self.data = {"images":[],"labels":[],"words":[],"counting_labels":[] }
        for idx in range(len(self)):
            name, *labels = self.labels[idx].strip().split()
            name = name.split('.')[0] if name.endswith('jpg') else name
            image = self.images[name]
            image = torch.Tensor(255-image) / 255
            image = image.unsqueeze(0)
            labels.append('eos')
            words = self.words.encode(labels)
            words = torch.LongTensor(words)
            self.data["images"].append(image)
            self.data["labels"].append(labels)
            self.data["words"].append(words)
            self.data["counting_labels"].append(gen_counting_label(words.unsqueeze(0), self.out_channel, True))
        

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
    
    def get_shape(self,idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images_hand[name]
        return image.shape

class AdaptiveBatchSampler(Sampler[List[int]]):
    r"""Wraps the dataset to yield a mini-batch of indices, the num of each batch is adaptively adjusted.

    Args:
        dataset (Dataset): The dataset to sample from.
        max_batch_size (int): The maximum number of data points in a batch.
        max_image_size (int): The maximum number of image pixels in a batch.
        random_sample (bool): If ``True``, samples are drawn randomly. Otherwise,
    """

    def __init__(self, dataset: Union[HMERDataset, HMERDatasetWithTwoSource], 
                    max_batch_size: int=-1, max_image_size:int=-1, 
                    max_tensor_size:int=-1, 
                    random_sample: bool = True) -> None:
        size_valid = lambda x: isinstance(x, int) and x > 0
        if size_valid(max_tensor_size):
            self.use_tensor_size = True 
            self.max_tensor_size = max_tensor_size
        elif (size_valid(max_batch_size) and size_valid(max_image_size)):
            self.use_tensor_size = False
            self.max_batch_size = max_batch_size
            self.max_image_size = max_image_size
        else:
            raise ValueError('USAGE: max_batch_size and max_image_size should be positive integers, or max_tensor_size should be a positive integer!')

        self.dataset = dataset
        self.length = None
        if random_sample:
            self._index_iter = RandomSampler(self.dataset)
        else:
            self._index_iter = SequentialSampler(self.dataset)
        self.__evaluate_length()


    def __iter__(self) -> Iterator[List[int]]:
        return self._get_generator()
    
    def _get_generator(self):
        ## when call self._get_generator(), it will return a generator RATHER THAN a list
        # print('_get_generator called')
        holdon = []
        sampler_iter = iter(self._index_iter)
        while True:
            try:
                batch = holdon
                holdon = []
                batch_size = len(batch)
                max_height, max_width = 0, 0
                if len(batch) > 0:
                    for idx in batch:
                        shape = self.dataset.get_shape(idx)
                        max_height = shape[0] if shape[0] > max_height else max_height
                        max_width = shape[1] if shape[1] > max_width else max_width
                
                if self.use_tensor_size:
                    while batch_size*max_height*max_width < self.max_tensor_size:
                        idx = next(sampler_iter)
                        shape = self.dataset.get_shape(idx)
                        
                        if (batch_size+1)*(max(shape[0],max_height))*(max(shape[1],max_width)) > self.max_tensor_size:
                            holdon.append(idx)
                            break
                        max_height = shape[0] if shape[0] > max_height else max_height
                        max_width = shape[1] if shape[1] > max_width else max_width
                        batch_size+=1
                        batch.append(idx)
                else: # use batch_size and image_size
                    while batch_size < self.max_batch_size:
                        idx = next(sampler_iter)
                        shape = self.dataset.get_shape(idx)  ## looks like (64, 256)

                        if shape[0] * max_width > self.max_image_size or shape[1] * max_height > self.max_image_size:
                            holdon.append(idx)
                            break
                        max_height = shape[0] if shape[0] > max_height else max_height
                        max_width = shape[1] if shape[1] > max_width else max_width
                        batch_size+=1
                        batch.append(idx)
                # print("DEBUG: ",batch_size,max_height,max_width)
                yield batch

            except StopIteration:
                batch = batch + holdon
                yield batch
                break

    def __evaluate_length(self) -> int:
        ## We iterate over the dataset 5 times and count the mean number of batches
        ## This is not efficient, but we don't know the length of the dataset
        N = 5
        length = []
        for i in range(N):
            length.append(len(list(self._get_generator())))
        self.length = int(np.mean(length))
        # print(f'DEBUG: HMER Dataset: Evaluated length = {self.length}')

    def __len__(self) -> int:
        # We don't know it!
        # We can only return the minimum number of batches or the maximum number of batches

        # raise NotImplementedError
        # return len(self.dataset)
        return self.length
        # return len(self.dataset) // self.max_batch_size
    ...

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
    print(f"训练数据路径 hand written images: {params['train_image_hand_path']} standard images: {params['train_image_standard_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 hand written images: {params['eval_image_hand_path']} standard images: {params['eval_image_standard_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDatasetWithTwoSource(params, params['train_image_hand_path'], params['train_image_standard_path'], params['train_label_path'],words, is_train=True)
    eval_dataset = HMERDatasetWithTwoSource(params, params['eval_image_hand_path'],params['eval_image_standard_path'], params['eval_label_path'], words, is_train=False)
    if not params['use_ada_sampler']:
        assert params['batch_size']%params['grad_accumulation'] == 0
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = RandomSampler(eval_dataset)
    
        train_loader = DataLoader(train_dataset, batch_size=int(params['batch_size']/params['grad_accumulation']), sampler=train_sampler,
                                num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
        eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                                num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    else:
        # train_sampler = AdaptiveBatchSampler(train_dataset, None,None, 1200000)
        train_sampler = AdaptiveBatchSampler(train_dataset, None,None, int(1350000*2))
        # train_sampler = AdaptiveBatchSampler(train_dataset, None,None, 1600000)
        eval_sampler = AdaptiveBatchSampler(eval_dataset, None,None, int(10000000*2))
        train_loader = DataLoader(train_dataset, batch_size=1, batch_sampler=train_sampler,
                                num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
        eval_loader = DataLoader(eval_dataset, batch_size=1, batch_sampler=eval_sampler,
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

def collate_fn_bak(batch_images):
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

def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        # if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
        #     continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images = torch.zeros((len(proper_items), channel, max_height, max_width))
    image_masks = torch.zeros((len(proper_items), 1, max_height, max_width))
    # images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))
    counting_labels = torch.zeros((len(proper_items), 111))


    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
        counting_labels[i] = proper_items[i][2]
    return images, image_masks, labels, labels_masks, counting_labels

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
