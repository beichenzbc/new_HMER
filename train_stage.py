import os
import time
import argparse
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils import load_config_two_source, save_checkpoint, load_checkpoint
from dataset import get_crohme_dataset_two_source
from models.model_stage import Prototype
from training_stage import train_standard, train_hand, train_contrastive, eval_hand, eval_standard

parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--check', action='store_true', help='测试代码选项')
args = parser.parse_args()

if not args.dataset:
    print('请提供数据集名称')
    exit(-1)

if args.dataset == 'CROHME':
    config_file = 'config_with_contrastive_loss.yaml'

"""加载config文件"""
params = load_config_two_source(config_file)

"""设置随机种子"""
random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

if args.dataset == 'CROHME':
    train_loader, eval_loader = get_crohme_dataset_two_source(params)

model = Prototype(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'

print(model.name)
model = model.to(device)

if args.check:
    writer = None
else:
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

optimizer_hand = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=1,
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay'])*1.5)
optimizer_contrastive = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=1,
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))
optimizer_standard = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=0.5,
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))
#optimizer_standard = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
#optimizer_contrastive = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
#optimizer_hand = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

'''
scheduler_standard = torch.optim.lr_scheduler.ExponentialLR(optimizer_standard, 0.95, last_epoch=-1, verbose=False)
scheduler_contrastive = torch.optim.lr_scheduler.ExponentialLR(optimizer_contrastive, 0.95, last_epoch=-1, verbose=False)
scheduler_hand = torch.optim.lr_scheduler.ExponentialLR(optimizer_hand, 0.97, last_epoch=-1, verbose=False)
'''

if params['finetune']:
    print('加载预训练模型权重')
    print(f'预训练权重路径: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])

if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')


"""在CROHME上训练"""
if args.dataset == 'CROHME':
    min_score, init_epoch = 0, 0

    for epoch in range(init_epoch, params['epochs']):
        if epoch < 80:
            train_loss, train_word_score, train_exprate = train_standard(params, model, optimizer_standard, epoch, train_loader, writer=writer)
        if epoch >= 80 and epoch < 160:
            contrastive_loss = train_contrastive(params, model, optimizer_contrastive, epoch, train_loader, writer=writer)
        if epoch >= 160:
            train_loss, train_word_score, train_exprate = train_hand(params, model, optimizer_hand, epoch, train_loader, writer=writer)
        '''
        if epoch < 70:
            eval_loss, eval_word_score, eval_exprate = eval_standard(params, model, epoch, eval_loader, writer=writer)
            print(f'Standard Epoch: {epoch+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
        '''

        if epoch >= params['valid_start'] and epoch >= 80:
            eval_loss, eval_word_score, eval_exprate = eval_hand(params, model, epoch, eval_loader, writer=writer)
            print(f'Hand Epoch: {epoch+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
            if eval_exprate > min_score and not args.check and epoch >= params['save_start']:
                min_score = eval_exprate
                if epoch < 160:
                    save_checkpoint(model, optimizer_contrastive, eval_word_score, eval_exprate, epoch+1,
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
                else:
                    save_checkpoint(model, optimizer_hand, eval_word_score, eval_exprate, epoch+1,
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])