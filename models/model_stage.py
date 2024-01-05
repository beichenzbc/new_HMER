import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.densenet import DenseNet
from models.counting import CountingDecoder as counting_decoder
from counting_utils import gen_counting_label

def contrastive_loss(hidden, hidden_norm=True, temperature=1.0, weights=None):
    # Use a smaller negative number for stability
    SMALL_NEG_NUM = -1e6

    # Normalize the hidden vectors if specified
    if hidden_norm:
        hidden = F.normalize(hidden, p=2, dim=-1)

    # Split the hidden vectors into two halves
    hidden1, hidden2 = torch.split(hidden, hidden.size(0) // 2, 0)
    batch_size = hidden1.size(0)
    torch.set_printoptions(precision=2, sci_mode=False)
    # Compute similarity scores
    logits_aa = torch.matmul(hidden1, hidden1.T) / temperature
    logits_bb = torch.matmul(hidden2, hidden2.T) / temperature
    logits_ab = torch.matmul(hidden1, hidden2.T) / temperature
    logits_ba = torch.matmul(hidden2, hidden1.T) / temperature
    
    # Mask out self-similarity in aa and bb
    masks = torch.eye(batch_size).to(hidden.device)
    logits_aa = logits_aa.masked_fill(masks.bool(), SMALL_NEG_NUM)
    logits_bb = logits_bb.masked_fill(masks.bool(), SMALL_NEG_NUM)

    # Combine logits
    logits = torch.cat([torch.cat([logits_ab, logits_aa], dim=1),
                        torch.cat([logits_bb, logits_ba], dim=1)], dim=0)
    # Create labels
    labels = torch.eye(2 * batch_size).to(hidden.device)
    # Compute the loss
    loss = F.cross_entropy(logits, labels.argmax(dim=1), reduction='mean', weight=weights)
    return loss, logits, labels

def sampling_from_two_source(tensor1, tensor2, prob_from_first=0.5, mask=None):
    if mask is None:
        # mask 的形状应与 tensor1 和 tensor2 的批次维度相同
        batch_size = tensor1.size(0)
        mask = torch.rand(batch_size) < prob_from_first
        mask = mask.to(tensor1.device)

    return mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float() * tensor1 + (1 - mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()) * tensor2, mask



class Prototype(nn.Module):
    def __init__(self, params=None):
        super(Prototype, self).__init__()
        self.params = params
        self.use_label_mask = params['use_label_mask']
        
        self.encoder_hand = DenseNet(params=self.params)
        self.encoder_standard = DenseNet(params=self.params)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # contrastive learning
        
        self.projector = nn.Sequential(
            nn.Linear(params['encoder']['out_channel'], params['encoder']['out_channel']),
            nn.ReLU(),
            nn.Linear(params['encoder']['out_channel'], params['projection_dim'])
        )
        '''
        self.projector_standard = nn.Sequential(
            nn.Linear(params['encoder']['out_channel'],  params['encoder']['out_channel']),
            nn.ReLU(),
            nn.Linear(params['encoder']['out_channel'], params['projection_dim'])
        )
        '''
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']
        
        self.counting_decoder1 = counting_decoder(self.in_channel, self.out_channel, 3)
        self.counting_decoder2 = counting_decoder(self.in_channel, self.out_channel, 5)
        
        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
        self.contrastive_loss = contrastive_loss


        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

    def forward(self, images_hand, images_standard, images_hand_mask, images_standard_mask, labels, labels_mask,beta,temperature, is_train=True, is_standard = False):
        # encode images
        if is_standard:
            cnn_features = self.encoder_standard(images_standard)
            counting_mask = images_standard_mask[:, :, ::self.ratio, ::self.ratio]
            image_mask = images_standard_mask
            
        else:
            cnn_features = self.encoder_hand(images_hand)
            counting_mask = images_hand_mask[:, :, ::self.ratio, ::self.ratio]
            image_mask = images_hand_mask
      
        # breakpoint()
        # contrastive learning
        
        '''
        concat_features = torch.cat((project_features_hand,project_features_standard),0)
        batch_size = images_hand.shape[0]
        if is_train and batch_size > 1:
            contrastive_loss, _, _ = self.contrastive_loss(concat_features, hidden_norm=True, temperature=temperature)
        else:
            contrastive_loss = torch.tensor([0]).cuda()
        '''
        # generate counting label
        counting_labels = gen_counting_label(labels, self.out_channel, True)

        # counting
        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        
        counting_preds = (counting_preds1 + counting_preds2) / 2
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)

        # decoding
        word_probs, word_alphas = self.decoder(cnn_features, labels, counting_preds, image_mask, labels_mask, is_train=is_train)
        word_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss
        
        return word_probs, counting_preds, word_average_loss, counting_loss

    def encoder_both(self, images_hand, images_standard):
        cnn_features_hand = self.encoder_hand(images_hand)
        cnn_features_standard = self.encoder_standard(images_standard)
        return cnn_features_hand, cnn_features_standard

        # breakpoint()
        # contrastive learning
        project_features_hand = self.projector(self.avgpool(cnn_features_hand).squeeze())
        project_features_standard = self.projector(self.avgpool(cnn_features_standard).squeeze())
        project_features_hand = torch.nn.functional.normalize(project_features_hand, p=2, dim=1)
        project_features_standard = torch.nn.functional.normalize(project_features_standard, p=2, dim=1)
        return project_features_hand, project_features_standard

