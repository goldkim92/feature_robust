import os
from os.path import join
import numpy as np
from PIL import Image
from scipy import interpolate

import torch
import torch.nn as nn
import torchvision as tv

import dataloader
import model
import util


class CAM(object):
    def __init__(self, model_type):
        self.model_type = model_type 
        self.batch_size = 1 
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load data & build model
        self.load_dataset()
        self.build_model()


    def load_dataset(self):
        t_input = tv.transforms.Compose([
            tv.transforms.Resize((224,224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                    std=(0.229, 0.224, 0.225)),
        ])  
        self.valid_dataset = dataloader.FolderDataset(t_input)


    def build_model(self):
        if self.model_type == 'vgg':
            self.model_f_extractor = model.vgg.FeatureExtractor()
            self.retain_grad = model.vgg.retain_grad
            self.model_classifier = model.vgg.Classifier()
        else:
            raise Exception("'model_type' should in one of the ['vgg','resnet','googlenet']")
            
        self.model_f_extractor = self.model_f_extractor.to(self.device)
        self.model_classifier = self.model_classifier.to(self.device)
        self.model = nn.Sequential(
            self.model_f_extractor,
            self.model_classifier,
        )


    def get_item(self, index):
        input, target = self.valid_dataset[index]
        input, target = input.unsqueeze(0), torch.tensor(target).unsqueeze(0)
        input, target = input.to(self.device), target.to(self.device)
        return input, target
            

    def get_weights(self, input, att_idx, norm=True):
        heatmaps = []

        # model phase
        self.model_f_extractor.eval()
        self.model_classifier.eval()

        # forward-backward propagation
        features = self.model_f_extractor(input)
        features = self.retain_grad(features)
        score = self.model_classifier(features)
        score[:,att_idx].backward(retain_graph=True)

        # get Grad-CAM
        gradients = features.grad.squeeze(0) # size = [512,14,14]
        weights = torch.mean(gradients, dim=[1,2], keepdim=True) # size = [512,1,1]
        weights = weights.squeeze()
        return weights


    def get_heatmaps(self, input, att_idx, phase, mc, norm=True):
        heatmaps = []

        # model phase
        self.model_f_extractor.eval()
        if phase == 'test':
            self.model_classifier.eval()
        else:
            self.model_classifier.train()

        # forward-backward propagation
        features = self.model_f_extractor(input)
        features = self.retain_grad(features, phase)
        for _ in range(mc):
            score = self.model_classifier(features)
            score[:,att_idx].backward(retain_graph=True)

            # get Grad-CAM
            gradients = features.grad.squeeze(0) # size = [512,14,14]
            weights = torch.mean(gradients, dim=[1,2], keepdim=True) # size = [512,1,1]
            grad_cam = features.detach().squeeze(0) * weights # size = [512,14,14]
            grad_cam = grad_cam.mean(dim=0).cpu()
            grad_cam = torch.max(grad_cam, torch.tensor(0.)) #size = [14,14]

            
            # get heatmap(s)
            if norm:
                heatmap = util.cam2heatmap(grad_cam)
                heatmaps.append(heatmap)
            else:
                grad_cam = grad_cam.numpy()
                f = interpolate.interp2d(np.arange(14),np.arange(14), grad_cam)
                heatmap = f(np.arange(0,14,1/16),np.arange(0,14,1/16))
                heatmaps.append(heatmap)

        heatmaps = np.array(heatmaps).squeeze()
        return heatmaps


    def get_values(self, data_idx, att_idx, th1=0.2, th2=10, mc=30, phase='test'):
        # get input, target, and img (PIL.Image format)
        input, target = self.get_item(data_idx)
        img = util.torch2pil(input)

        # make boolmap from heatmap
        if phase == 'test':
            heatmap = self.get_heatmaps(input, att_idx, phase='test', mc=1)
            boolmap = util.heatmap2boolmap(heatmap, a=th1)
        else:
            heatmaps = self.get_heatmaps(input, att_idx, phase='train', mc=mc)
            heatmap_mean = heatmaps.mean(0)
            heatmap_std = heatmaps.std(0)
            
            boolmap_mean = util.heatmap2boolmap(heatmap_mean, a=th1)
            boolmap_std = util.heatmap2boolmap(heatmap_std, a=th2)
            boolmap = np.logical_or(boolmap_mean, boolmap_std)
            
        # segment the biggest component
        boolmap_biggest = util.get_biggest_component(boolmap)
        
        # get bbox
        bbox = util.boolmap2bbox(boolmap_biggest)
        
        if phase == 'test':
            return img, heatmap, boolmap, boolmap_biggest, bbox
        else:
            return img, heatmap_mean, heatmap_std, boolmap, boolmap_biggest, bbox
