import os
from os.path import join

import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg16_bn, vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # get the pretrained VGG network
        self.model = vgg16_bn(pretrained=True)

        # disect the network to access its last convolutional layer
        self.features_conv = self.model.features[:-1]

        # delete self.model variable
        del self.model

    def forward(self, x):
        return self.features_conv(x)


def retain_grad(features, phase='test'):
#     if phase == 'test':
#         features.retain_grad()
#     else:
    # detach and requires_grad = True
    features = features.detach()
    features.requires_grad = True
    return features


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # get the pretrained VGG network
        self.model = vgg16_bn(pretrained=True)

        # get the max pool of the features stem
        self.max_pool = self.model.features[-1:]

        # get the classifier of the vgg19
        self.classifier = self.model.classifier

        # delete self.model variable
        del self.model

    def forward(self, features):
        # apply the remaining pooling
        x = self.max_pool(features)
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x


# class VGG_for_CAM(nn.Module):
#     def __init__(self):
#         super(VGG_for_CAM, self).__init__()
#         
#         # get the pretrained VGG network
#         self.model = vgg16_bn(pretrained=True)
#         
#         # disect the network to access its last convolutional layer
#         self.features_conv = self.model.features[:-1]
#         
#         # get the max pool of the features stem
#         self.max_pool = self.model.features[-1:]
#         
#         # get the classifier of the vgg19
#         self.classifier = self.model.classifier
#         
#         # delete self.model variable
#         del self.model
#         
#         # placeholder for the gradients and feature_conv
#         self.gradients = None
#         self.features = None
#     
#     # hook for the gradients of the activations
#     def activations_hook(self, grad):
#         self.gradients = grad
#         
#     def forward(self, x):
#         self.features = self.features_conv(x)
#         
#         # register the hook
#         h = self.features.register_hook(self.activations_hook)
#         
#         # apply the remaining pooling
#         x = self.max_pool(self.features)
#         x = x.view((x.size(0), -1))
#         x = self.classifier(x)
#         return x
#     
#     # method for the gradient extraction
#     def get_activations_gradient(self):
#         return self.gradients
#     
#     # method for the activation exctraction
#     def get_activations(self):
#         return self.features
# 
