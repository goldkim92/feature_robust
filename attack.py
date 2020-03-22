import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

criterion = nn.CrossEntropyLoss()

def inference(model, input, idx2ctgr):
    model.eval()
    model.zero_grad()
    output = model(input)
    prob = F.softmax(output, dim=1)

    confidence = prob.max(dim=1)[0].detach().cpu().item()
    predict_idx = prob.max(dim=1)[1].cpu().item()
    predict_ctgr = idx2ctgr[predict_idx]
    return predict_idx, predict_ctgr, confidence

def attack_fgsm(model, input, target, eps):
    input.requires_grad = True

    model.eval()
    model.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

    attack = eps * input.grad.sign()
    return attack.cpu().squeeze(0)

def get_adversarial_image(img, attack):
    adv_img = tv.transforms.ToTensor()(img) + attack
    adv_img = adv_img.clamp(min=0., max=1.)
    adv_img = tv.transforms.ToPILImage()(adv_img)
    return adv_img

def image2input(img, norm, device):
    input = tv.transforms.ToTensor()(img)
    input = norm(input)
    input = input.unsqueeze(0).to(device)
    return input
