# -*- coding: utf-8 -*-
import time
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from scipy.fftpack import dct, idct
from scipy import linalg as la
import torch.nn.functional as F
import logging

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


logging.basicConfig(level = logging.INFO,format = '%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((1, 2))
        self.avgpool_gemap = nn.AvgPool2d((1, 6))
        self.avgpool_ComParE = nn.AvgPool2d((1, 32))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        #print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        if x.size(3) == 3:
            x = self.avgpool(x)
        if x.size(3) == 11:
            x = self.avgpool_gemap(x)
        if x.size(3) == 32:
            x = self.avgpool_ComParE(x)
            #x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)
        
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def _make_liftering(N, Q):
    return 1 + 0.5*Q*np.sin(np.pi*np.arange(N)/Q).astype(np.float32)


def _make_dct(input_dim, output_dim, inv, normalize):
    if normalize:
        norm = 'ortho'
    else:
        norm = None
    if inv:
        C = idct(np.eye(input_dim), type=2, norm=norm, overwrite_x=True)
    else:
        C = dct(np.eye(input_dim), type=2, norm=norm, overwrite_x=True)
    return C[:,:output_dim].astype(np.float32)

class LDE(nn.Module):
    def __init__(self, D, input_dim, with_bias=False, distance_type='norm', network_type='att', pooling='mean', regularization=None):
        super(LDE, self).__init__()
        self.dic = nn.Parameter(torch.randn(D, input_dim))
        nn.init.uniform_(self.dic.data, -1, 1)
        self.wei = nn.Parameter(torch.ones(D))
        if with_bias:
            self.bias = nn.Parameter(torch.zeros(D))
        else:
            self.bias = 0
        assert distance_type == 'norm' or distance_type == 'sqr'
        if distance_type == 'norm':
            self.dis = lambda x: torch.norm(x, p=2, dim=-1)
        else:
            self.dis = lambda x: torch.sum(x**2, dim=-1)
        assert network_type == 'att' or network_type == 'lde'
        if network_type == 'att':
            self.norm = lambda x: F.softmax(-self.dis(x) * F.log_softmax(self.wei, dim = -1) + self.bias, dim = -2)
        else:
            self.norm = lambda x: F.softmax(-self.dis(x) * (self.wei ** 2) + self.bias, dim = -1)
        assert pooling == 'mean' or pooling == 'mean+std'
        self.pool = pooling
        if regularization is None:
            self.reg = None
        else:
            raise NotImplementedError()

    def forward(self, x):
        r = x.view(x.size(0), x.size(1), 1, x.size(2)) - self.dic
        w = self.norm(r).view(r.size(0), r.size(1), r.size(2), 1)
        w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-9) #batch_size, timesteps, component
        if self.pool == 'mean':
            x = torch.sum(w * r, dim=1) 
        else:
            x1 = torch.sum(w * r, dim=1)
            x2 = torch.sqrt(torch.sum(w * r ** 2, dim=1)+1e-8)
            x = torch.cat([x1, x2], dim=-1)
        return x.view(x.size(0), -1)


class E2E(nn.Module):
    def __init__(self, input_dim, output_dim, Q, D, hidden_dim=128, distance_type='norm', network_type='att', pooling='mean', regularization=None, asoftmax=False, fc2_bias=True):
        super(E2E, self).__init__()
        self.lift = nn.Parameter(torch.from_numpy(1./_make_liftering(input_dim, Q)), requires_grad=False)
        self.dct  = nn.Parameter(torch.from_numpy(_make_dct(input_dim, input_dim, inv=True, normalize=True))) 
        self.res  = resnet34()
        self.pool = LDE(D, 128, distance_type=distance_type, network_type=network_type, pooling=pooling, regularization=regularization, with_bias=False) 
        if pooling=='mean':
            self.fc1  = nn.Linear(128*D, hidden_dim)
        if pooling=='mean+std':
            self.fc1  = nn.Linear(256*D, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim, bias=fc2_bias)
        self.asoftmax = asoftmax

    def forward_encoder(self, x):
        if isinstance(x, tuple):
            x = inp[0]
        #x = inp[0]
        x = x * self.lift
        x = F.linear(x, self.dct)
        x = self.res(x)
        return x
    
    def forward_decoder(self,x):
        x = self.pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #return x
        return F.log_softmax(x, dim=-1)
        
    def forward(self, inp):
        x = self.forward_encoder(inp)
        x = self.forward_decoder(x)
        return x

    def predict_feats(self, x):
        #if isinstance(x, tuple):
        #    x = self.forward_encoder(x)
        #else:
        x = self.forward_encoder(x)
        x = self.pool(x)
        if type(x) is tuple:
            x = x[0]
        feats = self.fc1(x)
        x = self.bn1(feats)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x, feats



class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte().detach()
        #index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.01*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp().detach()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


def load_pretrained_model(pretrained, pretrained_model, input_dim=23):
    if pretrained:
        print(f'loading pretrained_model {pretrained_model}')
        A=torch.load(pretrained_model, lambda a,b:a)
        if 'state_dict' in A:
            A = A['state_dict']

        A = {k.replace('model.',''):v for k,v in A.items()}
        final_layer_size = A['fc2.weight'].shape[0]
        if 'fc2.bias' in A:
            model=E2E(input_dim=input_dim,Q=22,output_dim=final_layer_size,
                      D=32,hidden_dim=400,asoftmax=False)
            model.load_state_dict(A)
        else:
            model=E2E(input_dim=input_dim,Q=22,output_dim=final_layer_size,
                        D=32,hidden_dim=400,asoftmax=False, fc2_bias=False)
            model.load_state_dict(A)
    else:
        final_layer_size = 12872
        model=E2E(input_dim=input_dim,Q=22,output_dim=final_layer_size,
                  D=32,hidden_dim=400,asoftmax=False)

    return model

    
class _XvectorModel(nn.Module):
    
    def __init__(self,pretrained, pretrained_model,output_channels = 4):        
        super().__init__()
        self.model = load_pretrained_model(pretrained, pretrained_model)
         
        fc2 = nn.Linear(400, output_channels)
        self.model.fc2 = fc2
        
    def forward(self, x):
        return self.model(x)

    def predict_feats(self, x):
        return self.model.predict_feats(x[0])


class _XvectorModel_ArcLoss(nn.Module):
    '''followed loss function from https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/blob/master/loss_functions.py
    '''
    
    def __init__(self,pretrained, pretrained_model,output_channels=4, input_dim=23, unnorm_feats=False):        
        super().__init__()
        self.model = load_pretrained_model(pretrained, pretrained_model, input_dim)
        self.unnorm_feats = unnorm_feats
        if unnorm_feats:
            self.model.norm_layer = nn.BatchNorm1d(input_dim)            
        
        fc2 = nn.Linear(400, output_channels, bias=False)
        self.model.fc2 = fc2
        #self.dropout = nn.Dropout(p=0.5)
        self.s = 64
        self.m = 0.5
        self.eps = 1e-7

    def forward(self, x, labels):
        if isinstance(x, tuple):
            x = x[0]
        if self.unnorm_feats:
            x = x.permute(0,2,1) 
            x = self.model.norm_layer(x)
            x = x.permute(0,2,1)
        x = self.model.forward_encoder(x)
        loss = self.forward_decoder(x, labels, training=True)
        return loss

    def forward_decoder(self,x, labels=None, training=False):
        x = self.model.pool(x)
        feats = self.model.fc1(x)
        x = self.model.bn1(feats)
        #x = self.dropout(x)
        x = F.relu(x)

        if training:
            ## using the arc loss
            assert len(x) == len(labels)
            assert torch.min(labels) >= 0

            for W in self.model.fc2.parameters():
                W = F.normalize(W, p=2, dim=1)                
            x = F.normalize(x, p=2, dim=1)
            wf = self.model.fc2(x)            
    
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)
        else:
            x = self.model.fc2(x)
            return x, feats
            #return F.log_softmax(x, dim=-1), feats

    def predict_feats(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if self.unnorm_feats:
            x = x.permute(0,2,1) 
            x = self.model.norm_layer(x)
            x = x.permute(0,2,1)
        x = self.model.forward_encoder(x)
        preds, feats = self.forward_decoder(x, labels=None, training=False)
        return preds, feats


class _XvectorModel_Regression(nn.Module):
    ''' This function can be used for both classification and regression. It just gives last layer activations and you can calculate the loss by yourself '''    
    def __init__(self,pretrained, pretrained_model,output_channels=4, input_dim=23, unnorm_feats=False):        
        super().__init__()
        self.model = load_pretrained_model(pretrained, pretrained_model, input_dim)
        self.unnorm_feats = unnorm_feats
        if unnorm_feats:
            self.model.norm_layer = nn.BatchNorm1d(input_dim)            
        
        self.no_classes = output_channels
        fc2 = nn.Linear(400, self.no_classes, bias=False)
        self.model.fc2 = fc2
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, labels):
        if isinstance(x, tuple):
            x = x[0]
        if self.unnorm_feats:
            x = x.permute(0,2,1) 
            x = self.model.norm_layer(x)
            x = x.permute(0,2,1)
        x = self.model.forward_encoder(x)
        out, feats = self.forward_decoder(x, labels, training=True)
        return out

    def forward_decoder(self,x, labels=None, training=False):
        x = self.model.pool(x)
        feats = self.model.fc1(x)
        x = self.model.bn1(feats)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.model.fc2(x)
        #x = torch.sigmoid(x)
        return x, feats

    def predict_feats(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if self.unnorm_feats:
            x = x.permute(0,2,1) 
            x = self.model.norm_layer(x)
            x = x.permute(0,2,1)
        pre_MHA_feats = self.model.forward_encoder(x)
        
        preds, feats = self.forward_decoder(pre_MHA_feats, labels=None, training=False)
        return [preds, feats, pre_MHA_feats]


def bilstm(input_channel):
    lstm = nn.LSTM(input_size=input_channel,
            hidden_size=256,
            num_layers=6,
            bidirectional=True,
            batch_first = True)
    return lstm


class E2E_BiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, Q, D, hidden_dim=128, distance_type='norm', network_type='att', pooling='mean', regularization=None, asoftmax=False, fc2_bias=True):
        super().__init__()
        self.lift = nn.Parameter(torch.from_numpy(1./_make_liftering(input_dim, Q)), requires_grad=False)
        self.dct  = nn.Parameter(torch.from_numpy(_make_dct(input_dim, input_dim, inv=True, normalize=True)))
        #self.res  = resnet34()
        self.bilstm = bilstm(input_dim)
        self.pool = LDE(D, 512, distance_type=distance_type, network_type=network_type, pooling=pooling, regularization=regularization, with_bias=False)
        if pooling=='mean':
            self.fc1  = nn.Linear(512*D, hidden_dim)
        if pooling=='mean+std':
            self.fc1  = nn.Linear(256*D, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim, bias=fc2_bias)
        self.asoftmax = asoftmax

    def forward_encoder(self, x):
        if isinstance(x, tuple):
            x = inp[0]
        #x = inp[0]
        x = x * self.lift
        x = F.linear(x, self.dct)
        x = self.bilstm(x)
        x = x[0]
        return x


class _BiLSTM_Model_Regression(nn.Module):
    ''' This function can be used for both classification and regression. It just gives last layer activations and you can calculate the loss by yourself '''
    def __init__(self,pretrained, pretrained_model,output_channels=4, input_dim=23, unnorm_feats=False):
        super().__init__()
        final_layer_size = output_channels
        self.model=E2E_BiLSTM(input_dim=input_dim,Q=22,output_dim=final_layer_size,
                  D=16,hidden_dim=400,asoftmax=False)

        self.unnorm_feats = unnorm_feats
        if unnorm_feats:
            self.model.norm_layer = nn.BatchNorm1d(input_dim)

        self.no_classes = output_channels
        fc2 = nn.Linear(400, self.no_classes, bias=True)
        self.model.fc2 = fc2

    def forward(self, x, labels):
        if isinstance(x, tuple):
            x = x[0]
        if self.unnorm_feats:
            x = x.permute(0,2,1)
            x = self.model.norm_layer(x)
            x = x.permute(0,2,1)
        x = self.model.forward_encoder(x)
        out, feats = self.forward_decoder(x, labels, training=True)
        return out

    def forward_decoder(self,x, labels=None, training=False):
        x = self.model.pool(x)
        feats = self.model.fc1(x)
        x = self.model.bn1(feats)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.model.fc2(x)
        #x = torch.sigmoid(x)
        return x, feats

    def predict_feats(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if self.unnorm_feats:
            x = x.permute(0,2,1)
            x = self.model.norm_layer(x)
            x = x.permute(0,2,1)
        x = self.model.forward_encoder(x)
        preds, feats = self.forward_decoder(x, labels=None, training=False)
        return preds, feats


class _XvectorModel_MultiTarget(nn.Module):
    
    def __init__(self,pretrained, pretrained_model, output_channels1, output_channels2):        
        super().__init__()
        self.model = load_pretrained_model(pretrained, pretrained_model)

        self.fc2_1 = nn.Linear(400, output_channels1)
        self.fc2_2 = nn.Linear(400, output_channels2)
        
        #self.model.fc2 = fc2
        
    def forward(self, inp):
        x = self.model.forward_encoder(inp)    
        preds, feats = self.forward_decoder(x)
        return preds

    def forward_decoder(self,x):
        x = self.model.pool(x)
        feats = self.model.fc1(x)
        x = self.model.bn1(feats)
        x = F.relu(x)
        out1 = self.fc2_1(x)
        out2 = self.fc2_2(x)
        return (F.log_softmax(out1, dim=-1), F.log_softmax(out2, dim=-1)), feats

    def predict_feats(self, inp):
        x = self.model.forward_encoder(inp)
        preds, feats = self.forward_decoder(x)
        return preds, feats


class _XvectorModel_MultiTarget_2TargetSpecificLayers(nn.Module):
    
    def __init__(self,pretrained, pretrained_model, output_channels1, output_channels2):        
        super().__init__()
        self.model = load_pretrained_model(pretrained, pretrained_model)

        self.fc2_1 = nn.Linear(400, 400)
        self.fc3_1 =  nn.Linear(400, output_channels1)
        self.fc2_2 = nn.Linear(400, 400)
        self.fc3_2 = nn.Linear(400, output_channels2)
        
        #self.model.fc2 = fc2
        
    def forward(self, inp):
        x = self.model.forward_encoder(inp)    
        preds, feats = self.forward_decoder(x)
        return preds

    def forward_decoder(self,x):
        x = self.model.pool(x)
        feats = self.model.fc1(x)
        x = self.model.bn1(feats)
        x = F.relu(x)
        out1 = self.fc2_1(x)
        out1 = F.relu(out1)
        out1 = self.fc3_1(out1)

        out2 = self.fc2_2(x)
        out2 = F.relu(out2)
        out2 = self.fc3_2(out2)
        return (F.log_softmax(out1, dim=-1), F.log_softmax(out2, dim=-1)), feats

    def predict_feats(self, inp):
        x = self.model.forward_encoder(inp)
        preds, feats = self.forward_decoder(x)
        return preds, feats


class _XvectorGemapsModel(nn.Module):
    
    def __init__(self, output_channels = 4):        
        super().__init__()
        self.model=E2E(input_dim=88,Q=22,output_dim=output_channels,
                  D=32,hidden_dim=400,asoftmax=False)
        
    def forward(self, x):
        return self.model(x)

    
class _GRL(nn.Module):
    def __init__(self, no_aux, D=32,distance_type='norm', network_type='att', pooling='mean', regularization=None):
        super().__init__()
        self.pool = LDE(D, 128, distance_type=distance_type, network_type=network_type, pooling=pooling, regularization=regularization, with_bias=False) 
        if pooling=='mean':
            self.fc1  = nn.Linear(128*D, 400)
        if pooling=='mean+std':
            self.fc1  = nn.Linear(256*D, 400)        
        self.bn1 = nn.BatchNorm1d(400)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(400,no_aux)
    def forward(self, x):
        x = self.pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


_ADD = _GRL
    
   
class _OneLayerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_dim)

    def forward(self, x, labels=None):
        if isinstance(x, tuple):
            x = x[0]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def predict_feats(self, x):
        if isinstance(x, tuple):
            x = x[0]
        x = self.forward(x)
        return x, x


