import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn


'''Copied from https://github.com/wangleiofficial/label-smoothing-pytorch/blob/main/label_smoothing.py
'''

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = CrossEntropyLoss().ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)



#class DiceLoss(nn.Module):
#    '''Copied from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
#    '''
#    def __init__(self, weight=None, size_average=True):
#        super(DiceLoss, self).__init__()
#
#    def forward(self, inputs, targets, smooth=1):
#        
#        #comment out if your model contains a sigmoid or equivalent activation layer
#        #inputs = F.sigmoid(inputs)       
#        
#        #flatten label and prediction tensors
#        inputs = inputs.view(-1)
#        targets = targets.view(-1)
#        
#        intersection = (inputs * targets).sum()                            
#        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#        
#        return 1 - dice
