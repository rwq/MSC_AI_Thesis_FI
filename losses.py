import torch
from torch import nn
import torchvision
import numpy as np
import constants
# from code.pytorch_pwc2.PWCNetnew import PWCNet
import utilities


class LossNetwork(nn.Module):
    
    def __init__(self, layers):
        super(LossNetwork, self).__init__()
        
        # reconstruct pretrained VGG
        vgg = torchvision.models.vgg16_bn(pretrained=True)
        
        # delete unnecessary layers
        del vgg.avgpool
        del vgg.classifier
        
        last_layer = max(layers)
        vgg.features = vgg.features[:last_layer+1]
        
        self.model = nn.Sequential(
            *list(vgg.features.children())
        )
        
        # for layer in layers:
        #     self.model[layer].register_forward_hook(self.hook)

        torch.cuda.empty_cache()
        
            
    # def hook(self, module, input, output):
    #     self.outputs.append(output)
    
    def forward(self, x):
        '''
        Returns the activation for each of specified layer index in layers
        '''
        
        return self.model(x)
    




class PerceptualLoss(nn.Module):
    
    def __init__(self, loss_network):
        super(PerceptualLoss, self).__init__()
        self.loss_network = loss_network


    
    def forward(self, y_hat, y):

        y_cat = torch.cat([y_hat, y], axis=0)

        loss_out = self.loss_network(y_cat)

        feat_pred_layer, feat_true_layer = loss_out.chunk(chunks=2, dim=0)
        loss = (feat_pred_layer-feat_true_layer).pow(2).mean().sqrt()
        
        return loss


class FlowLoss(nn.Module):
    
    # TODO dynamisch maken nr frames ###!!!
    
    def __init__(self, alpha=1, quadratic=False):
        super(FlowLoss, self).__init__()
        
        # load flow network
        self.l1 = nn.L1Loss(reduction='none')
        self.alpha = alpha
        self.f1 = 1 if quadratic else 0
        self.f2 = 2 if quadratic else 1

    def forward(self, y_hat, y, X):
        l1_batch = self.l1(y_hat, y).mean(dim=(1,2,3))

        move1 = (y-X[:,self.f1]).pow(2)
        move2 = (y-X[:,self.f2]).pow(2)
        weights = (move1 + move2).mean(dim=(1,2,3))


        flow_loss = (l1_batch @ (1+weights*self.alpha)).mean()
        
        return flow_loss


