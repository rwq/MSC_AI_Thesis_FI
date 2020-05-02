import torch
from torch import nn
import torchvision


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
        
        for layer in layers:
            self.model[layer].register_forward_hook(self.hook)

        torch.cuda.empty_cache()
        
            
    def hook(self, module, input, output):
        self.outputs.append(output)
    
    def forward(self, x):
        '''
        Returns the activation for each of specified layer index in layers
        '''
        
        self.outputs = []
        _ = self.model(x)
        
        return self.outputs
    

class PerceptualLoss(nn.Module):
    
    def __init__(self, loss_network):
        super(PerceptualLoss, self).__init__()

        self.loss_network = loss_network
    
    def forward(self, y_hat, y):
        feat_pred = self.loss_network(y_hat)
        feat_true = self.loss_network(y)
        t_loss = 0

        for feat_pred_layer, feat_true_layer in zip(feat_pred, feat_true):
            t_loss += (feat_pred_layer-feat_true_layer).pow(2).mean().sqrt()
        
        return t_loss