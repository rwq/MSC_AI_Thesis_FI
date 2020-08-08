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
        
        # self.outputs = []
        # _ = self.model(x)
        
        return self.model(x)
    

# class PerceptualLoss(nn.Module):
    
#     def __init__(self, loss_network, epoch=0, n_epochs=None, scheme=None, include_input=False):
#         super(PerceptualLoss, self).__init__()
#         self.n_epochs = n_epochs
#         self.scheme = scheme
#         self.epoch = epoch
#         self.loss_network = loss_network
#         self.use_inp = include_input

#         if scheme == 'lp-constant':
#             self.weights = self.constant()
#         elif scheme == 'lp-cosine':
#             self.weights = self.cosine()
#         elif scheme == 'lp-random':
#             self.weights = self.random()
#         else:
#             print('USING CONSTANT WEIGHTS FOR MSPL')
#             self.weights = self.constant()

    
#     def forward(self, y_hat, y):

#         y_cat = torch.cat([y_hat, y], axis=0)

#         loss_out = self.loss_network(y_cat)

#         losses = torch.zeros(len(loss_out), dtype=torch.float64)

#         # if self.use_inp:
#         #     losses[0] = (y_hat-y).pow(2).mean().sqrt()

#         for i, loss_tensor in enumerate(loss_out):
#             feat_pred_layer, feat_true_layer = loss_tensor.chunk(chunks=2, dim=0)
#             losses[i] = (feat_pred_layer-feat_true_layer).pow(2).mean().sqrt()
        
#         return losses.sum()

#     def next_epoch(self):
#         self.epoch += 1
#         if self.scheme == 'lp-constant':
#             self.weights = self.constant()
#         elif self.scheme == 'lp-cosine':
#             self.weights = self.cosine()
#         elif self.scheme == 'lp-random':
#             self.weights = self.random()
#         else:
#             print('USING CONSTANT WEIGHTS FOR MSPL')
#             self.weights = self.constant()


#     ## Weighting functions for Multi-stage perceptual loss
#     def constant(self):
#         return torch.ones(4, dtype=torch.float64)/4

#     def random(self):
#         w = torch.rand(4)
#         w.div_(w.sum())
#         w = w.type(torch.float64)
#         return w

#     # def cosine(self):
#     #     w = np.zeros(3)
#     #     w[0] = np.cos(self.epoch*np.pi/self.n_epochs)+1
#     #     w[1] = np.cos(2*(self.epoch-self.n_epochs/2)*np.pi/self.n_epochs)+1
#     #     w[2] = np.cos((self.epoch-self.n_epochs)*np.pi/self.n_epochs)+1
#     #     w /= w.sum() 
#     #     w = torch.from_numpy(w)
#     #     return w

#     def cosine(self):
#         w = np.zeros(4)
        
#         w[0] = np.cos((self.epoch-self.n_epochs)*np.pi/self.n_epochs)+1
#         w[1] = np.cos(5/2*(self.epoch-3*self.n_epochs/5)*np.pi/self.n_epochs)+1
#         w[2] = np.cos(5/2*(self.epoch-2*self.n_epochs/5)*np.pi/self.n_epochs)+1
#         w[3] = np.cos(self.epoch*np.pi/self.n_epochs)+1
        
#         if self.epoch > 40:
#             w[2] = 0
#         if self.epoch < 10:
#             w[1] = 0
    
#         w /= w.sum()
#         w = torch.from_numpy(w)
#         w = w.type(torch.float64)
#         return w


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






# class FlowLoss(nn.Module):
    
#     # TODO dynamisch maken nr frames
    
#     def __init__(self):
#         super(FlowLoss, self).__init__()
        
#         # load flow network
#         self.flow_model = PWCNet()
#         self.flow_model.load_state_dict(torch.load(constants.FP_PWC))
#         self.flow_model.cuda().eval()


    
#     def forward(self, y_hat, y, X):
#         second_frame = int(X.size(1)/2)
#         pixel_wise = (y_hat-y).pow(2).mean(dim=(1,))
        
        
#         # forward flow
#         flow = self.flow_model(y, X[:,second_frame])
#         flow_magnitude = flow.
        
        
#         flow_loss = (pixel_wise * flow_magnitude).mean().sqrt()
        
#         return flow_loss

# class FlowLoss(nn.Module):
    
#     # TODO dynamisch maken nr frames ###!!!
    
#     def __init__(self, quadratic=False):
#         super(FlowLoss, self).__init__()
        
#         # load flow network
#         self.EM = utilities.EdgeMap(quadratic=quadratic)
#         self.l1 = nn.L1Loss(reduction='none')
#         self.alpha = 1/3

#     def forward(self, y_hat, y, X):
#         l1_batch = self.l1(y_hat, y).mean(dim=(1,2,3))
#         em = self.EM(X,y).mean(dim=(1,2))
#         flow_loss = (l1_batch @ (1+em*self.alpha)).mean()
        
#         return flow_loss



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


