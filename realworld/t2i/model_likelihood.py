import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalCouplingLayer(nn.Module):
    """
    Conditional Affine Coupling Layer for Conditional RealNVP
    """
    def __init__(self, input_dim, cond_dim, hidden_dim, mask):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer('mask', mask)
        
        in_channels = int(mask.sum()) + cond_dim
        out_channels = (input_dim - int(mask.sum())) * 2
        
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_channels)
        )
        
        self.scale_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x, c, reverse=False):
        x_masked = x * self.mask
        net_input = torch.cat([x_masked[:, self.mask.bool()], c], dim=1)
        params = self.net(net_input)
        s, t = params.chunk(2, dim=1)
        s = torch.tanh(s / 2) * self.scale_factor
        
        s_full = torch.zeros_like(x)
        t_full = torch.zeros_like(x)
        s_full[:, ~self.mask.bool()] = s
        t_full[:, ~self.mask.bool()] = t
        
        exp_neg_s = torch.exp(-s_full)
        z = x_masked + (x * (1 - self.mask) - t_full) * exp_neg_s
        
        log_det = -s_full.sum(dim=1)
        
        return z, log_det

class ConditionalFlow(nn.Module):
    def __init__(self, z_dim, cond_dim, n_layers=4, hidden=128, depth=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.z_dim = z_dim
        
        for i in range(n_layers):
            mask = torch.zeros(z_dim)
            if i % 2 == 0:
                mask[::2] = 1
            else:
                mask[1::2] = 1
                
            self.layers.append(
                ConditionalCouplingLayer(z_dim, cond_dim, hidden, mask)
            )
            
        self.register_buffer('base_loc', torch.zeros(z_dim))
        self.register_buffer('base_scale', torch.ones(z_dim))

    def forward(self, x, c):
        log_det_sum = 0
        z = x
        
        for layer in self.layers:
            z, log_det = layer(z, c)
            log_det_sum += log_det
            
        return z, log_det_sum

    def log_prob(self, target, condition):
        z, log_det = self.forward(target, condition)
        log_prob_z = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        return log_prob_z + log_det


###########################################
#            likelihood model             #
###########################################

class ConceptFlowModel(nn.Module):
    def __init__(self, 
                 num_image_concepts=64,   # zI
                 num_text_concepts=64,    # zT 
                 image_dim=64,            # zI 
                 text_dim=768,            # zT 
                 flow_hidden=128,
                 flow_layers=16,
                 hard_mask_threshold=0.1):
        super().__init__()
        
        self.num_image_concepts = num_image_concepts
        self.num_text_concepts = num_text_concepts
        self.image_dim = image_dim
        self.hard_mask_threshold = hard_mask_threshold

        # Each text concept gets its own conditional flow, all sharing the same image-conditioning space.
        self.flows = nn.ModuleList()
        
        total_cond_input_dim = num_image_concepts * image_dim

        for i in range(num_text_concepts):
            self.flows.append(
                ConditionalFlow(
                    z_dim=text_dim,     
                    cond_dim=total_cond_input_dim, 
                    n_layers=flow_layers,
                    hidden=flow_hidden,
                )
            )
            
        # Learn which image concepts should contribute to each text-concept likelihood term.
        self.mask_logits = nn.Parameter(torch.randn(num_text_concepts, num_image_concepts) * 0.1)

    def get_mask(self, hard=False):
        # Convert learned logits into a soft mask, or threshold it for a hard selection.
        soft_mask = torch.sigmoid(self.mask_logits)
        if hard:
            return (soft_mask > self.hard_mask_threshold).float()
        return soft_mask

    def forward(self, zI, zT):
        batch_size = zI.size(0)
        mask = self.get_mask(hard=False)
        
        total_log_prob = 0
        
        # Score each text concept against the corresponding masked image concepts.
        for i in range(self.num_text_concepts):
            m_i = mask[i].view(1, -1, 1)
            masked_zI = zI * m_i 
            cond_input = masked_zI.view(batch_size, -1)
            target_zT = zT[:, i, :]
            log_prob = self.flows[i].log_prob(target_zT, condition=cond_input)
            
            total_log_prob += log_prob.sum()

        # Return the average negative log-likelihood over the batch and text concepts.
        nll = -total_log_prob / (batch_size * self.num_text_concepts)
        return nll, mask


