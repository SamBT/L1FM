import torch
import torch.nn as nn

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, internal_activation='relu', output_activation=None, input_activation=None):
        super().__init__()
        layers = []
        if input_activation is not None:
            layers.append(input_activation())
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activations[internal_activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation is not None:
            layers.append(activations[output_activation])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
class DeepSetsEncoder(nn.Module):
    def __init__(self, phi, f):
        super().__init__()
        self.phi = phi
        self.f = f

    def forward(self, x, mask=None):  
        # x shape: (batch_size, N, D) # N = number of particle, D = dimension of input
        # mask shape: (batch_size, N), zero means masked (no particle)
        x = self.phi(x)
        if mask is None:
            x = x.sum(dim=-2) # sum over "particles"
        else:
            x = (mask.unsqueeze(-1)*x).sum(dim=-2)
        return self.f(x)
    
class SimCLRModel(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x, **kwargs):
        x = self.encoder(x, **kwargs)
        x = self.projector(x)
        return x
    
    def embed(self,x, **kwargs):
        x = self.encoder(x, **kwargs)
        return x