import torch
import torch.nn.functional as F

dropout_prob = 0




class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # self.dropout_prob = dropout_prob
        self.encoder = torch.nn.Linear(input_dim, hidden_dim[0])
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim[i], hidden_dim[i+1]) for i in range(len(hidden_dim)-1)])
        self.classifier = torch.nn.Linear(hidden_dim[-1], output_dim)
        # self.dropout = torch.nn.Dropout(p=self.dropout_prob)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.encoder(x))
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
                # x = self.dropout(x)
            logits = self.classifier(x) 
        else:
            x = F.relu(F.linear(x, weight=params['encoder.weight'], bias=params['encoder.bias']))
            for i, layer in enumerate(self.hidden_layers):
                x = F.relu(F.linear(x, weight=params[f'hidden_layers.{i}.weight'], bias=params[f'hidden_layers.{i}.bias']))
                # x = self.dropout(x)
            logits = F.linear(x, weight=params['classifier.weight'], bias=params['classifier.bias'])
        return logits
