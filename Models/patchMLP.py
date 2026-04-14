import torch.nn as nn


class PatchMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=None, activation="ReLU", dropout=0.0):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32, 16]
        self.output_size = int(num_classes)
        self.hidden_size = [int(i * int(input_dim)) for i in hidden_layers]
        activation = str(activation).lower()
        if activation == "LeakyReLu":
            self.act_layer = nn.LeakyReLU()
        elif activation == "tanh":
            self.act_layer = nn.Tanh()
        elif activation == "gelu":
            self.act_layer = nn.GELU()
        else:
            self.act_layer = nn.ReLU()


        self.input = nn.Linear(self.input_dim, self.hidden_size[0])

        self.hidden = nn.ModuleList()

        for i in range(len(hidden_layers) - 1):
            self.hidden.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))

        self.output = nn.Linear(self.hidden_size[-1], self.output_size)    


    def forward(self, x):
        x = self.input(x)
        x = self.act_layer(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            x = self.act_layer(x)
        x = self.output(x)

        return x