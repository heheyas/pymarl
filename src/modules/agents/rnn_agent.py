import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    
class MLPAgent(RNNAgent):
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        q = self.fc2(x)
        return q, hidden_state

class CNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        
        n_input_channels = input_shape[0]
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
        
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(n_input_channels, 25, 5, padding="same")), nn.LeakyReLU(),
            init_(nn.Conv2d(25, 25, 3, padding="same")), nn.LeakyReLU(),
            init_(nn.Conv2d(25, 25, 3, padding="valid")), nn.LeakyReLU(), 
            nn.Flatten(),
        )
        
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(torch.zeros(*input_shape)).float()
            ).shape[1]
            
        self.linear = nn.Sequential(
            init_(nn.Linear(n_flatten, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, args.rnn_hidden_size)),
            nn.LeakyReLU()
        )
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        
        self.critic_linear = init_(nn.Linear(args.rnn_hidden_size, 1))
        
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = self.linear(self.cnn(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_state)
        h = self.rnn(x, h_in)
        q = self.critic_linear(h)
        return q, h
        