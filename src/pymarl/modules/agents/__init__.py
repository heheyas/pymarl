REGISTRY = {}

from .rnn_agent import MLPAgent, RNNAgent, CNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["cnn"] = CNNAgent