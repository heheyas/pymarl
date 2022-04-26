REGISTRY = {}

from .rnn_agent import RNNAgent, CNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["cnn"] = CNNAgent