REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .q_agent import QAgent
REGISTRY["q"] = QAgent