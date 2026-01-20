# config.py
from random import random

# Load configuration settings
def load_config():
    return {
        "epochs": 20,           # number of training epochs
        "num_input": 5,         # number of input neurons
        "num_output": 1,        # number of output neurons
        "pos_range": 5,         # position range for neurons
        "seed": 42,             # random seed for reproducibility
        "learning_rate": 0.1,   # learning rate for weight adjustment
    }

def set_seed(config):
    random.seed(config["seed"])