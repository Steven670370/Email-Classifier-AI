# config.py
from random import random

# Load configuration settings
def load_config():
    return {
        "epochs": 50,           # number of training epochs
        "batch_size": 32,       # size of each training batch
        "num_input": 10,        # number of input neurons
        "num_output": 1,        # number of output neurons
        "pos_range": 10,        # position range for neurons
        "seed": 42,             # random seed for reproducibility
        "learning_rate": 0.1,   # learning rate for weight adjustment
    }

def set_seed(config):
    random.seed(config["seed"])