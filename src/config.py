# config.py
import random

# Load configuration settings
def load_config():
    return {
        "epochs": 40,           # number of training epochs
        "batch_size": 10,       # size of each training batch
        "num_input": 15,        # number of input neurons (feature size)
        "pos_range": 1,         # position range for neurons
        "num_hidden": 60,       # number of hidden neurons
        "seed": 26,             # random seed for reproducibility
        "learning_rate": 0.2,   # learning rate for weight adjustment
        "pos_rate": 0.05,       # position adjustment rate
        "layer": 5              # number of layers in the network
    }

def set_seed(config):
    random.seed(config["seed"])