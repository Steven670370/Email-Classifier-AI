# config.py
import random

# Load configuration settings
def load_config():
    return {
        "epochs": 20,           # number of training epochs
        "batch_size": 10,       # size of each training batch
        "num_input": 15,        # number of input neurons (feature size)
        "pos_range": 5,         # position range for neurons
        "num_hidden": 25,       # number of hidden neurons
        "seed": 10,             # random seed for reproducibility
        "learning_rate": 0.5,   # learning rate for weight adjustment
        "pos_rate": 2,           # position adjustment rate
    }

def set_seed(config):
    random.seed(config["seed"])