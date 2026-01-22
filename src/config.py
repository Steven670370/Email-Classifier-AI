# config.py
import random

# Load configuration settings
def load_config():
    return {
        "epochs": 40,           # number of training epochs
        "batch_size": 10,       # size of each training batch
        "num_input": 20,        # number of input neurons (feature size)
        "pos_range": 3,         # position range for neurons
        "num_hidden": 100,      # number of hidden neurons
        "seed": 25,             # random seed for reproducibility
        "learning_rate": 0.1,   # learning rate for weight adjustment
        "pos_rate": 0.5,        # position adjustment rate
    }

def set_seed(config):
    random.seed(config["seed"])