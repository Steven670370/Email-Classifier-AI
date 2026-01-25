import numpy as np
import math
import random
from config import load_config, set_seed
from typing import List

config = load_config() 
rng = set_seed(config) # If you want random distance, just delete it


class Neuron:
    _id_counter = 0 # Variable for the Neuron class

    def __init__(self, pos_range, layer=-1, rng=None):
        self.rng = rng or random
        self.distance = self.rng.uniform(-pos_range, pos_range)

        self.layer = layer          # layer index
        self.weight_error = {}      # weight error term for backpropagation
        self.distance_error = 0.0   # distance error for backpropagation
        self.error = 0.0            # error term for training
        self.value = 0.5            # activation value
        self.weights = {}           # weights to other neurons
        # for snapshot
        self.id = Neuron._id_counter
        Neuron._id_counter += 1

def init_model(config):
    num_input = config["num_input"]
    num_hidden = config["num_hidden"]
    pos_range = config["pos_range"]
    layer_count = config["layer"]

    # 0: input
    # 1~layer_count-1: hidden
    # layer_count: output 
    all_neurons = [[] for _ in range(layer_count + 2)]
    activated_neurons = []

    # Create input neurons
    for _ in range(num_input):
        neuron = Neuron(pos_range, layer=0, rng=rng) # Fixed random number
        all_neurons[0].append(neuron)
        activated_neurons.append(neuron)

    # Create hidden neurons
    per_layer = num_hidden // (layer_count-1) + 1
    count = per_layer * (layer_count-1) - num_hidden
    for i in range(1, layer_count):
        count -= 1
        this_layer_neurons = []
        for _ in range(per_layer if count < 0 else per_layer - 1):
            neuron = Neuron(pos_range, layer=i, rng=rng)
            for activated_neuron in activated_neurons:
                neuron.weights[activated_neuron] = random.uniform(-0.5, 0.5)
            all_neurons[i].append(neuron)
            this_layer_neurons.append(neuron)
        activated_neurons = this_layer_neurons

    # Create output neuron
    output_neuron = Neuron(pos_range, layer=layer_count, rng=rng)

    all_neurons[layer_count+1].append(output_neuron)
    for activated_neuron in activated_neurons:
        output_neuron.weights[activated_neuron] = random.uniform(-0.5, 0.5)

    model = {
        "neurons": all_neurons
    }
    return model

def forward_propagation(model):
    neurons = model["neurons"]

    for layer in neurons:
        for neuron in layer:
            if neuron.layer != 0:
                neuron.value = 0.0

    for layer in neurons:
        for neuron in layer:
            if neuron.layer == 0:
                continue
            else :
                total_input = 0.0
                for target, weight in neuron.weights.items():
                    if neuron.distance != float('inf') and target.distance != float('inf'):
                        total_input += target.value * weight * (abs(neuron.distance - target.distance)+1)**-1
                    else:
                        total_input += target.value * weight
                if neuron.layer == neurons[-1][0].layer:
                    neuron.value = 1 / (1 + math.exp(-total_input))
                else :
                    neuron.value = total_input if total_input > 0 else 0.5*total_input  # ReLU activation


def backward_propagation(model, target_value, learning_rate, pos_rate):
    neurons = model["neurons"]
    output_neuron = neurons[-1][0]
    y_hat = output_neuron.value
    output_neuron.error = (y_hat - target_value) * y_hat * (1 - y_hat)

    for layer_idx in range(len(neurons) - 2, -1, -1): # skip output layers
        layer = neurons[layer_idx]
        next_layer = neurons[layer_idx + 1]
    
        for neuron in layer:
            neuron_error = 0
            for next_neuron in next_layer:
                for prev_neuron, weight in next_neuron.weights.items():
                    if prev_neuron is neuron:
                        relu_grad = 1 if neuron.value > 0 else 0.5

                        # target neuron is affected by neuron in forward propagation, so we don't check its value here
                        # only check neuron.value > 0 for ReLU
                        neuron_error += relu_grad * next_neuron.error * next_neuron.weights[neuron] * (abs(next_neuron.distance - neuron.distance)+1)**-1

                        # Update weight for y=v*((1+d)**-2)*weight
                        grad = next_neuron.error * neuron.value * (abs(next_neuron.distance - neuron.distance)+1)**-1
                        next_neuron.weight_error[neuron] = grad

                        # Update distance for y=v*((1+d)**-2)*weight
                        distance_grad = next_neuron.error * next_neuron.weights[neuron] * neuron.value * ((abs(next_neuron.distance - neuron.distance)+1)**-2)
                        sign = 1 if next_neuron.distance > neuron.distance else -1
                        distance_grad *= sign
                        next_neuron.distance_error += distance_grad

            # accumulate error
            neuron.error = neuron_error
                
     
    for layer in neurons:
        for neuron in layer:
            for target in neuron.weights:
                neuron.weights[target] -= learning_rate * neuron.weight_error.get(target, 0)
            neuron.distance += pos_rate * neuron.distance_error
            neuron.weight_error = {}
            neuron.distance_error = 0.0
            neuron.error = 0.0


# Return the value of output neuron
def conclusion(model):
    neurons = model["neurons"]
    return neurons[-1][0].value


def train_one_epoch(model, batch, config):
    learning_rate = config["learning_rate"]
    pos_rate = config.get("pos_rate", 0.0)

    total_loss = 0.0

    for x_values, y in batch:
        # 1. set input layer
        input_layer = model["neurons"][0]
        for neuron, x in zip(input_layer, x_values):
            neuron.value = x
            neuron.distance = 0

        # 2. forward propagation
        forward_propagation(model)

        # 3. backward propagation (in-place update)
        backward_propagation(model, y, learning_rate, pos_rate)

        # 4. loss (optional, for logging/debug)
        y_hat = conclusion(model)
        total_loss += 0.5 * ((y_hat - y)**2)

    return total_loss / len(batch)