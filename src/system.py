import numpy as np
import math
import random
from typing import List

class Neuron:
    def __init__(self, pos_range, layer=float('inf')):
        self.distance = random.uniform(-pos_range, pos_range)
        self.layer = layer          # layer index
        self.weight_error = {}      # weight error term for backpropagation
        self.distance_error = 0.0   # distance error for backpropagation
        self.error = 0.0            # error term for training
        self.value = 0.5            # activation value
        self.weights = {}           # weights to other neurons

def init_model(config):
    num_input = config["num_input"]
    num_hidden = config["num_hidden"]
    pos_range = config["pos_range"]
    layer_count = config["layer"]

    all_neurons = [[]]
    activated_neurons = []

    # Create input neurons
    for _ in range(num_input):
        neuron = Neuron(pos_range, layer=0)
        all_neurons[0].append(neuron)
        activated_neurons.append(neuron)

    # Create hidden neurons
    per_layer = num_hidden // layer_count+1
    count = per_layer * layer_count - num_hidden
    for i in range(1, layer_count):
        count -= 1
        this_layer_neurons = []
        for _ in range(per_layer if count < 0 else per_layer - 1):
            neuron = Neuron(pos_range, layer=i)
            for activated_neuron in activated_neurons:
                neuron.weights[activated_neuron] = random.uniform(-0.5, 0.5)
            all_neurons[i].append(neuron)
            this_layer_neurons.append(neuron)
        activated_neurons = this_layer_neurons

    # Create output neuron
    output_neuron = Neuron(pos_range, layer=float('inf'))
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
            total_input = 0.0
            for target, weight in neuron.weights.items():
                if neuron.distance != float('inf') and target.distance != float('inf'):
                    total_input += target.value * weight * ((neuron.distance-target.distance)**-1)
                else:
                    total_input += target.value * weight
            neuron.value = max(0, total_input)  # ReLU activation

def backward_propagation(model, target_value, learning_rate, pos_rate):
    neurons = model["neurons"]
    output_neuron = neurons[-1][0]
    output_error = output_neuron.value - target_value

    # Update weights and positions for output neuron
    for target, weight in output_neuron.weights.items():
        grad = output_error * output_neuron.value if output_neuron.value > 0 else 0
        output_neuron.weight_error[target] = learning_rate * grad
        output_neuron.error = output_error

    # Backpropagate to hidden layers
    for layer in reversed(neurons):
        if layer == neurons[-1]:
            continue  # skip output layers
        for neuron in layer:
            neuron_error = 0
            for target in neurons[neurons.index(layer)+1]:
                if neuron in target.weights:
                    neuron_error += target.error * target.weights[neuron] if neuron.value > 0 else 0
            neuron.error = neuron_error
            for target, weight in neuron.weights.items():
                grad = neuron_error * neuron.value
                neuron.weight_error[target] = learning_rate * grad
                # Update distance
                distance_grad = neuron_error * weight * neuron.value * (neuron.distance - target.distance)**-2 if neuron.distance != target.distance else 0
                neuron.distance_error += pos_rate * distance_grad

    for layer in neurons:
        for neuron in layer:
            for target in neuron.weights:
                neuron.weights[target] -= neuron.weight_error.get(target, 0)
            neuron.distance += neuron.distance_error
            neuron.weight_error = {}
            neuron.distance_error = 0.0
            neuron.error = 0.0

def conclusion_propagation(model, target_value, learning_rate, pos_rate):
    forward_propagation(model)
    backward_propagation(model, target_value, learning_rate, pos_rate)