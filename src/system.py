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
        self.value = 0.5              # activation value
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
    # 1~layer_count: hidden
    # layer_count+1: output 
    all_neurons = [[] for _ in range(layer_count + 2)]
    activated_neurons = []

    # Create input neurons
    for _ in range(num_input):
        neuron = Neuron(pos_range, layer=0, rng=rng) # Fixed random number
        all_neurons[0].append(neuron)
        activated_neurons.append(neuron)

    # Create hidden neurons
    per_layer = num_hidden // layer_count+1
    count = per_layer * layer_count - num_hidden
    for i in range(1, layer_count):
        count -= 1
        this_layer_neurons = []
        for _ in range(per_layer if count < 0 else per_layer - 1):
            neuron = Neuron(pos_range, layer=i, rng=rng)
            for activated_neuron in activated_neurons:
                neuron.weights[activated_neuron] = random.uniform(0.5, 1)
            all_neurons[i].append(neuron)
            this_layer_neurons.append(neuron)
        activated_neurons = this_layer_neurons

    # Create output neuron
    output_neuron = Neuron(pos_range, layer=float('inf'), rng=rng)
    output_neuron.distance = float('inf')

    all_neurons[layer_count+1].append(output_neuron)
    for activated_neuron in activated_neurons:
        output_neuron.weights[activated_neuron] = random.uniform(0.5, 1)

    model = {
        "neurons": all_neurons
    }
    return model

def forward_propagation(model):
    neurons = model["neurons"]
    for layer in neurons:
        for neuron in layer:
            if neuron.layer == 0:
                continue
            else :
                total_input = 0.0
                for target, weight in neuron.weights.items():
                    if neuron.distance != float('inf') and target.distance != float('inf'):
                        total_input += target.value * weight * math.log(abs(neuron.distance-target.distance))
                    else:
                        total_input += target.value * weight
                neuron.value = total_input if total_input > 0 else 0.01 * total_input  # ReLU activation

def backward_propagation(model, target_value, learning_rate, pos_rate):
    neurons = model["neurons"]
    output_neuron = neurons[-1][0]
    output_error = output_neuron.value - target_value

    # Update weights and positions for output neuron
    for target, weight in output_neuron.weights.items():
        grad = output_error * 1 if output_neuron.value > 0 else 0.01 * target.error * target.weights[neuron]  
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
                    # target neuron is affected by neuron in forward propagation, so we don't check its value here
                    # only check neuron.value > 0 for ReLU
                    neuron_error += target.error * target.weights[neuron] if neuron.value > 0 else 0.01 * target.error * target.weights[neuron]  
            # accumulate error
            neuron.error = neuron_error
            for target, weight in neuron.weights.items():
                grad = neuron_error * neuron.value * math.log(abs(neuron.distance - target.distance))
                neuron.weight_error[target] = learning_rate * grad
                # Update distance
                distance_grad = neuron_error * weight * neuron.value * (neuron.distance - target.distance)**-1 if neuron.distance != target.distance else 0
                neuron.distance_error += pos_rate * distance_grad
     
    for layer in neurons:
        for neuron in layer:
            for target in neuron.weights:
                neuron.weights[target] -= neuron.weight_error.get(target, 0)
            neuron.distance += neuron.distance_error
            neuron.weight_error = {}
            neuron.distance_error = 0.0
            neuron.error = 0.0


# Return the value of output neuron
def conclusion(model):
    neurons = model["neurons"]
    print(neurons[-1][0].value)
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

        # 2. forward propagation
        forward_propagation(model)

        # 3. backward propagation (in-place update)
        backward_propagation(model, y, learning_rate, pos_rate)

        # 4. loss (optional, for logging/debug)
        y_hat = conclusion(model)
        total_loss += abs(y_hat - y)

    return total_loss / len(batch)


"""
import csv

def init_model(config):
    num_input = config["num_input"]
    num_hidden = config["num_hidden"]
    pos_range = config["pos_range"]
    layer_count = config["layer"]

    all_neurons = [[] for _ in range(layer_count + 2)]
    activated_neurons = []

    # Create input neurons
    for _ in range(num_input):
        neuron = Neuron(pos_range, layer=0, rng=rng)
        all_neurons[0].append(neuron)
        activated_neurons.append(neuron)

    # Create hidden neurons
    per_layer = num_hidden // layer_count+1
    count = per_layer * layer_count - num_hidden
    for i in range(1, layer_count):
        count -= 1
        this_layer_neurons = []
        for _ in range(per_layer if count < 0 else per_layer - 1):
            neuron = Neuron(pos_range, layer=i, rng=rng)
            for activated_neuron in activated_neurons:
                #____________
                neuron.weights[activated_neuron] = random.uniform(0.5, 1)
                #____________
            all_neurons[i].append(neuron)
            this_layer_neurons.append(neuron)
        activated_neurons = this_layer_neurons

    # Create output neuron
    output_neuron = Neuron(pos_range, layer=float('inf'), rng=rng)
    output_neuron.distance = float('inf')
    all_neurons[layer_count+1].append(output_neuron)
    for activated_neuron in activated_neurons:
        #__________
        output_neuron.weights[activated_neuron] = random.uniform(0.5, 1)
        #__________

    # Debug print neuron values
    print("=== INIT MODEL NEURON VALUES ===")
    for i, layer in enumerate(all_neurons):
        print(f"Layer {i}: {[n.value for n in layer]}")

    return {"neurons": all_neurons}

def conclusion(model):
    neurons = model["neurons"]
    print(neurons[-1][0].value)
    return neurons[-1][0].value


def forward_propagation(model):
    neurons = model["neurons"]
    print("=== FORWARD PROP START ===")
    for l_idx, layer in enumerate(neurons):
        for n_idx, neuron in enumerate(layer):
            if l_idx == 0:
                print(f"Layer {l_idx} Neuron {n_idx} value: {neuron.value}")
                continue

            total_input = 0.0
            for target, weight in neuron.weights.items():
                if neuron.distance != float('inf') and target.distance != float('inf'):
                    # 避免除零或非常小，加入 epsilon
                    diff = abs(neuron.distance - target.distance)
                    total_input += target.value * weight * (1.0 / (diff + 1e-6))
                else:
                    total_input += target.value * weight
                    

            # Leaky ReLU 防止全 0
            neuron.value = total_input if total_input > 0 else 0 * total_input

            print(f"Layer {l_idx} Neuron {n_idx} value: {neuron.value}")
    print("=== FORWARD PROP END ===")


def backward_propagation(model, target_value, learning_rate, pos_rate):
    neurons = model["neurons"]
    output_neuron = neurons[-1][0]
    output_error = output_neuron.value - target_value

    # Output neuron weight update
    for target, weight in output_neuron.weights.items():
        grad = output_error * 1 if output_neuron.value >= 0 else 0
        output_neuron.weight_error[target] = learning_rate * grad
        output_neuron.error = output_error

    # Backpropagate to hidden layers
    for l_idx, layer in enumerate(reversed(neurons)):
        if layer == neurons[-1]:
            continue
        for n_idx, neuron in enumerate(layer):
            neuron_error = 0
            for target in neurons[len(neurons) - l_idx]:
                if neuron in target.weights:
                    neuron_error += target.error * target.weights[neuron] if neuron.value >= 0 else 0
            neuron.error = neuron_error
            for target, weight in neuron.weights.items():
                grad = neuron_error * neuron.value
                neuron.weight_error[target] = learning_rate * grad
                distance_grad = neuron_error * weight * neuron.value * (neuron.distance - target.distance)**-2 if neuron.distance != target.distance else 0
                neuron.distance_error += pos_rate * distance_grad

    # Apply updates and reset
    for l_idx, layer in enumerate(neurons):
        for n_idx, neuron in enumerate(layer):
            for target in neuron.weights:
                neuron.weights[target] -= neuron.weight_error.get(target, 0)
            neuron.distance += neuron.distance_error
            neuron.weight_error = {}
            neuron.distance_error = 0.0
            neuron.error = 0.0
            print(f"Layer {l_idx} Neuron {n_idx} value after backward: {neuron.value}")

def train_one_epoch(model, batch, config, log_file="neuron_log.csv"):
    learning_rate = config["learning_rate"]
    pos_rate = config.get("pos_rate", 0.0)

    total_loss = 0.0

    # 打开文件，准备写入
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)

        for sample_idx, (x_values, y) in enumerate(batch):
            # --- INPUT 阶段 ---
            input_layer = model["neurons"][0]
            for neuron, x in zip(input_layer, x_values):

                # ____________________
                neuron.value = 0.5
                # ____________________

            # 写入 INPUT neuron 值
            writer.writerow([f"=== Sample {sample_idx} | Stage: INPUT ==="])
            writer.writerow([n.value for n in input_layer])

            # --- FORWARD 阶段 ---
            forward_propagation(model)

            # 写入 FORWARD neuron 值（全部层）
            writer.writerow([f"=== Sample {sample_idx} | Stage: FORWARD ==="])
            for layer in model["neurons"]:
                writer.writerow([n.value for n in layer])

            # --- BACKWARD 阶段 ---
            backward_propagation(model, y, learning_rate, pos_rate)

            # 写入 BACKWARD neuron 值（全部层）
            writer.writerow([f"=== Sample {sample_idx} | Stage: BACKWARD ==="])
            for layer in model["neurons"]:
                writer.writerow([n.value for n in layer])

            # --- CONCLUSION 阶段 ---
            y_hat = conclusion(model)
            total_loss += abs(y_hat - y)

            writer.writerow([f"=== Sample {sample_idx} | Stage: CONCLUSION ==="])
            writer.writerow([y_hat])

    return total_loss / len(batch)
"""