import numpy as np
import math
import random
from typing import List

# Define the Neuron class globally
class Neuron:
    def __init__(self, pos_range=10):
        self.value = 0.5
        # Randomly generate a 2D position
        self.distance = random.uniform(-pos_range, pos_range)
        # Weights to other neurons
        self.weights = {}  # key: target neuron, value: weight
        # Layer
        self.layer = -1  # infinity means unassigned
        # Input vectors
        self.input_vector = []  # Neurons that feed into this neuron
        self.output_vector = [] # Neurons that this neuron feeds into


# Function to create a Neuron object
def create_neuron(pos_range):
    return Neuron(pos_range)

# ------------------------------------------------------------------

# Activate function (for single layer)
def activate(activated_neurons: List[Neuron], target_neurons: List[Neuron], training=False):
    """
    Activate a group of target neurons based on nearby active neurons
    """
    for target_neuron in target_neurons:
        total_contribution = 0.0
        active_neurons = 0
        """
        Activate each neuron based on nearby layers
        """
        for neuron in activated_neurons:
            # Calculate distance between neuron and target_neuron
            dx = target_neuron.distance - neuron.distance
            distance = abs(dx)
            # Get weight
            weight = target_neuron.weights.get(neuron, 1)
            total_contribution += neuron.value * ((1+distance)**-1) * weight
            active_neurons += 1
            # Update layer relationship
            target_neuron.layer = neuron.layer + 1
            # Store input relationship
            if(training and neuron not in target_neuron.input_vector):
                target_neuron.input_vector.append(neuron)
            if(training and target_neuron not in neuron.output_vector):
                neuron.output_vector.append(target_neuron)
        # If no neuron contributes, keep old value
        if active_neurons == 0:
            average_value = target_neuron.value
        else:
            average_value = math.tanh(total_contribution / active_neurons)
        target_neuron.value = average_value

# Forward propagation function (for multiple layers)
def forward_propagate(all_neurons, input_neurons,  config, training=False):
    activated_neurons = input_neurons
    for neuron in input_neurons:
        neuron.layer = 0  # Ensure input neurons have layer 0
    # Remaining neurons that are not yet activated
    unactivated_neurons = [n for n in all_neurons if n not in input_neurons]
    unactivated_neurons.sort(key=lambda n: n.distance)
    result = np.array_split(unactivated_neurons, config["layer"])
    for i in range(config["layer"]):
        activate(activated_neurons, result[i].tolist(), training)
        activated_neurons += result[i].tolist()

# ------------------------------------------------------------------

# Conclusion function
def conclusion(neurons, weight_judge, training):
    total = 0.0
    weight_sum = 0.0
    for neuron in neurons:
        if training:
            w = weight_judge.get(neuron, 1)
            total += neuron.value * w
            weight_sum += w
        else :
            w = weight_judge.get(neuron, 0)
            total += neuron.value * weight_judge.get(neuron, 0)
            weight_sum += w
    return total / weight_sum if weight_sum != 0 else 0.5


# Adjust weights and neuron positions by layers
def adjust_neurons(neuron: Neuron, error: float, learning_rate=0.1, pos_rate=0.2):
    for upper_layer_neuron in neuron.input_vector:
        total_distance = 0.0
        if neuron.layer != -1:
            dx = neuron.distance - upper_layer_neuron.distance
            distance = abs(dx)
            # Adjust weights and neuron positions by layers
            old_weight = neuron.weights.get(upper_layer_neuron, 0.1)
            # Recursively adjust the upper layer neuron
            adjust_neurons(upper_layer_neuron, error*old_weight , learning_rate, pos_rate)
            # Adjust weight (gradient-like rule)
            new_weight = old_weight - learning_rate * upper_layer_neuron.value * error * ((1+distance)**-1)
            neuron.weights[upper_layer_neuron] = new_weight
            # Adjust position
            total_distance += (pos_rate * error * ((1+distance)**-2) * old_weight * upper_layer_neuron.value)
        else:
            influence = ((1+distance)**-1)
            # Update position towards origin
            neuron.distance += pos_rate * error * influence
        neuron.distance += total_distance



# Initialize model
def init_model(config):
    input_neurons = [Neuron(config["pos_range"]) for _ in range(config["num_input"])]
    hidden_neurons = [Neuron(config["pos_range"]) for _ in range(config["num_hidden"])]
    all_neurons = input_neurons + hidden_neurons
    for n in input_neurons:
        n.value = 1.0
        n.layer = 0 # input layer (origin point)
        n.position = (0.0, 0.0) # center position
    for n in hidden_neurons:
        n.value = 0
        n.layer = -1 # hidden layer

    return {
        "input": input_neurons,
        "hidden": hidden_neurons,
        "all": all_neurons,
        "neurons": all_neurons,
    }


# Training for one epoch
def train_one_epoch(model, dataset, config):
    input_neurons = model["input"]
    hidden_neurons = model["hidden"]
    all_neurons = model["all"]
    # Initialize weight judge
    weight_judge = {}
    # Iterate through dataset
    for inputs, target in dataset:
        # 1. input values
        for neuron, value in zip(input_neurons, inputs):
            neuron.value = value
        # 2. forward activation
        forward_propagate(hidden_neurons, input_neurons, config, training=True)
        # 3. compute output
        activated_neurons = [n for n in all_neurons if n.layer != -1]
        average_value = conclusion(activated_neurons, weight_judge, training=True)
        print(f"Target: {target:.4f}, Predicted: {average_value:.4f}")
        # 4. backpropagation
        error = target - average_value
        for neuron in activated_neurons:
            weight_judge[neuron] = weight_judge.get(neuron, 0) - error * neuron.value
            # Adjust neuron weights and positions
            adjust_neurons(
                neuron,
                error,
                learning_rate=config["learning_rate"],
                pos_rate=config["pos_rate"]
            )
        for neuron in hidden_neurons:
            neuron.output_vector = []
            neuron.input_vector = []
            neuron.layer = -1
    return weight_judge
