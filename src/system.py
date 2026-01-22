import math
import random
from typing import List

# Define the Neuron class globally
class Neuron:
    def __init__(self, pos_range=10):
        self.value = 0.5
        # Randomly generate a 2D position
        self.position = (random.uniform(-pos_range, pos_range),
                         random.uniform(-pos_range, pos_range))
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
    layer_activated_neurons = list()
    layer_unactivated_neurons = list()
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
            dx = target_neuron.position[0] - neuron.position[0]
            dy = target_neuron.position[1] - neuron.position[1]
            distance = (dx**2 + dy**2) ** 0.5
            # Get weight
            weight = target_neuron.weights.get(neuron, 1)
            total_contribution += neuron.value * ((1+distance)**-1) * weight
            active_neurons += 1
            # Update layer relationship
            target_neuron.layer = max(target_neuron.layer, neuron.layer + 1)
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
        # Only add to actived layer if activated
        if active_neurons > 0:
            layer_activated_neurons.append(target_neuron)
        else:
            layer_unactivated_neurons.append(target_neuron)
    return (layer_activated_neurons, layer_unactivated_neurons)

# Forward propagation function (for multiple layers)
def forward_propagate(all_neurons, input_neurons,  config, training=False):
    activated_neurons = input_neurons
    for neuron in input_neurons:
        neuron.layer = 0  # Ensure input neurons have layer 0
    # Remaining neurons that are not yet activated
    unactivated_neurons = [n for n in all_neurons if n not in input_neurons]
    
    """ A fucking Error here: config is passed instead of max_layer """
    max_layer = config.get("pos_range",3)
    for i in range(1, max_layer):
        # Select neurons in the current "layer" (distance from center <= i)
        layer_neurons = [
            n for n in unactivated_neurons
            if (max(abs(n.position[0]), abs(n.position[1]))) <= i
        ]
        if not layer_neurons:
            continue
        # Activate neurons
        layer_activated_neurons, _ = activate(activated_neurons, layer_neurons, training)
        # Update activated list
        activated_neurons += layer_activated_neurons
        # Remove activated neurons from unactivated list
        for n in layer_activated_neurons:
            unactivated_neurons.remove(n)

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
    total_x = 0.0
    total_y = 0.0
    for upper_layer_neuron in neuron.input_vector:
        if neuron.layer != -1:
            dx = neuron.position[0] - upper_layer_neuron.position[0]
            dy = neuron.position[1] - upper_layer_neuron.position[1]
            distance = (dx**2 + dy**2) ** 0.5
            # Adjust weights and neuron positions by layers
            old_weight = neuron.weights.get(upper_layer_neuron, 0.1)
            # Recursively adjust the upper layer neuron
            adjust_neurons(upper_layer_neuron, error*old_weight , learning_rate, pos_rate)
            # Adjust weight (gradient-like rule)
            new_weight = old_weight - learning_rate * upper_layer_neuron.value * error * ((1+distance)**-1)
            neuron.weights[upper_layer_neuron] = new_weight
            # Adjust position
            total_x += 2 * (pos_rate * error * ((1+distance)**-2) * old_weight * upper_layer_neuron.value * dx)
            total_y += 2 * (pos_rate * error * ((1+distance)**-2) * old_weight * upper_layer_neuron.value * dy)
        else:
            # For input layer neurons, adjust position towards origin
            dx = -neuron.position[0]
            dy = -neuron.position[1]
            distance = math.sqrt(dx*dx + dy*dy) + 1e-6
            # Unit vector towards origin
            ux = dx / distance
            uy = dy / distance
            influence = ((1+distance)**-1)
            # Update position towards origin
            neuron.position = (
                neuron.position[0] + pos_rate * error * influence * ux,
                neuron.position[1] + pos_rate * error * influence * uy
            )
        neuron.position = (
            neuron.position[0] + min(total_x, 0.2),
            neuron.position[1] + min(total_y, 0.2)
        )



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
