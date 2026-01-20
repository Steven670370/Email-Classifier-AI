import random
from typing import List

# Define the Neuron class globally
class Neuron:
    def __init__(self, pos_range=10):
        self.value = 0
        # Randomly generate a 2D position
        self.position = (random.uniform(-pos_range, pos_range),
                         random.uniform(-pos_range, pos_range))
        # Weights to other neurons
        self.weights = {}  # key: target neuron, value: weight
        # Layer
        self.layer = float('inf')
        # Input vectors
        self.input_vector = []  # Neurons that feed into this neuron


# Function to create a Neuron object
def create_neuron(pos_range):
    return Neuron(pos_range)

# ------------------------------------------------------------------

# Activate function (for single layer)
def activate(activated_neurons: List[Neuron], target_neurons: List[Neuron]):
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
            if distance < 1 and neuron.value != 0:
                # Get weight (default = 1 if not defined)
                weight = neuron.weights.get(target_neuron, 1)
                total_contribution += neuron.value * (1 - distance) * weight
                active_neurons += 1
                # Update layer relationship
                target_neuron.layer = min(
                    target_neuron.layer,
                    neuron.layer + 1
                )
                # Store input relationship
                target_neuron.input_vector.append(neuron)
        # If no neuron contributes, keep old value
        if active_neurons == 0:
            average_value = target_neuron.value
        else:
            average_value = total_contribution / active_neurons
        # Sigmoid activation
        import math
        target_neuron.value = 1 / (1 + math.exp(-average_value))
        # Only add to actived layer if activated
        if active_neurons > 0 and target_neuron.value > 0.5:
            layer_activated_neurons.append(target_neuron)
        else:
            layer_unactivated_neurons.append(target_neuron)
    return (layer_activated_neurons, layer_unactivated_neurons)

# Forward propagation function (for multiple layers)
def forward_propagate(all_neurons, input_neurons, config):
    activated_neurons = input_neurons
    for neuron in input_neurons:
        neuron.layer = 0  # Ensure input neurons have layer=0
    # Remaining neurons that are not yet activated
    unactivated_neurons = [n for n in all_neurons if n not in input_neurons]
    # Maximum layer based on pos_range
    max_layer = config["pos_range"]

    for i in range(1, max_layer + 1):
        # Select neurons in the current "layer" (distance from center <= i)
        layer_neurons = [
            n for n in unactivated_neurons
            if int(max(abs(n.position[0]), abs(n.position[1]))) == i
        ]
        if not layer_neurons:
            continue
        # Activate neurons
        layer_activated_neurons, _ = activate(activated_neurons, layer_neurons)
        # Update activated list
        activated_neurons += layer_activated_neurons
        # Remove activated neurons from unactivated list
        for n in layer_activated_neurons:
            unactivated_neurons.remove(n)

# ------------------------------------------------------------------

# Conclusion function
def conclusion(neurons: List[Neuron]):
    """
    Create a temporary global readout node.
    It is NOT part of the network and will be discarded immediately.
    """
    # virtual global node (not stored)
    total_value = sum(n.value for n in neurons)
    average_value = total_value / len(neurons) if neurons else 0.0
    return average_value


 # Adjust weights and neuron positions by layers
def adjust_neurons(neurons: List[Neuron], average_value: float, target_value: float, learning_rate=0.1, pos_rate=0.05):
    error = target_value - average_value    # difference between desired and current value
    for neuron in neurons:
        if neuron.layer == float('inf'):
            import math
            dx = -neuron.position[0]
            dy = -neuron.position[1]
            distance = math.sqrt(dx*dx + dy*dy) + 1e-6
            # Unit vector towards origin
            ux = dx / distance
            uy = dy / distance
            influence = math.exp(-distance)
            # Update position towards origin
            neuron.position = (
            neuron.position[0] + pos_rate * error * influence * ux,
            neuron.position[1] + pos_rate * error * influence * uy
            )
        else:
            # Remove distant neurons from input_vector
            to_remove = []
            total_x = 0
            total_y = 0
            for upper_layer_neuron in neuron.input_vector:
                dx = upper_layer_neuron.position[0] - neuron.position[0]
                dy = upper_layer_neuron.position[1] - neuron.position[1]
                distance = (dx**2 + dy**2) ** 0.5
                if distance < 1:
                    # Adjust weight (gradient-like rule)
                    old_weight = upper_layer_neuron.weights.get(neuron, 1)
                    new_weight = old_weight + learning_rate * upper_layer_neuron.value * error * (1 - distance)
                    upper_layer_neuron.weights[neuron] = new_weight
                    # Accumulate position adjustments
                    total_x += pos_rate * (1 - distance) * old_weight * upper_layer_neuron.value * dx
                    total_y += pos_rate * (1 - distance) * old_weight * upper_layer_neuron.value * dy
                else:
                    to_remove.append(upper_layer_neuron)
            for upper in to_remove:
                # Remove distant neurons from input_vector and weights
                neuron.input_vector.remove(upper)
                # Also remove weight connection
                if neuron in upper.weights:
                    del upper.weights[neuron]
            # Adjust position slightly (closer neurons contribute more)
            # Update position
            neuron.position = (
                neuron.position[0] + total_x,
                neuron.position[1] + total_y
            )
            # Reset neuron layer for next iteration
            neuron.layer = float('inf')


# Initialize model
def init_model(config):
    input_neurons = [Neuron(config["pos_range"]) for _ in range(config["num_input"])]
    output_neurons = [Neuron(config["pos_range"]) for _ in range(config["num_output"])]
    for n in input_neurons:
        n.layer = 0 # input layer (origin point)
    return {
        "input": input_neurons,
        "output": output_neurons,
    }


# Training for one epoch
def train_one_epoch(model, dataset, config):
    input_neurons = model["input"]
    all_neurons = model["all"]
    # Iterate through dataset
    for inputs, target in dataset:
        # 1. input values
        for neuron, value in zip(input_neurons, inputs):
            neuron.value = value
        # 2. forward activation
        forward_propagate(all_neurons, input_neurons, config)
        # 3. compute output
        average_value = conclusion(all_neurons)
        # 4. backpropagation
        adjust_neurons(
            all_neurons,
            average_value,
            target,
            learning_rate=config["learning_rate"]
        )

# 可视化功能（可选）
