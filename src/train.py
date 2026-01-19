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


# Activate function
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
            layer_activated_neurons.append(target_neuron)
    return (layer_activated_neurons, layer_unactivated_neurons)


# Conclusion function
def conclusion(neurons: List[Neuron]):
    total_value = sum(neuron.value for neuron in neurons)
    average_value = total_value / len(neurons) if neurons else 0
    return average_value


 # Adjust weights and neuron positions by layers
def adjust_neurons(neurons: List[Neuron], average_value: float, target_value: float, learning_rate=0.1, pos_rate=0.05):
    error = target_value - average_value    # difference between desired and current value
    for neuron in neurons:
        # Remove distant neurons from input_vector
        to_remove = []
        for upper_layer_neuron in neuron.input_vector:
            dx = upper_layer_neuron.position[0] - neuron.position[0]
            dy = upper_layer_neuron.position[1] - neuron.position[1]
            distance = (dx**2 + dy**2) ** 0.5
            if distance < 1:
                # Adjust weight (gradient-like rule)
                old_weight = upper_layer_neuron.weights.get(neuron, 1)
                new_weight = old_weight + learning_rate * upper_layer_neuron.value * error * (1 - distance)
                upper_layer_neuron.weights[neuron] = new_weight
                # Adjust position slightly (closer neurons contribute more)
                # Update position
                upper_layer_neuron.position = (
                    upper_layer_neuron.position[0]
                    + pos_rate * error * upper_layer_neuron.value * old_weight * (1.0 - distance) * dx,

                    upper_layer_neuron.position[1]
                    + pos_rate * error * upper_layer_neuron.value * old_weight * (1.0 - distance) * dy
                )
            else:
                to_remove.append(upper_layer_neuron)
        for upper in to_remove:
            # Remove distant neurons from input_vector and weights
            neuron.input_vector.remove(upper)
            # Also remove weight connection
            if neuron in upper.weights:
                del upper.weights[neuron]
        # Reset neuron layer for next iteration
        neuron.layer = float('inf')