from system import forward_propagate, conclusion

def evaluate(model, dataset, config):
    total_error = 0.0
    input_neurons = model["input"]
    all_neurons = model["all"]
    # Iterate through dataset
    for inputs, target in dataset:
        # 1. reset transient state
        for n in all_neurons:
            n.value = 0.0
            n.layer = float('inf')
        # 2. set input
        for neuron, value in zip(input_neurons, inputs):
            neuron.value = value
        # 3. forward (no training side effects)
        forward_propagate(all_neurons, input_neurons, config)
        # 4. output
        output = conclusion(all_neurons)
        total_error += abs(target - output)
    return total_error / len(dataset)