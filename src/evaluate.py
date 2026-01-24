from system import forward_propagation, conclusion

def evaluate(model, dataset, config):
    total_error = 0.0
    neurons = model["neurons"]
    input_layer = neurons[0]
    output_neuron = neurons[-1][0]

    # Iterate through dataset
    for x_values, target in dataset:
        # 1. set input
        for neuron, x in zip(input_layer, x_values):
            neuron.value = x

        # 2. forward only
        forward_propagation(model)

        # 3. output
        output = conclusion(model)

        # 4. error
        total_error += abs(output - target)

    return total_error / len(dataset)