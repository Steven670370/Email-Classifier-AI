from system import forward_propagate, conclusion

# Predict for a single input
def predict(model, inputs, config):
    input_neurons = model["input"]
    all_neurons = model["all"]
    # 1. reset transient state
    for n in all_neurons:
        n.value = 0.0
        n.layer = float('inf')
    # 2. set input values
    for neuron, value in zip(input_neurons, inputs):
        neuron.value = value
    # 3. forward propagation (no training side effects)
    forward_propagate(
        all_neurons,
        input_neurons,
        config,
        training=False
    )
    # 4. output
    return conclusion(all_neurons)

# Predict for a batch of inputs
def predict_batch(model, dataset, config):
    outputs = []
    for inputs, _ in dataset:
        outputs.append(predict(model, inputs, config))
    return outputs