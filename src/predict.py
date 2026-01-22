from system import forward_propagate, conclusion

# Predict for a single input
def predict(model, inputs, weight_judge, config):
    input_neurons = model["input"]
    all_neurons = model["all"]
    # 1. reset transient state
    for n in all_neurons:
        n.value = 0.0
        n.layer = -1
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
    return conclusion(all_neurons, weight_judge)

# Predict for a batch of inputs
def predict_batch(model, dataset, config):
    outputs = []
    for inputs, _ in dataset:
        outputs.append(predict(model, inputs, config))
    return outputs


# Additional utility to predict from raw email text
import joblib
from predict import predict

vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_from_text(model, text, config):
    # Convert text to TF-IDF vector
    X = vectorizer.transform([text]).toarray()[0]
    # Predict using the neural network
    return predict(model, X, config)