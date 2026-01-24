from system import forward_propagation, conclusion

# Predict for a single input
def predict(model, inputs, config):
    neurons = model["neurons"]
    input_layer = neurons[0]

    # 1. set input
    for neuron, value in zip(input_layer, inputs):
        neuron.value = value

    # 2. forward only
    forward_propagation(model)

    # 3. output
    return conclusion(model)


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

    input_dim = len(model["neurons"][0])
    if len(X) != input_dim:
        raise ValueError(
            f"Input dim mismatch: got {len(X)}, expected {input_dim}"
        )

    # Predict using the neural network
    return predict(model, X, config)