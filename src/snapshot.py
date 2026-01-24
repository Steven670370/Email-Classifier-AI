import os
import json

# Convert a neuron object to a JSON-serializable dictionary
def neuron_to_json(neuron,layer_idx, neuron_idx):
    layer = None if neuron.layer == -1 else neuron.layer
    # Serialize weights dict
    weights_serializable = {k.id: v for k, v in neuron.weights.items()}

    return {
        "id": neuron.id,
        "distance": neuron.distance,    # For distance
        "value": neuron.value,          # For lightness
        "layer": layer,                 # For color   
        "weights": weights_serializable # For lines
    }

# Save a snapshot of the neurons' states at a given epoch
def save_epoch_snapshot(neurons, epoch, out_dir="snapshots"):
    os.makedirs(out_dir, exist_ok=True)
    # Prepare data
    all_neurons = [n for layer in neurons for n in layer]
    data = {
        "epoch": epoch,
        "neurons": [neuron_to_json(n) for n in all_neurons]
    }
    # Write to file
    path = os.path.join(out_dir, f"epoch_{epoch:03d}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)