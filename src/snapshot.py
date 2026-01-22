import os
import json

# Convert a neuron object to a JSON-serializable dictionary
def neuron_to_json(neuron):
    layer = None if neuron.layer == float('inf') else neuron.layer
    return {
        "distance": neuron.distance,
        "value": neuron.value,
        "layer": layer
    }

# Save a snapshot of the neurons' states at a given epoch
def save_epoch_snapshot(neurons, epoch, out_dir="snapshots"):
    os.makedirs(out_dir, exist_ok=True)
    # Prepare data
    data = {
        "epoch": epoch,
        "neurons": [neuron_to_json(n) for n in neurons]
    }
    # Write to file
    path = os.path.join(out_dir, f"epoch_{epoch}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)