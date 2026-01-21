# train.py
from system import init_model, train_one_epoch
from preprocess import load_data
from config import load_config, set_seed
from evaluate import evaluate
from snapshot import save_epoch_snapshot
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)

# Main function
def main():
    config = load_config()
    set_seed(config)
    dataset = load_data()
    model = init_model(config)
    # Get all neurons for snapshotting
    all_neurons = model["neurons"]
    # Training loop
    for epoch in range(config["epochs"]):
        train_one_epoch(model, dataset, config)  # training step
        save_epoch_snapshot(all_neurons, epoch)     # snapshot step
        error = evaluate(model, dataset)         # evaluation step
        logging.info(f"Epoch {epoch}: avg error = {error:.4f}")

if __name__ == "__main__":
    main()