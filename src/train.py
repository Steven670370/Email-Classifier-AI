# train.py
from system import init_model, train_one_epoch
from preprocess import load_data
from config import load_config, set_seed
from evaluate import evaluate
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

    for epoch in range(config["epochs"]):
        train_one_epoch(model, dataset, config)  # training step
        error = evaluate(model, dataset)         # evaluation step
        logging.info(f"Epoch {epoch}: avg error = {error:.4f}")

if __name__ == "__main__":
    main()