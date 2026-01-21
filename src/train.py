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
    # ------------------------------------------------
    # Split dataset into train, val, test
    import numpy as np
    from sklearn.model_selection import train_test_split
    # Split dataset into train, val, test
    X = np.array([x for x, y in dataset])
    y = np.array([y for x, y in dataset])
    # train / temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # val / test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    # Reassemble datasets
    train_dataset = list(zip(X_train, y_train))
    val_dataset   = list(zip(X_val, y_val))
    test_dataset  = list(zip(X_test, y_test))
    # ------------------------------------------------
    model = init_model(config)
    # Get all neurons for snapshotting
    all_neurons = model["neurons"]
    batch_size = config.get("batch_size", 32) # default batch size
    # Training loop with mini-batches
    for epoch in range(config["epochs"]):
        # Shuffle training data
        np.random.shuffle(train_dataset)
        # Process mini-batches
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            train_one_epoch(model, batch, config)
        # Save snapshot
        save_epoch_snapshot(all_neurons, epoch)
        # Validation error
        val_error = evaluate(model, val_dataset, config)
        logging.info(f"Epoch {epoch}: avg validation error = {val_error:.4f}")
    # Final evaluation on test set
    final_error = evaluate(model, test_dataset, config)
    logging.info(f"Final test error = {final_error:.4f}")

if __name__ == "__main__":
    main()