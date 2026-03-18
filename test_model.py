import argparse
import os
from scripts.test import show_training, show_batch, accuracy_test
from scripts.train import Trainer

def get_last_epoch(save_path):
    """Find the last epoch file in the save folder"""
    save_path = os.path.join(save_path, "models")
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"Save path '{save_path}' does not exist")
    model_files = [f for f in os.listdir(save_path) if f.startswith("model_epoch") and f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {save_path}")
    # Extract epoch numbers and get max
    epochs = [int(f.split("epoch")[1].split(".pth")[0]) for f in model_files]
    return max(epochs)

def run_testing(function_call, epoch_to_load, gen_name):
    # Verify gen_name
    if not gen_name:
        raise ValueError(f"'--gen' is required for all function calls")

    dataset_path = f"outputs/{gen_name}/"
    save_path = f"./outputs/model_{gen_name}/"

    # Initialize trainer
    trainer = Trainer(dataset_path=dataset_path,
                      save_path=save_path,
                      test=True)

    metrics = trainer.metrics
    dataloader = trainer._testloader

    if function_call == "accuracy":
        model_files = [f for f in os.listdir(os.path.join(save_path, "models")) if f.startswith("model") and f.endswith(".pth")]
        available_epochs = [int(f.split("epoch")[1].split(".pth")[0]) for f in model_files]
        
        if not available_epochs:
            raise FileNotFoundError(f"No model files found in '{save_path}'")

        if epoch_to_load is None:
            epoch_to_load = max(available_epochs)
            print(f"No epoch given. Using last epoch: {epoch_to_load}")

        if epoch_to_load not in available_epochs:
            raise ValueError(f"Epoch {epoch_to_load} not found in '{save_path}'. Available epochs: {available_epochs}")

        trainer.load_net(epoch_to_load)

        y_true, y_pred = trainer.predict(dataloader)
        accuracy_test(y_true, y_pred)

    elif function_call == "batch":
        trainer.load_net(get_last_epoch(save_path))
        show_batch(dataloader)

    elif function_call == "training":
        show_training(metrics)

    else:
        raise ValueError(f"Unknown function call: {function_call}")

def parse_args():
    parser = argparse.ArgumentParser(description="Test a given model on data.")
    parser.add_argument("--show", type=str, choices=["accuracy", "batch", "training"], default="accuracy",
                        help="Function to call for the testing")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to load (required for accuracy; default = last epoch)")
    parser.add_argument("--gen", type=str, required=True, help="Generation name (required)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_testing(args.show, args.epoch, args.gen)

    '''
    EXAMPLE OF USAGE

    python test_model.py --show accuracy --epoch 3 --gen gen1
    python test_model.py --show batch --gen gen1
    python test_model.py --show training --gen gen1
    python test_model.py --show accuracy --gen gen1  # will pick last epoch automatically
    '''
