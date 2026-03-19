import os
import argparse
from scripts.train import Trainer
from scripts.DeepLearningCV.models import TinyNet, Net

def run_training(dataset_path, model_name, batch_size, epoch_size, gen_name):

    if not os.path.exists(dataset_path):
        raise ValueError(f"{dataset_path} not found !")

    # Select the model class
    if model_name.lower() == "tinynet":
        net = TinyNet
    elif model_name.lower() == "net":
        net = Net
    else:
        raise ValueError("Invalid model_name. Choose 'TinyNet' or 'Net'.")

    save_path = f"./outputs/model_{gen_name}/"

    trainer = Trainer(net=net,
                      batch_size=batch_size,
                      dataset_path=dataset_path,
                      save_path=save_path)
    trainer.fit(epoch_size)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on generated data.")
    parser.add_argument("--data", type=str, required=True, help="Data path")
    parser.add_argument("--model", type=str, choices=["TinyNet", "Net"], default="TinyNet",
                        help="Model to use: TinyNet or Net")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--name", type=str, required=True, help="Generation name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training(args.data, args.model, args.batch, args.epochs, args.name)

    '''
    EXAMPLE OF USAGE

    python train_model.py --data ./outputs/gen1 --model TinyNet --batch 8 --epochs 1 --name gen1
    python train_model.py --data ./outputs/gen1 --model Net --batch 16 --epochs 5 --name gen2
'''
