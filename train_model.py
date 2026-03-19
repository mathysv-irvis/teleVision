import os
import argparse
from scripts import Trainer

def run_training(dataset_path, model_name, batch_size, epoch_size, gen_name):

    if not os.path.exists(dataset_path):
        raise ValueError(f"{dataset_path} not found !")

    save_path = f"./outputs/model_{gen_name}/"

    trainer = Trainer(net_name=model_name.lower(),
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
