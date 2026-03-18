import os
import torch
import pandas as pd
from tqdm import tqdm

from .trainer import Trainer

def train(net, epoch_size, batch_size, dataset_path, save_path):

    trainer = Trainer(net=net,
                      batch_size=batch_size,
                      dataset_path=dataset_path,
                      save_path=save_path)

    for epoch in range(epoch_size):
        trainer.net.train()
        running_loss = 0.0

        loop = tqdm(trainer.trainloader, desc=f"Epoch {epoch+1}/{epoch_size}", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(trainer.device), labels.float().to(trainer.device)

            trainer.optimizer.zero_grad()
            outputs = trainer.net(inputs)
            loss = trainer.criterion(outputs, labels)
            loss.backward()
            trainer.optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(loss=running_loss / (loop.n+1))

        epoch_loss = running_loss / len(trainer.trainloader)

        # --- Evaluate F1 on train ---
        trainer.net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            eval_loop = tqdm(trainer.trainloader, desc=f"Eval Epoch {epoch+1}", leave=False)
            for inputs, labels in eval_loop:
                inputs, labels = inputs.to(trainer.device), labels.float().to(trainer.device)
                outputs, probs, preds = trainer.get_output(inputs)
                all_preds.append(preds)
                all_labels.append(labels)

        epoch_f1 = trainer.get_f1_score(all_labels, all_preds)

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Train F1: {epoch_f1:.4f}")

        trainer.save_checkpoint(epoch+1, epoch_loss, epoch_f1)


