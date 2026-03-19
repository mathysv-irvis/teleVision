import os
import sys
import shutil
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch

from .DeepLearningCV.models import Net, TinyNet
from .DeepLearningCV.preprocess import ArtifactDataset
from .constant import PROBS, IM_SIZE, CLASSES, TRAINING_SIZE, LEARNING_RATE

class Trainer:

    def __init__(self, dataset_path, save_path, net=None, batch_size=None, test=False):
        
        if os.path.isdir(save_path) and not test:

            res = input(
                f"The path '{save_path}' already exists! Continue and overwrite? (y/N): "
            )

            if res.lower() != "y":
                print("Abort training, exiting...")
                sys.exit(0)
            else:
                shutil.rmtree(save_path)
                os.makedirs(save_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.im_size    = IM_SIZE
        self.classes = CLASSES
        self.class_size = len(self.classes)
        
        self.batch_size = batch_size
       
        #transform = transforms.Compose([
        #    transforms.Resize((self.im_size, self.im_size)),
        #    #transforms.RandomRotation(10),
        #    transforms.ToTensor()
        #])
        transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        
        self.save_dir   = os.path.join(save_path)
        self.save_model = os.path.join(save_path, "models")
        self.metrics_file = os.path.join(self.save_dir, "training_metrics.csv")
        self.parameters_file = os.path.join(self.save_dir, "training_parameters.csv")
        
        if test:
            self._metrics_df    = pd.read_csv(self.metrics_file)
            self.parameters_df  = pd.read_csv(self.parameters_file)
            self.batch_size     = int(self.parameters_df["batch_size"][0])
            self.setup_net_type()
        else:
            os.makedirs(self.save_model, exist_ok=True)
            
            self._net = net(self.class_size).to(self.device)
            
            self._metrics_df = pd.DataFrame(columns=["epoch", "loss", "f1_score", "accuracy"])
            self._metrics_df.to_csv(self.metrics_file, index=False)
           
            self.parameters_df = pd.DataFrame(columns=["net", "epochs", "batch_size", "lr", "im_size", "training_size", "probs"])
            self.parameters_df["batch_size"]    = [self.batch_size]
            self.parameters_df["lr"]            = [LEARNING_RATE]
            self.parameters_df["training_size"] = [TRAINING_SIZE]
            self.parameters_df["im_size"]       = [IM_SIZE]
            self.parameters_df["probs"]         = [PROBS]
            self.parameters_df["net"]           = [self._net.name]
            self.parameters_df.to_csv(self.parameters_file, index=False)

        self.dataset_path = dataset_path
        self.dataset_df   = "df_"+self.dataset_path.split("/")[-2]+".csv"
        self.dataset = ArtifactDataset(os.path.join(self.dataset_path, self.dataset_df), transform=transform)

        self.preprocess()

    @property
    def net(self):
        return self._net 

    @property
    def metrics(self):
        return self._metrics_df.copy()

    def setup_net_type(self):
        if self.parameters_df["net"][0] == "net":
            self._net = Net(self.class_size).to(self.device)
        
        elif self.parameters_df["net"][0] == "tinynet":
            self._net = TinyNet(self.class_size).to(self.device)


    def get_f1_score(self, y_true, y_pred):
        return f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        )

    def preprocess(self):

        # SPLIT DATA
        total_size = len(self.dataset)
        train_size = int(TRAINING_SIZE * total_size)
        test_size = total_size - train_size

        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])

        self._trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self._testloader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self._criterion = self.get_criterion()
        self._optimizer = optim.Adam(self._net.parameters(), lr=LEARNING_RATE)
        #self._criterion = nn.CrossEntropyLoss()
        #self._optimizer = optim.SGD(self._net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    def batchshow(self):
        # SHOW BATCH
        dataiter = iter(self._trainloader)
        images, labels = next(dataiter)
        imshow(torchvision.utils.make_grid(images))

        for j in range(self.batch_size):
            active = [classes[i] for i in range(self.class_size) if labels[j][i] == 1]

            if len(active) == 0:
                active = ["clean"]

            print(", ".join(active))
        return

    def get_criterion(self):

        df_data = pd.read_csv(os.path.join(self.dataset_path, self.dataset_df))
        pos_counts = df_data[["pixel_art", "color_art", "column_art"]].sum().values
        neg_counts = len(df_data) - pos_counts
        pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(self.device)

        return  nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_output(self, inputs):
        outputs = self._net(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > PROBS).float()
        print(probs)
        return outputs, probs, preds

    def predict(self, dataloader):

        self.net.eval()

        all_preds = []
        all_labels = []

        total = len(dataloader.dataset)
        seen = 0

        loop = tqdm(
            dataloader,
            desc="Eval",
            total=len(dataloader),
            leave=False
        )

        with torch.no_grad():
            for images, labels in loop:

                bs = labels.size(0)
                seen += bs

                images = images.to(self.device)
                labels = labels.float().to(self.device)

                _, _, preds = self.get_output(images)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                loop.set_postfix(samples=f"{seen}/{total}")

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        return y_true, y_pred

    def save_checkpoint(self, epoch, loss, f1, accuracy):
        self.parameters_df["epochs"] = [epoch]
        self.parameters_df.to_csv(self.parameters_file, index=False)
        
        new_row = pd.DataFrame([{"epoch": epoch, "loss": loss, "f1_score": f1, "accuracy": accuracy}])
        self._metrics_df = pd.concat([self._metrics_df, new_row], ignore_index=True)
        self._metrics_df.to_csv(self.metrics_file, index=False)
        
        torch.save(self._net.state_dict(), os.path.join(self.save_model, f"model_epoch{epoch:03d}.pth"))


    def load_net(self, epoch_to_load):
        self._net.load_state_dict(torch.load(os.path.join(self.save_model, f"model_epoch{epoch_to_load:03d}.pth")))

    
    def fit(self, epoch_size):
        running_loss = 0.0
        self._net.train()
        for epoch in range(epoch_size):

            loop = tqdm(self._trainloader, desc=f"Epoch {epoch+1}/{epoch_size}", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)

                self._optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=running_loss / (loop.n+1))

            epoch_loss = running_loss / len(self._trainloader)
        
            y_true, y_pred = self.predict(self._trainloader)
            epoch_f1 = self.get_f1_score(y_true, y_pred)
            #accuracy = accuracy_score(y_true, y_pred)
            accuracy = (y_true==y_pred).mean()

            print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Train Acc: {accuracy:.4f} | Train F1: {epoch_f1:.4f}")

            self.save_checkpoint(epoch+1, epoch_loss, epoch_f1, accuracy)


