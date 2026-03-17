import os
import pandas as pd

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch

from .DeepLearningCV.preprocess import ArtifactDataset

class Trainer:

    def __init__(self, net, classes, batch_size, im_size, dataset_path, save_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.im_size    = im_size
        self.classes = classes
        self.class_size = len(self.classes)
        
        self.net = net(self.class_size).to(self.device)
       
        transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size)),
            #transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

        self.save_dir = os.path.join(save_path, "models")
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.save_dir, "training_metrics.csv")
        
        if not os.path.exists(self.metrics_file):
            self.metrics_df = pd.DataFrame(columns=["epoch", "loss", "f1_score"])
            self.metrics_df.to_csv(self.metrics_file, index=False)
        else:
            self.metrics_df = pd.read_csv(self.metrics_file)

        self.dataset_path = dataset_path
        self.dataset_df   = "df_"+self.dataset_path.split("/")[-2]+".csv"
        self.dataset = ArtifactDataset(os.path.join(self.dataset_path, self.dataset_df), transform=transform)

        self.batch_size = batch_size

        self.preprocess()

    def imshow(img):
        img = img / 2 + 0.5
        plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
        plt.show()

    def get_f1_score(self, y_true, y_pred):
        y_pred    = torch.cat(y_pred, 0)
        y_true    = torch.cat(y_true, 0)
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        return f1_score(y_true_np, y_pred_np, average="samples")

    def preprocess(self, train_proportion=0.8):

        # SPLIT DATA
        total_size = len(self.dataset)
        train_size = int(train_proportion * total_size)
        test_size = total_size - train_size

        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])

        self.trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.testloader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.criterion = self.get_criterion()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def batchshow(self):
        # SHOW BATCH
        dataiter = iter(self.trainloader)
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
        outputs = self.net(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.3).float()
        return outputs, probs, preds


    def save_checkpoint(self, save_path):
        torch.save(self.net.state_dict(), save_path)

