from torchvision.utils import make_grid
from torch import no_grad
from sklearn.metrics import precision_score, accuracy_score

import matplotlib.pyplot as plt
import numpy as np

from .constant import CLASSES
from .train import Trainer

def show_training(metrics_df):
    plt.figure(figsize=(10,5))
    plt.plot(metrics_df["epoch"], metrics_df["loss"], label="Loss")
    plt.plot(metrics_df["epoch"], metrics_df["f1_score"], label="F1 Score", linestyle="dashdot")
    plt.plot(metrics_df["epoch"], metrics_df["accuracy"], label="Accuracy score", linestyle="dotted")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Evolution of metrics thru epochs")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def imshow(img, title=None):
    img = img / 2 + 0.5
    npimg = img.numpy()

    plt.figure(figsize=(10,6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    if title is not None:
        plt.title(title, fontsize=12)

    plt.axis("off")
    plt.show()

def show_batch(testloader):
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    batch_size = images.shape[0]

    cols = 4
    rows = int(np.ceil(batch_size / cols))

    plt.figure(figsize=(12, 6))

    for j in range(batch_size):

        img = images[j] / 2 + 0.5
        img = np.transpose(img.numpy(), (1, 2, 0))

        active = [CLASSES[i] for i in range(len(CLASSES)) if labels[j][i] == 1]
        if not active:
            active = ["clean"]

        plt.subplot(rows, cols, j + 1)
        plt.imshow(img)
        plt.xlabel(", ".join(active))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def accuracy_test(y_true, y_pred):

    precision = precision_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Model accuracy over {len(y_pred)} samples : {accuracy*100} %')
    for i, classname in enumerate(CLASSES):
        print(f'\t| Precision over {classname:12s} : {precision[i]*100:.2f} %')

