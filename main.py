from scripts.test import test
from scripts.DeepLearningCV.models import TinyNet

gen_name = "gen1"
dataset_path = f"./outputs/{gen_name}/"
save_path = f"./outputs/model_{gen_name}/"

test(TinyNet, dataset_path, save_path)

