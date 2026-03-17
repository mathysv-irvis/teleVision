from scripts.camera import Camera
import pandas as pd
import random
import os

from tqdm import tqdm

OUTPUT_DIR = "./outputs"

def even_distribution(tuple_size=3):
    if bool(random.getrandbits(1)):
        return tuple(bool(random.getrandbits(1)) for _ in range(tuple_size))
    return tuple(False for _ in range(tuple_size))

def run_generation(gen_name, n_child, image_test=None):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_gen = pd.DataFrame({"fname":[],
                           "pixel_art":[],
                           "color_art":[],
                           "column_art":[]})


    cam = Camera(raw=True, art=True,image_test=image_test)

    for i_child in range(n_child):
        fname = os.path.join(OUTPUT_DIR, gen_name, f"child{i_child:03d}")

        pixel_art, color_art, column_art = even_distribution()
        cam.set_artifact(pixel_art, color_art, column_art)
        cam.snapshot()
        cam.save(fname)
        df_gen.loc[len(df_gen)] = [fname, pixel_art, color_art, column_art]
        df_gen.to_csv(os.path.join(OUTPUT_DIR, gen_name, f"df_{gen_name}.csv"), index=False)

def run_training():
    classes = ("pixel_art", "color_art", "column_art")
    net = TinyNet
    epoch_size = 1
    batch_size = 8
    dataset_path = "outputs/gen1/"
    save_path = "./outputs/model_gen1/"
    train(net, classes, epoch_size, batch_size, dataset_path, save_path)


if __name__ == "__main__":
    """
    generation = "gen1"
    gen_size = 1
    test_image = "./ressource/Monitor-Calibration.png"
    run_generation(generation,gen_size,test_image)
    """
    from scripts.DeepLearningCV.models import TinyNet
    from scripts.train import train
    run_training()
