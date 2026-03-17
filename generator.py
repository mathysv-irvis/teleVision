# generator.py
from scripts.camera import Camera
import pandas as pd
import random
import os
import argparse
from tqdm import tqdm

OUTPUT_DIR = "./outputs"

def even_distribution(tuple_size=3):
    if bool(random.getrandbits(1)):
        return tuple(bool(random.getrandbits(1)) for _ in range(tuple_size))
    return tuple(False for _ in range(tuple_size))

def run_generation(gen_name, n_child, image_source=None):
    
    gen_path = os.path.join(OUTPUT_DIR, gen_name)
    os.makedirs(gen_path, exist_ok=True)

    df_gen = pd.DataFrame({"fname":[],
                           "pixel_art":[],
                           "color_art":[],
                           "column_art":[]})

    # Decide the Camera input source
    if image_source == "calibration":
        cam = Camera(raw=True, art=True, image_test="./ressource/Monitor-Calibration.png")
    elif image_source == "webcam":
        cam = Camera(raw=True, art=True, image_test=None)
    else:
        raise ValueError("Invalid --source. Must be 'webcam' or 'calibration'.")

    for i_child in tqdm(range(n_child), desc=f"Generating {gen_name}"):
        fname = os.path.join(gen_path, f"child{i_child:03d}")

        pixel_art, color_art, column_art = even_distribution()
        cam.set_artifact(pixel_art, color_art, column_art)
        cam.snapshot()
        cam.save(fname)
        df_gen.loc[len(df_gen)] = [fname, pixel_art, color_art, column_art]

    # Save CSV once at the end
    df_gen.to_csv(os.path.join(gen_path, f"df_{gen_name}.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run artifact generation.")
    parser.add_argument("--gen", type=str, default="gen1", help="Generation name")
    parser.add_argument("--size", type=int, default=200, help="Number of children to generate")
    parser.add_argument("--source", type=str, choices=["webcam", "calibration"], default="calibration",
                        help="Image source: 'webcam' or 'calibration'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generation(args.gen, args.size, args.source)

    '''
    EXAMPLE OF USAGE

    python generator.py --gen gen1 --size 200 --source calibration
    python generator.py --gen test --size 50 --source webcam
    '''
