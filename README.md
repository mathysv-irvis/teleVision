# TeleVision: Artifact Generation and Training

This repository contains tools to generate synthetic image artifacts and train deep learning models to detect and classify them.

---

## Project Structure

.
├── generator.py          # CLI for generating artifact images
├── train_model.py        # CLI for training models on generated data
├── scripts/
│   ├── camera.py         # Camera & artifact generation logic
│   └── DeepLearningCV/   # Neural network models & training functions
├── outputs/              # Generated data and trained models (ignored by Git)
└── README.md

---

## Requirements

Install packages using:

```bash
pip install -r requirements.txt
```

## Usage
### Generate Artifacts

generator.py allows you to generate synthetic image artifacts.

Run:
```python
python generator.py --gen <generation_name> --size <number_of_children> --source <webcam|calibration>
```

Examples:

```python
python generator.py --gen gen1 --size 200 --source calibration
python generator.py --gen test --size 50 --source webcam
```

--gen : Name of the generation (folder in outputs/)

--size: Number of children to generate

--source: "webcam" or "calibration" image

### Train a Model

train_model.py trains a neural network on generated data.

Run:
```python
python train_model.py --model <TinyNet|Net> --batch <batch_size> --epochs <num_epochs> --gen <generation_name>
```

Examples:

```python
python train_model.py --model TinyNet --batch 8 --epochs 1 --gen gen1
python train_model.py --model Net --batch 16 --epochs 5 --gen gen2
```

--model : Network to train (TinyNet or Net)

--batch : Batch size

--epochs: Number of epochs

--gen : Generation folder to train on
