# TeleVision: Artifact Generation and Training

This repository contains tools to generate synthetic image artifacts and train deep learning models to detect and classify them.

---

## Project Structure

```bash
project-root/
├── generator.py          # CLI for generating artifact images
├── train_model.py        # CLI for training models on generated data
├── scripts/
│   ├── camera.py         # Camera & artifact generation logic
│   └── DeepLearningCV/   # Neural network models & training functions
├── outputs/              # Generated data and trained models (ignored by Git)
└── README.md
```

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
python train_model.py --data ./outputs/data --model TinyNet --batch 8 --epochs 1 --name gen1
python train_model.py --data ./outputs/data --model Net --batch 16 --epochs 5 --name gen2
```

--data : Data path to train on

--model : Network to train (TinyNet or Net)

--batch : Batch size

--epochs: Number of epochs

--name : Generation name


### Test a Model

test_model.py test and plot the results of a trained neural network on generated data.

Run:
```python
python test_model.py --show <training|batch|accuracy> --gen <generation_name> --epochs <epoch to show (only for accuracy)>
```

Examples:

```python
python test_model.py --show accuracy --epoch 3 --gen gen1
python test_model.py --show batch --gen gen1
python test_model.py --show training --gen gen1
python test_model.py --show accuracy --gen gen1  # will pick last epoch automatically
```

--show : Function to call (training : plot metrics of training / batch : plot images from the first batch / accuracy : print the accuracy and precision of a given epoch (last one if not given)

--epoch : Accuracy of the given epoch or last epoch if epoch is not given (only for function accuracy)

--gen : Generation folder to test

