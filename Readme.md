# Audio CNN Project for Environmental Sound Classification

## Overview
This project implements a Convolutional Neural Network (CNN) for environmental sound classification using the ESC-50 dataset. The model uses a ResNet-like architecture to classify audio samples into 50 different environmental sound categories.

## Features
- Audio spectogram transformation with MelSpectrogram
- ResNet-style CNN architecture with residual blocks
- Data augmentation techniques including frequency masking, time masking, and mixup
- Training with AdamW optimizer and OneCycleLR scheduler
- Modal deployment for cloud-based training
- TensorBoard integration for monitoring training progress

## Project Structure
- `model.py`: Contains the AudioCNN model implementation with ResidualBlock architecture
- `train.py`: Contains the training pipeline, dataset class, and Modal deployment configuration
- `requirements.txt`: Lists all the required Python dependencies

## Prerequisites
- Python 3.x
- Modal account and CLI setup
- Required Python packages (install with `pip install -r requirements.txt`):
  - torch
  - pandas
  - torchaudio
  - tqdm
  - numpy
  - modal
  - tensorboard

## Dataset
The project uses the ESC-50 dataset (Environmental Sound Classification), which consists of 2000 environmental audio recordings organized into 50 classes. The dataset is automatically downloaded and prepared during the Modal setup process.

## Model Architecture
- Input: Mel spectrograms of audio samples
- Backbone: ResNet-like CNN with residual blocks
- Output: 50-class classification (environmental sounds)

## Usage

### Training
To train the model using Modal:

```bash
modal run train.py
```

This will:
1. Download and prepare the ESC-50 dataset
2. Initialize the AudioCNN model
3. Train for 100 epochs using AdamW optimizer
4. Save the best model based on validation accuracy
5. Log metrics to TensorBoard

### TensorBoard Monitoring
Training progress can be monitored using TensorBoard. After training has started, you can access the logs through Modal's volume mounts.

## Model Performance
The model typically achieves validation accuracies of 70-80% on the ESC-50 dataset after sufficient training.

## License
This project uses the ESC-50 dataset which is available under the Attribution-NonCommercial license.

## Acknowledgments
- ESC-50 dataset: https://github.com/karolpiczak/ESC-50
- Modal for cloud deployment
