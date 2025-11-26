# MusicNet Incremental CNN

This project implements a Convolutional Neural Network (CNN) for audio transcription (based on the MusicNet dataset) using an **incremental training** approach.

The goal is to process large volumes of data (Mel Spectrograms) in batches, enabling training in environments with limited RAM, such as Google Colab.

## 🚀 Features

* **Incremental Training:** Loads and trains the model on file subsets to save memory.
* **Continuous Validation:** Evaluates the model with a fixed validation set after every cycle.
* **Early Stopping:** Stops training if the F1-Score reaches a defined threshold.
* **Checkpointing:** Saves model and optimizer state after each processed batch.

## 🛠️ Tech Stack

* Python
* PyTorch (CNN Modeling and Dataloaders)
* Scikit-learn (Metrics: F1, Precision, Recall)
* NumPy

## 📋 Prerequisites

The code expects pre-processed data (`.pt` files containing spectrograms) in the following folders:
* `/content/Processed_Musics` (Training)
* `/content/Validation_Musics` (Validation)

*Note: Change the paths in the code (`processed_path` and `validation_path`) if running locally.*

## 📦 How to Use

1. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio scikit-learn numpy
