
# Brain Tumor Detection and Classification using Vision Transformers

This repository contains an implementation of a Vision Transformer (ViT) model designed to classify brain tumor images into four categories (Meningioma, Pituitary, Glioma and No tumor). The model can be trained on any dataset of brain tumor MRI scans.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Testing on a New Image](#testing-on-a-new-image)
- [Results](#results)
- [Acknowledgements](#acknowledgements)


## Folder Structure

```
├── data/                   # Dataset directory (Not included)   
├── best_model.pth          # Saved model with the best validation accuracy 
├── cleanup.py              # Script for cleaning up the dataset   
├── requirements.txt        # List of required Python packages
├── test.py                 # Script for testing the model on new images
├── train.py                # Script for training the model
└── transformer.py          # Definition of the Vision Transformer model
```

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/marvelefe/VIT-POC.git
    cd VIT-POC
    ```

2. **Create a virtual environment and install dependencies**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Prepare the dataset**: 
   - Place your dataset under the `./data` directory and split your training and validation so you have two sub-directories: `./data/Training` and `./data/Testing` directories. 
 
> A sample dataset can be downloaded here on Kaggle https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
   

## Training the Model

To train the model, run:

```bash
python train.py
```

Training progress, including loss and accuracy for both training and validation sets, will be displayed and saved as plots (`accuracy.png`, `loss.png`). The best-performing model is saved as `best_model.pth`.

## Evaluating the Model

During training, the model is evaluated against the validation dataset. Post-training, a confusion matrix is generated and saved as `confusion_matrix.png`, and provides insights into the model's classification performance.

## Testing on a New Image

To classify a new image, use the `test.py` script:

```bash
python test.py
```

Replace the image path in the script with your target image. The model predicts the tumor class, and a confidence level is displayed along with a visual bar chart saved as `prediction_result.png`.

## Results

The model achieves competitive accuracy in classifying tumor images across four distinct classes. Performance metrics, including accuracy and loss plots, are available in the repository.

## Acknowledgements

This project uses the [ViT-pytorch](https://github.com/lucidrains/vit-pytorch) library for the Vision Transformer implementation.
 
