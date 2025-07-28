# Text-to-Image-Quality-Evaluator

# Text-to-Image Quality Evaluator

This project implements a Text-to-Image Quality Evaluator using a PyTorch-based model. The model, named TSPMGS, is designed to assess the quality of images generated from text prompts, considering both perceptual quality and alignment with the prompt. It uses the CLIP model for feature extraction from both images and text.

-----

## 📖 Table of Contents

  * [Features](https://www.google.com/search?q=%23-features)
  * [Getting Started](https://www.google.com/search?q=%23-getting-started)
      * [Prerequisites](https://www.google.com/search?q=%23-prerequisites)
      * [Installation](https://www.google.com/search?q=%23-installation)
  * [Usage](https://www.google.com/search?q=%23-usage)
  * [Model Architecture](https://www.google.com/search?q=%23-model-architecture)
  * [Dataset](https://www.google.com/search?q=%23-dataset)
  * [Results](https://www.google.com/search?q=%23-results)

-----

## ✨ Features

  * **Custom Dataset Handling**: Implements a custom PyTorch `Dataset` class (`AIGIDataset`) for loading and preprocessing image-text data.
  * **Multi-modal Feature Extraction**: Utilizes the `open_clip` library to extract features from both images and text prompts.
  * **Patch-based Image Analysis**: The model processes images by breaking them down into smaller patches to capture local details.
  * **Task-Specific Prompts**: Constructs task-specific prompts to evaluate both the alignment of the image with the text and its perceptual quality.
  * **Multi-Head Prediction**: The model has two prediction heads, one for perceptual quality and another for alignment quality, allowing for a comprehensive evaluation.
  * **Training and Evaluation**: Includes a complete training pipeline with an AdamW optimizer, a cosine annealing learning rate scheduler, and an L1 loss function. The model's performance is evaluated using Spearman's Rank Correlation Coefficient (SRCC) and Pearson's Linear Correlation Coefficient (PLCC).
  * **Visualization**: Generates scatter plots to visualize the correlation between the predicted scores and the ground truth Mean Opinion Scores (MOS) for both perceptual quality and alignment.

-----

## 🚀 Getting Started

### Prerequisites

This project requires Python and several libraries. Make sure you have the following installed:

  * Python 3.10
  * PyTorch
  * pandas
  * Pillow (PIL)
  * scikit-learn
  * open\_clip\_torch
  * transformers
  * matplotlib
  * tqdm

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Text-to-Image-Quality-Evaluator.git
    cd Text-to-Image-Quality-Evaluator
    ```
2.  **Install the required libraries:**
    ```bash
    pip install torch pandas Pillow scikit-learn open_clip_torch transformers matplotlib tqdm
    ```

-----

## 💻 Usage

To run the project, execute the cells in the Jupyter Notebook (`Text_to_Image_Quality_Evaluator.ipynb`) sequentially.

1.  **Data Setup**: Make sure the `AGIQA_3k` dataset is available and the paths in the notebook are correctly configured.
2.  **Training**: The training process will start, and the model will be saved as `tsp_mgs_model.pth` after each epoch and `tsp_mgs_model_overall.pth` upon completion.
3.  **Evaluation**: After training, the model is evaluated on the test set, and the results, including correlation scores and plots, are displayed.

-----

## 🤖 Model Architecture

The core of this project is the **TSPMGS** model. It's a multi-head neural network that leverages the power of the CLIP model. Here's a breakdown of its architecture:

  * **CLIP Backbone**: The model uses a pretrained CLIP model (`ViT-B-32`) to encode both the input images and various text prompts into high-dimensional feature vectors.
  * **Image and Patch Features**: The model processes the entire image as well as smaller patches extracted from it. The features from these patches are averaged to get a representation of local image characteristics.
  * **Task-Specific Prompts**: The model uses specially constructed prompts to measure:
      * **Perceptual Quality**: Prompts like "A photo of good quality." or "bad photo."
      * **Alignment**: Prompts like "A photo that perfectly matches {pt}." where `{pt}` is the initial prompt.
  * **Similarity Scores**: It computes both coarse-grained (image-level) and fine-grained (word-level) similarities between image and text features.
  * **Prediction Heads**: The similarity scores are fed into two separate prediction heads, each a small neural network, to output the final perceptual and alignment quality scores.

-----

## 📊 Dataset

The model is trained and evaluated on the **AGIQA-3k** dataset. This dataset contains 3,000 images, each with a corresponding text prompt and Mean Opinion Scores (MOS) for both quality and alignment.

  * **Data Split**: The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.
  * **Preprocessing**: Images are resized to 224x224 and normalized before being fed into the model.

-----

## 📈 Results

After 200 epochs of training, the model achieves the following performance on the test set:

### Perceptual Quality Prediction

  * **SRCC**: 0.8696
  * **PLCC**: 0.8845

### Alignment Quality Prediction

  * **SRCC**: 0.8037
  * **PLCC**: 0.8117

The scatter plots generated by the notebook show a strong positive correlation between the model's predicted scores and the ground truth MOS values, indicating that the model is effective at evaluating the quality and alignment of text-to-image outputs.
