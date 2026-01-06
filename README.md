# DNN_Image-Text_Generate
# Multimodal Sequence Modelling for Visual Story Understanding.
# Project Overview
# Project Name: StoryReasoning-Multimodal
# Author: Eashkumar Kaki


## Quick Links
- *[Experiments Notebook](experiment_notebook.ipynb)* – Full experimental workflow and implementation  
- *[Evaluation Results](results/)* – All The results are in this folder 
- *[Model Architecture](src/)* – Encoders, fusion, temporal modelling, and decoders.

- ## Innovation Summary
*StoryReasoning-Multimodal** is a multimodal deep learning system designed to perform **visual story reasoning** using sequences of images and their corresponding textual descriptions. The model jointly processes visual and textual information, aligns them using contrastive learning, and models temporal dependencies using an **LSTM-based sequence model*


The objective is to predict the **next image** in a story sequence given a fixed number of previous steps.

---

## Key Features
- Multimodal learning with images + text  
- Pretrained ResNet for visual encoding  
- DistilBERT for textual encoding  
- Contrastive alignment loss (CLIP-style)  
- Temporal modeling using LSTM  
- Dual decoders for image reconstruction and text generation  
- Automatic saving of plots, tables, and results  
- CPU-friendly and memory-efficient training  

---

## Dataset
- **Source:** Hugging Face  
- **Dataset Name:** daniel3303/StoryReasoning  
- **Split Used:** Train  
- **Dataset Size:** 300 samples  
- **Streaming Enabled:** Yes  
- **Sequence Length (K):** 4  
- **Prediction Target:** (K + 1) image and text  

---
## Executive Summary
This project investigates multimodal sequence modelling for visual story understanding, a task requiring the integration of visual perception, natural language understanding, and temporal modelling. Given four image–text pairs forming a narrative, the system predicts the fifth image and its corresponding textual description while preserving narrative coherence.

The model combines a convolutional neural network for visual encoding, a transformer-based language model for text encoding, multimodal feature fusion, and a recurrent temporal model. A dual-decoder design enables simultaneous generation of the next image and text from a shared narrative representation. Transfer learning is employed to reduce computational requirements, and an explicit contrastive alignment loss is introduced to strengthen semantic correspondence between visual and textual embeddings.

Experiments conducted on a subset of the StoryReasoning dataset demonstrate that the model successfully captures narrative structure and semantic continuity, achieving competitive performance under limited data and training epochs.

## Key Results

| Metric       | Score |
|--------------|-------|
| BLEU         | *0.71* |
| ROUGE-L      | *0.84* |
| METEOR       | *0.85* |

These results indicate strong lexical overlap, semantic consistency, and narrative coherence in the generated textual outputs.

---

## Evaluation Metrics
Text generation quality is evaluated using standard NLP metrics:
- BLEU  
- ROUGE-L  
- METEOR 

## Project Structure
ProjectUsername/
│
├── src/
│ ├── utils.py # Dataset loading, preprocessing, visualizations
│ ├── model.py # Multimodal architecture definition
│ └── train.py # Training loop, evaluation, metrics
│
├── results/
│ ├── figures/ # Saved plots and visual outputs
│ └── tables/ # Dataset statistics and metric tables
│
├── config.yaml # Project configuration
├── requirements.txt # Dependencies
└── README.md



---


## Model Architecture

### Visual Encoder
- Backbone: ResNet (pretrained)  
- Output embedding size: 256  
- Backbone parameters frozen for efficiency  

### Text Encoder
- Backbone: DistilBERT (pretrained)  
- CLS token representation  
- Output embedding size: 256  
- Backbone frozen during training  

### Multimodal Alignment
- Contrastive loss aligns image and text embeddings  
- L2-normalized embeddings  
- Symmetric cross-entropy loss (image-to-text & text-to-image)  

### Temporal Modeling
- Fused embeddings processed using **LSTM**  
- Captures narrative progression over time  
- Final hidden state used as context vector  

### Decoders
- **Image Decoder:** ConvTranspose-based reconstruction network  
- **Text Decoder:** GRU with teacher forcing and vocabulary projection  

---

## Training Configuration
- Batch Size: 2  
- Epochs: 3  
- Learning Rate: 0.0001  
- Optimizer: AdamW  
- Device: CPU  
- Contrastive Loss Weight (λ): 0.5  

### Training Loss Components
- Image reconstruction loss (MSE)  
- Text generation loss (Cross-Entropy)  
- Contrastive multimodal alignment loss  

**Total Loss:**  
Loss = Image Loss + Text Loss + λ × Contrastive Loss

---
---

## Visualizations and Outputs
All results are automatically saved to the `results/` directory.

**Figures Saved:**
- Dataset story length distribution  
- Token length distribution  
- Sample story sequences (inputs + target)  
- Embedding magnitude plots  
- Temporal feature norms  
- Contrastive similarity plots  
- Training loss curves  

**Tables Saved:**
- Dataset statistics  
- Training loss summary  
- Evaluation metric scores  

---
## Limitations and Future Work
- Small training subset limits generalization.
- Image decoder struggles with fine-grained visual detail.
- Recurrent temporal modelling lacks an explicit attention mechanism.

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt

Step 2: Run Dataset Preparation
python src/utils.py

Step 3: Train the Model
set KMP_DUPLICATE_LIB_OK=TRUE
python src/train.py

Environment Notes

Designed to run on CPU-only systems

Uses streaming dataset loading to avoid disk usage

Includes safeguards against OpenMP duplication issues

Conclusion

This project demonstrates that meaningful multimodal sequence reasoning can be achieved under limited computational resources by leveraging pretrained models, contrastive alignment, and efficient temporal modeling. The architecture aligns closely with modern multimodal learning principles while remaining practical and reproducible.
