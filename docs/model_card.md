
# Model Card – COVID CT Classifier (Baseline CNN)

## 1. Model Overview

| Field | Details |
|---|---|
| Model Name | Baseline CNN |
| Task | Binary classification: COVID vs. normal |
| Framework | PyTorch |
| Input | 224 × 224 grayscale CT images repeated across 3 channels |
| Output | COVID probability using sigmoid activation |
| Architecture | Custom Convolutional Neural Network (CNN) |
| Loss Function | Binary Cross-Entropy (BCE) |
| Model Scope | CNN trained from scratch |

## 2. Intended Use

- Research and educational exploration of deep learning for medical image classification.
- Learning CNN architecture, image preprocessing, model training, and evaluation.
- Comparing classification metrics and understanding model limitations.

**This model is not intended for clinical diagnosis or deployment in a medical setting.**

## 3. Dataset

| Field | Details |
|---|---|
| Dataset | COVID-CTset |
| Source | [Kaggle – COVID-CTset](https://www.kaggle.com/datasets/mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset) |
| Classes | Normal and COVID |
| Image Format | TIFF |
| Original Image Size | 512 × 512 pixels (inspected sample) |
| Model Input Size | 224 × 224 pixels |
| Data Splitting | Patient-level training, validation, and test sets |

The dataset contains more normal images than COVID images. Patient-level splitting is used to reduce data leakage by ensuring that images from the same patient do not appear in multiple splits.

## 4. Model Architecture

The baseline CNN uses convolutional layers to extract features from CT images and fully connected layers to perform binary classification.

- **Convolutional layers:** Extract visual features from CT scans.
- **ReLU activation:** Introduces non-linearity into the network.
- **Pooling layers:** Reduce the spatial dimensions of feature maps.
- **Adaptive average pooling:** Produces a fixed-size feature representation.
- **Fully connected layers:** Perform binary classification.
- **Dropout:** Helps reduce overfitting.
- **Output layer:** Produces a single logit, which is converted into a COVID probability using sigmoid activation.

## 5. Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | As configured in `train.py` |
| Batch Size | As configured in `train.py` |
| Number of Epochs | As configured in `train.py` |
| Learning Rate Scheduler | ReduceLROnPlateau, if enabled |
| Loss Function | Binary Cross-Entropy (BCE) |
| Output Activation | Sigmoid during evaluation |
| Model Checkpoint | `results/baseline_cnn_best.pth` |

*Note: The training configuration and checkpoint filename must match the actual implementation.*

## 6. Performance Evaluation

The following metrics will be recorded after training and evaluation. No performance values are reported until the experiments have been completed.

| Metric | Validation Set | Test Set |
|---|---|---|
| Accuracy | TBD | TBD |
| AUC-ROC | TBD | TBD |
| Precision (COVID Class) | TBD | TBD |
| Recall (COVID Class) | TBD | TBD |
| F1-Score (COVID Class) | TBD | TBD |

### Evaluation Artifacts

- Training curves: `results/training_curves.png`
- Confusion matrix: `results/confusion_matrix.png`
- ROC curve: `results/roc_curve.png`
- Test metrics: `results/test_metrics.json`

The validation set will be used for model selection and classification-threshold selection. The test set will be reserved for final evaluation.

## 7. Limitations and Risks

- The model may overfit because of the limited number of patients.
- Class imbalance may affect classification performance.
- Image characteristics may vary across scanners, hospitals, and patient populations.
- Image-level evaluation may not accurately represent patient-level performance.
- The model has not been externally validated or evaluated in a clinical workflow.
- High accuracy alone does not establish clinical usefulness.

## 8. Ethical Considerations

- The model must not be used to make or influence clinical decisions.
- Patient data must be handled according to applicable privacy requirements and dataset usage terms.
- Reported metrics reflect performance on the evaluated dataset and may not generalize to other populations.
- Predictions are experimental outputs and must not be treated as medical advice.

## 9. Future Work

1. Improve the CNN architecture and tune its training configuration.
2. Evaluate accuracy, precision, recall, F1-score, and AUC-ROC.
3. Investigate class imbalance and classification-threshold selection.
4. Explore patient-level evaluation and external validation.
