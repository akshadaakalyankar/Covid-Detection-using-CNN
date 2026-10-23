<<<<<<< HEAD
# Covid CT Classification

Binary classification of lung CT scans as **Covid-19 positive** or **normal** using PyTorch.

---

## Project Structure

```
covid-ct-classification/
├── notebooks/
│   ├── 01_data_exploration.ipynb          # EDA – class distribution, sample images
│   └── covid-detector-project.ipynb       # CNN project and patient-level evaluation
├── src/
│   ├── dataset.py                         # Dataset & DataLoader factory
│   ├── models.py                          # BaselineCNN definition
│   ├── train.py                           # Training entry-point
│   └── evaluate.py                        # Evaluation, confusion matrix, curves
├── results/
│   ├── metrics.json                       # Val accuracy, loss history
│   ├── training_curves.png                # Loss / accuracy plot
│   └── confusion_matrix.png              # Per-class confusion matrix
├── docs/
│   ├── data_dictionary.md                 # Dataset fields and pre-processing
│   └── model_card.md                      # Model details and performance
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Organise your CT scans in an `ImageFolder`-compatible layout:

```
data/
├── train/
│   ├── covid/
│   └── normal/
└── val/
    ├── covid/
    └── normal/
```

### 3. Train

```bash
# CNN implementation
python src/train.py --data_dir data --epochs 20
```

### 4. Evaluate

```bash
python src/evaluate.py --checkpoint results/baseline_best.pth --data_dir data
```

### 5. Explore notebooks

Open the `notebooks/` folder in Jupyter Lab or run them on Kaggle.

---

## Results

See `docs/model_card.md` for a full performance table once training is complete.

---

## Disclaimer

This project is for **educational and research purposes only** and is not intended for clinical use.
=======
# Covid-Detection-using-CNN
>>>>>>> 6384a89b88853912bf506362a9d868eac6c18951
