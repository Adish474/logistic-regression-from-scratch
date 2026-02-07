# Logistic Regressions From Scratch

## Project Overview
This project implements Logistic Regression from scratch using NumPy.
The model is trained using Gradient Descent optimization and compared
with scikit-learn's implementation.

## Features
- Sigmoid activation function
- Binary cross-entropy cost function
- Gradient Descent optimization
- Early stopping using tolerance
- Cost tracking during training
- Comparison with scikit-learn
- Decision boundary visualization

## Technologies Used
- Python
- NumPy
- Matplotlib
- scikit-learn

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the training script:
   python train.py

## Results
The custom implementation achieves accuracy comparable to scikit-learn
on a synthetic binary classification dataset.

## Future Improvements
- Add train-test split
- Add regularization (L1/L2)
- Add confusion matrix visualization
- Support multi-class classification
