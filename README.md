# Wetland Classification Using Machine Learning

## Overview
This project implements and compares various machine learning classification methods for wetland mapping using multispectral remote sensing data. The dataset contains spectral features (near-infrared, red, green, blue bands) with binary labels indicating wetland (1) or dryland (0).

## Project Structure
```
Lab1_Wetland_Classification/
│
├── Lab1_Wetland_Classification_ML_Analysis.ipynb    # Main notebook (includes custom LR)
├── myLR.py                                          # Standalone file for the custom logistic regression.  
├── train.csv                                        # Training dataset (1000 samples)
├── smalltrain.csv                                   # Smaller training set (for overfitting demo)
├── test.csv                                         # Test dataset (1000 samples)
└── README.md                                        # This file
```

**Note**: The custom logistic regression implementation (MyLR class) is included directly within the Jupyter notebook in Part 5.

## Dataset Description
- **Features**: 4 spectral bands
  - Near Infrared
  - Red
  - Green  
  - Blue
- **Target**: Binary classification (0 = Dryland, 1 = Wetland)
- **Training samples**: 1000 (560 dryland, 440 wetland)
- **Test samples**: 1000 (565 dryland, 435 wetland)

## Implemented Methods

### Part 1: Decision Tree Classifier
- Baseline performance evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score
- **Results**: 78.7% accuracy, 0.759 F1-score

### Part 2: Overfitting Analysis
- Demonstrated overfitting with reduced training data
- Model complexity reduction using `min_samples_split`
- Key finding: Regularization improves generalization

### Part 3: Model Comparison
Evaluated three classification algorithms:
1. **Decision Tree**: 78.7% accuracy
2. **Logistic Regression**: 75.7% accuracy
3. **Support Vector Machine**: 81.0% accuracy (best)

### Part 4: Ensemble Methods
- **Bagging Classifier**: 82.3% accuracy (best overall)
- **Random Forest**: 81.0% accuracy
- Demonstrated superiority of ensemble methods

### Part 5: Custom Implementation (Extra Credit)
Implemented logistic regression from scratch with advanced features:
- Polynomial feature engineering
- L2 regularization
- Momentum-based gradient descent
- Adaptive learning rate decay
- **Results**: 80.0% accuracy, 82.8% recall - **outperformed scikit-learn!**

## Key Findings

### Performance Rankings
1. Bagging Classifier: 82.3% accuracy
2. SVM: 81.0% accuracy
3. **Custom Logistic Regression: 80.0% accuracy**
4. Random Forest: 81.0% accuracy
5. Decision Tree: 78.7% accuracy
6. Scikit-Learn Logistic Regression: 75.7% accuracy

### Important Insights
- **Recall is critical**: For wetland detection, missing wetlands (false negatives) has serious ecological consequences
- **Ensemble methods excel**: Combining multiple models significantly improves performance
- **Feature engineering matters**: Custom LR with polynomial features outperformed standard LR
- **Overfitting is real**: Small training sets lead to poor generalization without regularization

## Requirements

### Python Version
- Python 3.x

### Required Libraries
```python
pandas
numpy
scikit-learn
matplotlib (optional, for visualization)
```

### Installation
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

### Running the Notebook
1. Ensure all CSV files (`train.csv`, `smalltrain.csv`, `test.csv`) are in the same directory as the notebook
2. Open Jupyter Notebook:
   ```bash
   jupyter notebook Lab1_Wetland_Classification_ML_Analysis.ipynb
   ```
3. Run cells sequentially from top to bottom (Cell > Run All)
4. All code, including the custom logistic regression implementation, is self-contained within the notebook

### Using Custom Logistic Regression
The custom logistic regression implementation is contained within Part 5 of the notebook. To use it:

1. Run all cells up to and including Part 5 (Extra Credit)
2. The `MyLR` class is defined in a code cell
3. The class is then instantiated, trained, and evaluated in subsequent cells

```python
# Already implemented in the notebook
model = MyLR(learning_rate=0.5, max_iterations=3000, reg_lambda=0.01)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
```

## Custom Implementation Details

### MyLR Class Features
- **Polynomial Features**: Adds squared terms and interaction features
- **Feature Normalization**: Z-score standardization
- **L2 Regularization**: Prevents overfitting (lambda=0.01)
- **Momentum Optimization**: Accelerates convergence (momentum=0.9)
- **Adaptive Learning Rate**: Decays by 5% every 500 iterations
- **Early Stopping**: Prevents unnecessary training

### Mathematical Foundation
- **Sigmoid Function**: σ(z) = 1 / (1 + e^(-z))
- **Cost Function**: Cross-entropy loss with L2 penalty
- **Optimization**: Gradient descent with momentum

## Evaluation Metrics

### Confusion Matrix
```
                Predicted
              Dry    Wet
Actual  Dry   TN     FP
        Wet   FN     TP
```

### Metrics Explained
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) - How many predicted wetlands are actually wetlands
- **Recall**: TP / (TP + FN) - How many actual wetlands were detected
- **F1-Score**: Harmonic mean of precision and recall

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom LR | 0.800 | 0.742 | 0.828 | 0.783 |
| Bagging | 0.823 | 0.793 | 0.802 | 0.798 |
| SVM | 0.810 | 0.741 | 0.867 | 0.799 |
| Random Forest | 0.810 | 0.773 | 0.798 | 0.785 |
| Decision Tree | 0.787 | 0.747 | 0.772 | 0.759 |
| Sklearn LR | 0.757 | 0.731 | 0.699 | 0.714 |

## Future Improvements
- Cross-validation for more robust evaluation
- Hyperparameter tuning using grid search
- Additional feature engineering (NDVI, NDWI indices)
- Deep learning approaches (CNN for spatial features)
- Cost-sensitive learning to handle class imbalance

## Contributors
- Ramey (Programming solutions and extra credit)
- Lay (Review and verification)

## License
Educational project for machine learning coursework.

## Acknowledgments
- Dataset: Wetland mapping project using multispectral imagery
- Scikit-learn library for baseline implementations
- Course materials and guidance from instructor

## Contact
For questions or issues, please contact through the course management system.

---
**Last Updated**: September 2025
