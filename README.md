# Heart Disease Prediction

This project is a machine learning practical project for the 2025-2026 academic year.
It implements a heart disease prediction workflow using the UCI Cleveland heart disease dataset.
#link colab:
https://colab.research.google.com/drive/1mIHeRBkTLOmR1Fai6ZvGVmu9wSrTQd6a?usp=sharing
## Files
- `Heart Disease Prediction.ipynb` - Jupyter Notebook with the full analysis workflow.
- `heart_disease_data.csv` - Dataset used for model training and evaluation.
- `generate_screenshots.py` - Script to create evaluation plots and save them as PNG files.
- `screenshots/` - Saved images for the confusion matrix, ROC curve, and correlation matrix.

## How to run
1. Install required Python packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
2. Open `Heart Disease Prediction.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the notebook cells from top to bottom.
4. To regenerate screenshots, run:
   ```bash
   python generate_screenshots.py
   ```

## Notebook structure
1. Importing libraries
2. Loading dataset and previewing data
3. Data preprocessing
4. Model building and training
5. Model evaluation
6. Visualization
7. Reflections on model performance
