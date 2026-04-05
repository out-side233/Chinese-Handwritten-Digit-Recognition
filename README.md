# Chinese-Handwritten-Character-Recognition
Handwritten Chinese  recognition using Random Forest and SVM with PCA optimization. Achieved ~93% accuracy on test sets.
This project implements handwritten character recognition using **Random Forest** and **SVM** algorithms.

##  Performance
- **Random Forest**: 93.1% Accuracy
- **SVM (PCA Optimized)**: 91.8% Accuracy

##  Requirements
Install dependencies via:
`pip install -r requirements.txt`

##  Data Structure
To run the code, ensure your local directory has the following structure:
- `data/`: Contains subfolders of different characters expressed by `0` to `9` with `.png` images.
- `src/`: Python scripts.

##  Methodology
- **Preprocessing**: 14x14 Resizing & Pixel Normalization.
- **Dimensionality Reduction**: PCA (85% variance maintained).
- **Hyperparameter Tuning**: GridSearchCV.
