# Heart Disease Prediction

Kaggle Playground (S6E2): Binary classification project to predict the presence or absence of heart disease based on patient clinical features.

## Dataset

- **Train**: 624,000 samples
- **Test**: 208,393 samples
- **Target**: Heart Disease (Presence/Absence)
- **Features**: 14 numeric features + ID

### Feature Breakdown

**Continuous Features (5)**
- Age, BP (Blood Pressure), Cholesterol, Max HR (Maximum Heart Rate), ST depression

**Discrete Features (8)** (treated as categorical)
- Sex, Chest pain type, FBS over 120, EKG results, Exercise angina, Slope of ST, Number of vessels fluro, Thallium

**ID**: Dropped during preprocessing

## Preprocessing Pipeline

1. **Feature Encoding**: One-hot encode discrete features
2. **Transformation**: Log1p transformation for right-skewed ST depression
3. **Scaling**: StandardScaler on continuous features (for linear models)
4. **Cross-Validation**: 5-fold stratified split
5. **Outlier Handling**: Kept (3σ detection); no removal

## Models

- **Baseline**: Logistic Regression (scaled data)
- **Best Model**: XGBoost (unscaled data)

### Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.7314 | 0.7398 | 0.7080 | 0.7237 | 0.8060 |
| XGBoost | 0.7450 | 0.7589 | 0.7207 | 0.7394 | 0.8187 |

**Improvement**: +1.57% ROC_AUC with XGBoost

## Project Structure

```
predict-heart-disease/
├── data/
│   ├── raw/                 # Original CSV files
│   ├── processed/           # Preprocessed datasets
│   └── submissions/         # Predictions output
├── notebooks/
│   └── project.ipynb        # Complete pipeline (EDA → Preprocessing → Modeling)
└── README.md
```

## Usage

### Run the Notebook

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
   ```

2. Open `notebooks/project.ipynb` and run all cells sequentially:
   - **Phase 1**: Feature Engineering & Encoding
   - **Phase 2**: Outlier & Distribution Handling
   - **Phase 3**: Scaling & Normalization
   - **Phase 4**: Train-Test Split & Validation
   - **Phase 5**: Feature Redundancy Analysis
   - **Phase 6**: Missing Values Verification
   - **Phase 7**: Model Training & Evaluation

### Output

- **predictions.csv**: Test set predictions with probabilities
  - Columns: `id`, `Heart Disease` (probability 0-1)
  - Location: `data/submissions/predictions.csv`

## Key Findings

### Feature Importance (XGBoost)
Top predictive features:
1. **Thallium** (Cramér's V: 0.606)
2. **Chest pain type** (0.525)
3. **Number of vessels fluro** (0.463)
4. **Exercise angina** (0.442)
5. **Max HR** (Spearman: -0.441)

### Data Quality
- **No missing values** after preprocessing
- **Outliers detected**: ~0.2-1.5% per continuous feature (kept for modeling)
- **Class balance**: ~54% Presence, 46% Absence (slight imbalance)
- **No data leakage** confirmed

## Notes

- Models trained with `class_weight='balanced'` to handle slight class imbalance
- Continuous features log-transformed when skewness |S| > 1
- Tree models (XGBoost) use unscaled data; linear models use scaled data
- CV strategy: Stratified 5-fold to preserve target distribution
- Baseline (Logistic Regression) establishes performance floor

## Files Generated

- `predictions.csv` - Test set predictions (id, probability)
- Cross-validation performance metrics
- Feature importance rankings
- Trained model artifacts

---

**Author**: Heart Disease Prediction Project  
**Last Updated**: February 2026
