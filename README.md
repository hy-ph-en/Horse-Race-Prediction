# Horse Racing Prediction System

A sophisticated machine learning system for predicting horse racing outcomes using ensemble methods and comprehensive feature engineering. This project implements a stacked ensemble model combining multiple algorithms to generate probabilistic predictions for horse race winners.

## 🏇 Overview

This system processes historical horse racing data to predict race outcomes using:
- **Advanced Feature Engineering**: 50+ racing-specific features including form analysis, trainer/jockey ratings, and market wisdom
- **Ensemble Learning**: Stacked model combining LightGBM, XGBoost, Neural Networks, and Logistic Regression
- **Meta-Learning**: XGBoost meta-model trained on base model predictions
- **Probability Calibration**: Configurable softmax normalization and calibration methods
- **Comprehensive Evaluation**: Multiple metrics including log-loss, ROC-AUC, and racing-specific metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, xgboost, lightgbm, optuna

### Basic Usage

```python
# Run the complete pipeline
from main_run import run
model_components, metrics = run()
```

### Manual Steps

```python
# 1. Data Preprocessing
from Data_Processing.preprocessing import Preprocessing
data = Preprocessing().check_and_preprocess()

# 2. Train Model
from Model.stack import model_run
model_components, train_data = model_run(data)

# 3. Evaluate Model
from Evaluation.evaluation import evaluate_model
metrics = evaluate_model(model_components, train_data)
```

## 📁 Project Structure

```
├── main_run.py                     # Main execution script
├── configuration.py                # Configuration settings
├── hyperparameter_optimization.py  # Automated hyperparameter tuning
├── submission.csv                   # Model predictions output
│
├── Data/                           # Raw and processed datasets
│   ├── trainData.csv               # Training data
│   ├── testData.csv                # Test data
│   ├── trainData_clean.csv         # Cleaned training data
│   └── testData_clean.csv          # Cleaned test data
│
├── Data_Processing/                # Data preprocessing pipeline
│   ├── preprocessing.py            # Main preprocessing orchestrator
│   ├── data_cleaner.py            # Data cleaning functions
│   ├── feature_engineering.py     # Feature creation pipeline
│   ├── metrics.py                 # Custom metrics and transformations
│   ├── Feature_Discussion.md      # Feature documentation
│   └── test_feature_engineering.py # Feature engineering tests
│
├── Model/                         # Machine learning models
│   ├── stack.py                   # Stacking ensemble implementation
│   └── Submodels/                 # Individual base models
│       ├── light_gmb.py           # LightGBM model
│       ├── xgboost.py             # XGBoost model
│       ├── neuralnet.py           # Neural Network model
│       └── logistic_regression.py # Logistic Regression model
│
└── Evaluation/                    # Model evaluation and diagnostics
    ├── evaluation.py              # Evaluation metrics and analysis
    └── diagnostics.py             # Model diagnostics and testing
```

## 🔧 Configuration

The system uses `configuration.py` for centralized settings:

### Key Configuration Options

```python
# Feature Selection
feature_cols = [
    'Speed_PreviousRun_diff', 'Speed_PreviousRun_rank',
    'TrainerRating_diff', 'TrainerRating_rank',
    'JockeyRating_diff', 'JockeyRating_rank',
    # ... additional features
]

# Model Calibration
softmax_scale_factor = 6           # Probability normalization strength
use_meta_calibration = False       # Enable post-hoc calibration
calibration_method = 'isotonic'    # 'isotonic' or 'sigmoid'

# Alternative Scaling
use_alternative_scaling = False    # Use weighted normalization
epsilon_scaling = 1e-8            # Epsilon for weighted scaling
```

## 🎯 Features

### Racing-Specific Features
- **Form Analysis**: Speed ratings, position trends, improvement patterns
- **Market Intelligence**: Betting odds analysis, implied probabilities
- **Entity Ratings**: Trainer, jockey, and breeding (sire/damsire) ratings
- **Temporal Features**: Layoff periods, seasonal patterns, race timing
- **Competition Context**: Field size effects, course specialization

### Advanced Feature Types
- **Differential Features**: Horse performance vs. field averages
- **Ranking Features**: Within-race position rankings
- **Historical Aggregations**: Career statistics and recent form
- **Interaction Features**: Trainer-jockey combinations, course-distance synergies

## 🤖 Models

### Base Models
1. **LightGBM**: Gradient boosting optimized for speed and memory efficiency
2. **XGBoost**: Robust gradient boosting with advanced regularization
3. **Neural Network**: Multi-layer perceptron for non-linear pattern recognition
4. **Logistic Regression**: Linear baseline with regularization

### Meta-Learning
- **XGBoost Meta-Model**: Learns optimal combination of base model predictions
- **Cross-Validation**: 5-fold GroupKFold by Race_ID to prevent data leakage
- **Probability Calibration**: Optional isotonic/sigmoid calibration

### Ensemble Strategy
- Out-of-fold predictions from base models serve as meta-features
- Meta-model learns optimal weighting and combination rules
- Final predictions normalized within each race to sum to 1.0

## 📊 Evaluation

### Metrics
- **Log-Loss**: Primary optimization metric for probabilistic predictions
- **ROC-AUC**: Area under ROC curve for binary classification performance
- **Top-N Accuracy**: Percentage of winners in top-N predictions
- **Profit/Loss**: Simulated betting performance analysis

### Validation Strategy
- **GroupKFold**: Prevents data leakage by grouping races
- **Temporal Splits**: Respects chronological order of races
- **Cross-Validation**: 5-fold validation for robust performance estimates

## 🎛️ Hyperparameter Optimization

Automated optimization using Optuna:

```python
from hyperparameter_optimization import run_hyperparameter_optimization
run_hyperparameter_optimization()
```

### Optimization Features
- **Multi-Model Optimization**: Optimizes all base models and meta-model
- **Bayesian Search**: Efficient parameter space exploration
- **Cross-Validation**: Robust parameter evaluation
- **Early Stopping**: Prevents overfitting during optimization

## 📈 Performance

The system achieves:
- **Log-Loss**: ~0.29
- **ROC-AUC**: ~0.77
- **Brier**: ~0.0846
- **Accuracy for Winner Prediction**: ~65%

*Performance varies based on data quality and racing conditions*

## 🔍 Data Requirements

### Required Columns
- **Race Info**: Race_Time, Race_ID, Course, Distance, Going
- **Horse Data**: Horse, Age, Position, timeSecs, pdsBeaten
- **Connections**: Trainer, Jockey, TrainerRating, JockeyRating
- **Form Data**: Speed_PreviousRun, NMFP, daysSinceLastRun
- **Market Data**: betfairSP, MarketOdds_PreviousRun
- **Breeding**: SireRating, DamsireRating

### Data Format
- CSV files with consistent column naming
- Chronologically ordered races for temporal features
- Complete market and form data for feature engineering

## 🚧 Development

### Adding New Features
1. Implement feature logic in `Data_Processing/feature_engineering.py`
2. Add feature to `configuration.py` feature lists
3. Test with `test_feature_engineering.py`
4. Update documentation

### Adding New Models
1. Create model file in `Model/Submodels/`
2. Implement `get_model()` function returning fitted estimator
3. Add import to `Model/stack.py`
4. Include in base_getters list

### Custom Evaluation Metrics
1. Add metric functions to `Evaluation/evaluation.py`
2. Include in evaluation pipeline
3. Update diagnostic outputs

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📧 Contact

For questions about the horse racing prediction system, please open an issue or contact the me

---

*Built with ❤️ for quantative people with a possible interest in horse racing analytics, but for all machine learning enthusiasts*
