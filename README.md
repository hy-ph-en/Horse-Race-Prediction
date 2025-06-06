﻿# Horse Racing Prediction System

A sophisticated machine learning system for predicting horse racing outcomes using ensemble methods and comprehensive feature engineering. This project implements a stacked ensemble model combining multiple algorithms to generate probabilistic predictions for horse race winners with **demonstrated edge over betting markets**.

## 🏇 Overview

This system processes historical horse racing data to predict race outcomes using:
- **Advanced Feature Engineering**: 50+ racing-specific features including form analysis, trainer/jockey ratings, and market wisdom
- **Ensemble Learning**: Stacked model combining LightGBM, XGBoost, Neural Networks, and Logistic Regression
- **Meta-Learning**: XGBoost meta-model trained on base model predictions
- **Probability Calibration**: Configurable softmax normalization and calibration methods
- **Market Analysis**: Comprehensive comparison against betting market predictions
- **Value Identification**: Systematic detection of betting opportunities with positive expected value

## 🎯 Key Performance Highlights

### **Model vs Market Performance**
- **Model AUC**: 0.7723 vs **Market AUC**: 0.7595 (**+1.28% improvement**)
- **Market Correlation**: 0.6448 (strong agreement with independent insights)
- **Edge Opportunities**: 1,513 scenarios with **>10% positive edge**
- **Simulated ROI**: **130.5%** on high-edge betting opportunities

### **High Confidence Performance**
- **>50% Confidence Picks**: 104 selections with **60.6% hit rate**
- **>60% Confidence Picks**: 8 selections with **75% hit rate**
- **Non-Market Favorites**: 57.6% hit rate on high-confidence longshots

### **Value Betting Analysis**
- **Best ROI Strategy**: Edge >10%, Confidence >5% = **130.5% ROI**
- **Conservative Strategy**: Edge >5%, Confidence >10% = **86% ROI**
- **High Volume Strategy**: 3,574 opportunities with **18% hit rate** at favorable odds

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

### Run Market Analysis

```python
# Compare your model against betting markets
from Evaluation.market_analysis import run_market_analysis
results = run_market_analysis()
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
│   ├── feature_preparation.py     # Feature scaling and encoding
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
    ├── diagnostics.py             # Model diagnostics and testing
    └── market_analysis.py         # Market comparison and value analysis
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

# Diagnostics
show_detailed_diagnostics = False  # Enable detailed analysis
```

## 🎯 Features

### Racing-Specific Features
- **Form Analysis**: Speed ratings, position trends, improvement patterns
- **Market Intelligence**: Betting odds analysis, implied probabilities (not used as features)
- **Entity Ratings**: Trainer, jockey, and breeding (sire/damsire) ratings
- **Temporal Features**: Layoff periods, seasonal patterns, race timing
- **Competition Context**: Field size effects, course specialization

### Advanced Feature Types
- **Differential Features**: Horse performance vs. field averages
- **Ranking Features**: Within-race position rankings
- **Historical Aggregations**: Career statistics and recent form
- **Interaction Features**: Trainer-jockey combinations, course-distance synergies

### Data Leakage Prevention
- **Strict Time Ordering**: Only historical data used for predictions
- **Market Separation**: Market odds used for evaluation only, never as features
- **GroupKFold Validation**: Prevents information leakage across races

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
- **Optimized Parameters**: Tuned for racing prediction performance

### Ensemble Strategy
- Out-of-fold predictions from base models serve as meta-features
- Meta-model learns optimal weighting and combination rules
- Final predictions normalized within each race to sum to 1.0
- Configurable softmax scaling for probability sharpening

## 📊 Evaluation & Market Analysis

### Core Performance Metrics
- **Log-Loss**: 0.2894 (primary optimization metric)
- **ROC-AUC**: 0.7723 (classification performance)
- **Brier Score**: 0.0841 (probabilistic accuracy)
- **Accuracy**: 89.6% at 0.5 threshold, 82.6% at optimal threshold

### Market Comparison Results
- **Market Outperformance**: AUC 0.7723 vs Market 0.7595 (+1.28%)
- **Edge Identification**: 1,513 opportunities with >10% positive edge
- **Value Betting ROI**: Up to 130.5% simulated returns
- **Calibration Quality**: Mean predicted probability (10.53%) matches actual (10.51%)

### Edge Analysis Buckets
| Edge Range | Count | Win Rate | Avg Odds | Performance |
|------------|-------|----------|----------|-------------|
| Very Positive (>15%) | 694 | 28.2% | 11.4 | **Excellent Value** |
| Positive (10-15%) | 819 | 22.1% | 15.4 | **Strong Value** |
| Small Positive (5-10%) | 2,061 | 12.4% | 23.8 | **Moderate Value** |
| Negative (<-10%) | 1,663 | 18.6% | 4.2 | **Avoid** |

### Validation Strategy
- **GroupKFold**: Prevents data leakage by grouping races
- **Temporal Splits**: Respects chronological order of races
- **Cross-Validation**: 5-fold validation for robust performance estimates
- **Market Backtesting**: Historical betting simulation analysis

## 💰 Value Betting Opportunities

### Systematic Edge Detection
The system identifies betting opportunities through:
- **Positive Edge Calculation**: Model probability > Market probability
- **Confidence Thresholds**: Minimum prediction confidence requirements
- **ROI Simulation**: Historical performance of betting strategies

### Recommended Strategies
1. **High Confidence + High Edge**: >50% confidence + >10% edge (60.6% hit rate)
2. **Volume Strategy**: >10% confidence + >5% edge (large sample, 86% ROI)
3. **Conservative**: >20% confidence + >15% edge (118% ROI, smaller sample)

### Risk Management
- **Avoid Negative Edge**: Model identifies 1,663 scenarios where market is superior
- **Diversification**: Multiple confidence and edge thresholds
- **Position Sizing**: Based on edge magnitude and confidence

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
- **Market Performance**: Optimization includes market comparison metrics

## 📈 Performance

The system achieves:
- **Log-Loss**: ~0.28
- **ROC-AUC**: ~0.77
- **Brier**: ~0.0836
- **Accuracy for Winner Prediction**: ~75%

*Performance varies based on data quality and racing conditions*

## 🔍 Data Requirements

### Required Columns
- **Race Info**: Race_Time, Race_ID, Course, Distance, Going
- **Horse Data**: Horse, Age, Position, timeSecs, pdsBeaten
- **Connections**: Trainer, Jockey, TrainerRating, JockeyRating
- **Form Data**: Speed_PreviousRun, NMFP, daysSinceLastRun
- **Market Data**: betfairSP (for evaluation only), MarketOdds_PreviousRun
- **Breeding**: SireRating, DamsireRating

### Data Format
- CSV files with consistent column naming
- Chronologically ordered races for temporal features
- Complete market and form data for feature engineering
- **Market data used for validation only, never as input features**

## 🚧 Development

### Adding New Features
1. Implement feature logic in `Data_Processing/feature_engineering.py`
2. Add feature to `configuration.py` feature lists
3. Test with `test_feature_engineering.py`
4. Validate no data leakage in market analysis
5. Update documentation

### Adding New Models
1. Create model file in `Model/Submodels/`
2. Implement `get_model()` function returning fitted estimator
3. Add import to `Model/stack.py`
4. Include in base_getters list
5. Test market performance impact

### Custom Evaluation Metrics
1. Add metric functions to `Evaluation/evaluation.py`
2. Include in evaluation pipeline
3. Add market comparison if relevant
4. Update diagnostic outputs

### Market Analysis Extensions
1. Modify `Evaluation/market_analysis.py` for new analyses
2. Add ROI calculations for new strategies
3. Include confidence interval analysis
4. Test on out-of-sample data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Contribution Guidelines
- Maintain data leakage prevention standards
- Include market analysis for model changes
- Add comprehensive tests for new features
- Update documentation and performance metrics

## 📧 Contact

For questions about the horse racing prediction system, please open an issue or contact the me

---

*Built with ❤️ for quantative people with a possible interest in horse racing analytics, but for all machine learning enthusiasts*
