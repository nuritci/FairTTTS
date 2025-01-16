# FairTTTS: Bias Mitigation via Post-Processing Decision Adjustment

## Overview
Algorithmic decision-making is increasingly used in critical domains, yet biases in machine learning models can produce discriminatory outcomes. FairTTTS is a post-processing bias mitigation method designed to improve fairness while maintaining or enhancing predictive performance. It is inspired by the Tree Test Time Simulation (TTTS) technique and provides a fairness adjustment step for decision trees and random forests without requiring model retraining.

### Abstract
FairTTTS builds on TTTS, originally developed for improving accuracy and robustness against adversarial inputs. By applying a distance-based heuristic at protected attribute nodes, FairTTTS adjusts decisions for unprivileged groups, ensuring fairness. As a post-processing method, it works with pre-trained models, diverse datasets, and fairness metrics. 

Key highlights:
- **Fairness Improvement**: Achieves an average 20.96% increase in fairness metrics, outperforming related methods.
- **Accuracy Enhancement**: Improves accuracy by 0.55%, unlike competing methods that reduce it.
- **Applicability**: Supports various datasets and fairness criteria without requiring retraining.

## Repository Structure
The repository includes the following components:

- **Code Files**:
  - `FairTTTSRF.py`: Implementation of FairTTTS for Random Forest models.
  - `FairTTTSDT.py`: Implementation of FairTTTS for Decision Tree models.

- **Datasets**:
  - `MIMIC2.csv`
  - `adult.csv`
  - `bank.csv`
  - `compas-scores-two-years_v1.csv`
  - `diabetes_prediction_dataset.csv`
  - `german_credit_data.csv`
  - `recruitmentdataset-2022-1.3.csv`

- **Documentation**:
  - `README.md`: Instructions and details about the repository.

## Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/your-repository/FairTTTS.git
cd FairTTTS
```

## Usage
To run FairTTTS, execute the respective Python file for your model type, passing the `alpha` parameter:
```bash
python FairTTTSRF.py --alpha <integer>
```
or
```bash
python FairTTTSDT.py --alpha <integer>
```

### Parameters
- `--alpha`: A positive integer greater than 0. Recommended values are between **5 and 10**.

### Example
```bash
python FairTTTSRF.py --alpha 7
```

## Data
The `Data` folder contains benchmark datasets used for evaluating FairTTTS. These include commonly used datasets for fairness research such as `adult.csv` and `compas-scores-two-years_v1.csv`.

## Methodology
FairTTTS adjusts decisions in decision paths by:
1. Applying a distance-based heuristic at protected attribute nodes.
2. Ensuring fairness for unprivileged groups.
3. Enhancing accuracy and fairness through post-processing adjustments.

The method supports a range of pre-trained models and fairness metrics, making it versatile and easy to integrate.

## Results
Extensive testing across seven benchmark datasets shows:
- **Fairness Improvement**: 20.96% average increase.
- **Accuracy Impact**: +0.55% on average.

In contrast, competing methods often result in reduced accuracy (-0.42%).

