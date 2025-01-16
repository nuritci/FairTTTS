# FairTTTS: Bias Mitigation via Post-Processing Decision Adjustment

## Overview
FairTTTS builds on TTTS, originally developed for improving accuracy and robustness against adversarial inputs. By applying a distance-based heuristic at protected attribute nodes, FairTTTS adjusts decisions for unprivileged groups, ensuring fairness. As a post-processing method, it works with pre-trained models, diverse datasets, and fairness metrics. 


- **Applicability**: Supports various datasets, model architectures, and fairness criteria without requiring retraining.

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
- `--alpha`: A positive integer greater than 0 that indicates a positive amplification toward fairness. Recommended values are between **5 and 10**.

### Example
```bash
python FairTTTSRF.py --alpha 7
```

## Data
The `Data` folder contains benchmark datasets used for evaluating FairTTTS. These include commonly used datasets for fairness research such as `adult.csv` and `compas-scores-two-years_v1.csv`.



