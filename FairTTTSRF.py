import argparse
import warnings
import pandas as pd
import numpy as np
import time
import re
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, log_loss, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import Bunch
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn.utils.validation import check_is_fitted
from fairlearn.postprocessing import ThresholdOptimizer
import category_encoders as ce
import os

# Constants
ALPHA = 2
PROB_TYPE = "distance"
S = 100

# Argument Parsing
parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--alpha", type=float, default=2, help="Alpha")
parser.add_argument("--probtype", type=str, default="distance", help="Probtype")
args = parser.parse_args()
ALPHA = args.alpha
PROB_TYPE = args.probtype

class FairTTTSMonteCarloRandomForestClassifier(RandomForestClassifier):
    def __init__(self, prob_type='distance', n_simulations=S, protected_attribute_name=None,
                 privileged_group=1, unprivileged_group=0, favorable_class=1, unfavorable_class=0,
                 n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                 max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
                 oob_score=False, n_jobs=-1, random_state=111, verbose=0,
                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        if prob_type not in ['fixed', 'depth', 'certainty', 'agreement', 'distance', 'confidence', 'bayes']:
            raise ValueError('Invalid prob_type')
        self.prob_type = prob_type
        self.n_simulations = n_simulations
        self.protected_attribute_name = protected_attribute_name
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.favorable_class = favorable_class
        self.unfavorable_class = unfavorable_class

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        # Check if feature_names_in_ is available (it is for DataFrame inputs)
        if hasattr(self, 'feature_names_in_'):
            self.feature_index_to_name = {i: feature for i, feature in enumerate(self.feature_names_in_)}
            self.feature_name_to_index = {feature: i for i, feature in enumerate(self.feature_names_in_)}
            if isinstance(self.protected_attribute_name, str):
                if self.protected_attribute_name in self.feature_name_to_index:
                    self.protected_attribute_index_in_tree = self.feature_name_to_index[self.protected_attribute_name]
                else:
                    raise ValueError("Protected attribute name not found in feature names.")
            elif isinstance(self.protected_attribute_name, int):
                # If the user passed an index directly
                self.protected_attribute_index_in_tree = self.protected_attribute_name
            else:
                raise ValueError("Protected attribute must be a name or an index.")
        else:
            # X is likely a NumPy array, so we treat feature indices numerically
            n_features = X.shape[1]
            self.feature_index_to_name = {i: i for i in range(n_features)}
            self.feature_name_to_index = {i: i for i in range(n_features)}
            # protected_attribute_name should be the index in this case
            if isinstance(self.protected_attribute_name, int):
                if 0 <= self.protected_attribute_name < n_features:
                    self.protected_attribute_index_in_tree = self.protected_attribute_name
                else:
                    raise ValueError("Protected attribute index out of range.")
            else:
                raise ValueError("For NumPy arrays, protected_attribute_name should be an integer index.")

        self.class_label_to_index = {label: idx for idx, label in enumerate(self.classes_)}

        # Precompute tree-specific statistics
        self.precomputed_trees = []
        for estimator in self.estimators_:
            tree = estimator.tree_
            precomp = {}
            if self.prob_type == 'distance':
                thresholds = tree.threshold
                valid_thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
                precomp['max_distance'] = np.max(np.abs(valid_thresholds)) if len(valid_thresholds) > 0 else 1.0
            if self.prob_type == 'confidence':
                # We'll compute global stats in predict_proba
                precomp['feature_avgs'] = None
                precomp['feature_stds'] = None
            self.precomputed_trees.append(precomp)
        return self

    def predict_proba(self, X, n_simulations=None):
        if n_simulations is None:
            n_simulations = self.n_simulations
        check_is_fitted(self)
        X = self._validate_X_predict(X).astype(np.float32)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=np.float32)

        # Precompute feature-wise statistics if needed
        if self.prob_type == 'confidence':
            feature_avgs = np.mean(X, axis=0)
            feature_stds = np.std(X, axis=0) + 1e-9
        else:
            feature_avgs = None
            feature_stds = None

        # Build tree information
        trees_info = []
        for estimator, precomp in zip(self.estimators_, self.precomputed_trees):
            tree = estimator.tree_
            tree_info = {
                'children_left': tree.children_left,
                'children_right': tree.children_right,
                'features': tree.feature,
                'thresholds': tree.threshold,
                'values': tree.value,
                'max_distance': precomp.get('max_distance', None),
            }
            trees_info.append(tree_info)

        # Precompute more_negative_direction for all nodes in all trees
        more_negative_dirs = []
        for tree_info in trees_info:
            more_neg_dir = {}
            for node in range(len(tree_info['features'])):
                if tree_info['features'][node] == _tree.TREE_UNDEFINED:
                    continue
                left_child = tree_info['children_left'][node]
                right_child = tree_info['children_right'][node]
                if left_child == _tree.TREE_LEAF and right_child == _tree.TREE_LEAF:
                    more_neg_dir[node] = None
                    continue
                unfavorable_class_index = self.class_label_to_index[self.unfavorable_class]
                left_unfavorable = tree_info['values'][left_child][0][unfavorable_class_index] if left_child != _tree.TREE_LEAF else 0
                right_unfavorable = tree_info['values'][right_child][0][unfavorable_class_index] if right_child != _tree.TREE_LEAF else 0
                if left_unfavorable > right_unfavorable:
                    more_neg_dir[node] = 'left'
                elif right_unfavorable > left_unfavorable:
                    more_neg_dir[node] = 'right'
                else:
                    more_neg_dir[node] = 'left'
            more_negative_dirs.append(more_neg_dir)

        # Iterate over each sample and simulate paths
        for i in range(n_samples):
            x = X[i]
            sample_proba = np.zeros(n_classes, dtype=np.float32)
            for tree_idx, tree_info in enumerate(trees_info):
                more_neg_dir = more_negative_dirs[tree_idx]
                current_node = 0
                depth = 0
                while True:
                    feature_index = tree_info['features'][current_node]
                    if feature_index == _tree.TREE_UNDEFINED:
                        # Leaf node
                        sample_proba += tree_info['values'][current_node][0]
                        break

                    threshold = tree_info['thresholds'][current_node]
                    if x[feature_index] <= threshold:
                        direction = 'left'
                        go_left = True
                    else:
                        direction = 'right'
                        go_left = False

                    # Calculate p based on prob_type
                    if self.prob_type == 'fixed':
                        p = 0.2
                    elif self.prob_type == 'depth':
                        p = min(0.05 * depth, 0.2)
                    elif self.prob_type == 'certainty':
                        node_values = tree_info['values'][current_node].flatten()
                        total = np.sum(node_values)
                        distribution = node_values / total if total > 0 else np.zeros_like(node_values)
                        max_certainty = np.max(distribution)
                        p = min(1 - max_certainty, 0.5)
                    elif self.prob_type == 'agreement':
                        node_values = tree_info['values'][current_node].flatten()
                        majority_class_ratio = np.max(node_values) / np.sum(node_values) if np.sum(node_values) > 0 else 0
                        p = min(1 - majority_class_ratio, 0.5)
                    elif self.prob_type == 'distance':
                        distance = abs(x[feature_index] - threshold)
                        p = min(0.1 - min((distance / tree_info['max_distance']), 0.1), 0.5)
                    elif self.prob_type == 'confidence':
                        avg = feature_avgs[feature_index]
                        std = feature_stds[feature_index]
                        distance = abs(x[feature_index] - avg)
                        p = max(0.1 - (distance / std), 0)
                        p = min(p, 0.5)
                    elif self.prob_type == 'bayes':
                        p = 0  # Not implemented in original code
                    else:
                        raise ValueError('Invalid prob_type')

                    # Adjust p if conditions are met
                    more_neg = more_neg_dir.get(current_node, None)
                    if (direction == more_neg) and \
                       (x[self.protected_attribute_index_in_tree] == self.unprivileged_group) and \
                       (feature_index == self.protected_attribute_index_in_tree):
                        p = min(p * ALPHA, 0.5)

                    rnd = np.random.rand()
                    if go_left:
                        if rnd > p:
                            current_node = tree_info['children_left'][current_node]
                        else:
                            current_node = tree_info['children_right'][current_node]
                    else:
                        if rnd > p:
                            current_node = tree_info['children_right'][current_node]
                        else:
                            current_node = tree_info['children_left'][current_node]
                    depth += 1

            proba[i] = sample_proba / len(self.estimators_)
        return proba

# Define the base path for file storage and lists of dataset filenames
base_path = os.path.join(os.getcwd(), "../data/")
dataset_list = [
    'ADULT_SEX','CREDIT_SEX','RECRUIT_SEX','DIABETES_AGE','BANK_AGE',
    'COMPAS_RACE','ADULT_RACE','MIMIC_SEX'
]

def load_dataset(file_name):
    try:
        if file_name == 'ADULT_SEX':
            TARGET_COL = 'income'
            PRIV_FEATURE = "sex"
            PRIV_GROUP = 0
            UNPRIV_GROUP = 1
            FAVORABLE_CLASS = 1
            UNFAVORABLE_CLASS = 0
            data = pd.read_csv(base_path+'adult.csv')
            data[TARGET_COL] = data[TARGET_COL].apply(lambda x: 0 if x == '<=50K' else 1)
            data['sex'] = np.where(data['sex'] == 'Male', 0,1)
            data['white'] = np.where(data['race'] == 'White', 1,0)
            feature_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
                               'marital.status', 'occupation', 'relationship', 'race', 'sex',
                               'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'white']
            CATEGORICAL = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country']

        elif file_name == 'CREDIT_SEX':
            data = pd.read_csv(base_path+'german_credit_data.csv')
            TARGET_COL = 'Risk'
            PRIV_FEATURE = 'Sex'
            PRIV_GROUP = 0
            UNPRIV_GROUP = 1
            FAVORABLE_CLASS = 1
            UNFAVORABLE_CLASS = 0
            data[TARGET_COL] = data[TARGET_COL].apply(lambda x: 0 if x == 'bad' else 1)
            data['Sex'] = np.where(data['Sex'] == 'male', 0,1)
            data = data.drop(["Unnamed: 0"],axis=1)
            data['young'] = data['Age'].apply(lambda x: 0 if x < 25 else 1)
            CATEGORICAL = ['Housing', 'Saving accounts' , 'Checking account', 'Purpose']
            feature_columns = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account','Credit amount', 'Duration', 'Purpose']

        elif file_name == 'RECRUIT_SEX':
            data = pd.read_csv(base_path+'recruitmentdataset-2022-1.3.csv')
            TARGET_COL = 'decision'
            PRIV_FEATURE = 'gender'
            PRIV_GROUP = 0
            UNPRIV_GROUP = 1
            FAVORABLE_CLASS = 1
            UNFAVORABLE_CLASS = 0
            data['gender'] = np.where(data['gender'] == 'male', 0,1)
            data['decision'] = np.where(data['decision'] == False, 0,1)
            CATEGORICAL = ['nationality', 'sport','ind-debateclub', 'ind-programming_exp', 'ind-international_exp',
                           'ind-entrepeneur_exp', 'ind-exact_study', 'ind-degree', 'company']
            feature_columns = ['gender', 'age', 'nationality', 'sport', 'ind-university_grade',
                               'ind-debateclub', 'ind-programming_exp', 'ind-international_exp',
                               'ind-entrepeneur_exp', 'ind-languages', 'ind-exact_study', 'ind-degree',
                               'company']

        elif file_name == 'DIABETES_AGE':
            data = pd.read_csv(base_path+'diabetes_prediction_dataset.csv')
            TARGET_COL = 'diabetes'
            data['age_binary'] = data['age'].apply(lambda x: 0 if x < 40 else 1)
            PRIV_FEATURE = 'age_binary'
            PRIV_GROUP = 1
            UNPRIV_GROUP = 0
            FAVORABLE_CLASS = 0
            UNFAVORABLE_CLASS = 1
            feature_columns = ['gender', 'age_binary', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            CATEGORICAL = ['smoking_history', 'gender']

        elif file_name == 'BANK_AGE':
            data = pd.read_csv(base_path+'bank.csv')
            data['age'] = np.where(data['age'] < 40, 0, 1)
            data['deposit'] = np.where(data['deposit'] == "no", 0,1)
            TARGET_COL = 'deposit'
            PRIV_FEATURE = 'age'
            PRIV_GROUP = 1
            UNPRIV_GROUP = 0
            FAVORABLE_CLASS = 1
            UNFAVORABLE_CLASS = 0
            feature_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                               'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
            CATEGORICAL = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

        elif file_name == 'COMPAS_RACE':
            data = pd.read_csv(base_path+'compas-scores-two-years_v1.csv')
            TARGET_COL = 'two_year_recid'
            data['race'] = np.where(data['race'] == 'African-American', 0,1)
            PRIV_FEATURE = 'race'
            PRIV_GROUP = 1
            UNPRIV_GROUP = 0
            FAVORABLE_CLASS = 0
            UNFAVORABLE_CLASS = 1
            feature_columns = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
                               'priors_count', 'c_charge_degree', 'score_text', 'v_score_text']
            CATEGORICAL = ['sex', 'age_cat', 'c_charge_degree', 'score_text', 'v_score_text']

        elif file_name == 'ADULT_RACE':
            TARGET_COL = 'income'
            PRIV_FEATURE = 'race'
            data = pd.read_csv(base_path+'adult.csv')
            data[TARGET_COL] = data[TARGET_COL].apply(lambda x: 0 if x == '<=50K' else 1)
            data['race'] = np.where(data['race'] == 'Black', 0,1)
            PRIV_GROUP = 1
            UNPRIV_GROUP = 0
            FAVORABLE_CLASS = 1
            UNFAVORABLE_CLASS = 0
            data['white'] = np.where(data['race'] == 'White', 1,0)
            feature_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
                               'marital.status', 'occupation', 'relationship', 'race', 'sex',
                               'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'white']
            CATEGORICAL = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'sex', 'native.country']

        elif file_name == 'MIMIC_SEX':
            TARGET_COL = 'outcome'
            PRIV_FEATURE = 'gendera'
            PRIV_GROUP = 1
            UNPRIV_GROUP = 0
            FAVORABLE_CLASS = 1
            UNFAVORABLE_CLASS = 0
            data = pd.read_csv(base_path+'MIMIC2.csv')
            data['gendera'] = np.where(data['gendera'] == 2, 0, 1)
            feature_columns = ['group', 'age', 'gendera', 'BMI', 'hypertensive',
                               'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias',
                               'depression', 'Hyperlipemia', 'Re-999l failure', 'COPD', 'heart rate',
                               'Systolic blood pressure', 'Diastolic blood pressure',
                               'Respiratory rate', 'temperature', 'SP O2', 'Urine output',
                               'hematocrit', 'RBC', 'MCH', 'MCHC', 'MCV', 'RDW', 'Leucocyte',
                               'Platelets', 'Neutrophils', 'Basophils', 'Lymphocyte', 'PT', 'INR',
                               'NT-proBNP', 'Creatine ki-999se', 'Creatinine', 'Urea nitrogen',
                               'glucose', 'Blood potassium', 'Blood sodium', 'Blood calcium',
                               'Chloride', 'Anion gap', 'Magnesium ion', 'PH', 'Bicarbo-999te',
                               'Lactic acid', 'PCO2', 'EF']
            CATEGORICAL = [
                'group', 'hypertensive', 'atrialfibrillation',
                'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression',
                'Hyperlipemia', 'Re-999l failure', 'COPD'
            ]

        else:
            raise ValueError(f"Unknown dataset: {file_name}")

        # Drop rows with missing values in feature_columns
        data = data.dropna(subset=feature_columns)
        data.reset_index(drop=True, inplace=True)

        if PRIV_FEATURE not in feature_columns:
            feature_columns.append(PRIV_FEATURE)

        encoder = ce.TargetEncoder(cols=CATEGORICAL)
        encoder.fit(data, data[TARGET_COL])
        transformed_data = encoder.transform(data)

        X = transformed_data[feature_columns]
        y = data[TARGET_COL]
        assert X.index.equals(y.index), "Indices of X and y do not align!"

        dataset_bunch = Bunch(data=X, target=y)
        dataset_bunch.data.columns = feature_columns
        dataset_bunch.privileged_feature = PRIV_FEATURE
        dataset_bunch.privileged_group = PRIV_GROUP
        dataset_bunch.unprivileged_group = UNPRIV_GROUP
        dataset_bunch.favorable_class = FAVORABLE_CLASS
        dataset_bunch.unfavorable_class = UNFAVORABLE_CLASS
        dataset_bunch.favour_group = PRIV_GROUP
        dataset_bunch.unfavour_group = UNPRIV_GROUP

        return (file_name, dataset_bunch)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

datasets = Parallel(n_jobs=-1)(
    delayed(load_dataset)(file_name) for file_name in dataset_list
)
datasets = [dataset for dataset in datasets if dataset is not None]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def compute_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan, np.nan, np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size != 4:
        tn = fp = fn = tp = 0
        if cm.shape == (1,1):
            if y_true[0] == y_pred[0]:
                if y_true[0] == 0:
                    tn = cm[0,0]
                else:
                    tp = cm[0,0]
        elif cm.shape == (2,1):
            tn, fn = cm[0,0], cm[1,0]
        elif cm.shape == (1,2):
            tn, fp = cm[0,0], cm[0,1]
    else:
        tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    return fpr, tpr, acc

def opportunity_diff_tpr(tpr1, tpr2):
    if np.isnan(tpr1) or np.isnan(tpr2):
        return np.nan
    return abs(tpr1 - tpr2)

def odds_diff(tpr1, tpr2, fpr1, fpr2):
    if any(np.isnan([tpr1, tpr2, fpr1, fpr2])):
        return np.nan
    return 0.5 * (abs(tpr1 - tpr2) + abs(fpr1 - fpr2))

def apply_reject_option_classification(predictions, protected_attribute, threshold=0.5, uncertainty_range=0.05, privileged_group=1, favorable_class=1):
    corrected_predictions = np.zeros(predictions.shape)
    decision_boundary_lower = threshold - uncertainty_range
    decision_boundary_upper = threshold + uncertainty_range
    for i, pred in enumerate(predictions):
        pred_class = int(pred > threshold)
        if decision_boundary_lower < pred < decision_boundary_upper:
            if protected_attribute[i] != privileged_group:
                if pred_class != favorable_class:
                    corrected_predictions[i] = 1 - pred_class
                else:
                    corrected_predictions[i] = pred_class
            else:
                corrected_predictions[i] = pred_class
        else:
            corrected_predictions[i] = pred_class
    return corrected_predictions

def evaluate_classifier(dataset_name, dataset, clf_name, clf, train_index, test_index):
    X = dataset.data.fillna(0).values.astype(np.float32)
    y = dataset.target.values
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    protected_attribute_train = dataset.data[dataset.privileged_feature].values[train_index].astype(np.int32)
    protected_attribute_test = dataset.data[dataset.privileged_feature].values[test_index].astype(np.int32)

    # Determine the index of the protected attribute
    if hasattr(dataset.data, 'columns'):
        try:
            protected_attribute_index = list(dataset.data.columns).index(dataset.privileged_feature)
        except ValueError:
            raise ValueError(f"Protected feature '{dataset.privileged_feature}' not found in feature columns.")
    else:
        protected_attribute_index = dataset.privileged_feature

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(f"Skipping fold for {dataset_name} with classifier {clf_name} due to only one class in y_train.")
        return [dataset_name, clf_name, np.nan, np.nan, np.nan, np.nan, np.nan]

    if clf_name == "RandomForest_baseline":
        clf = RandomForestClassifier(random_state=111, n_jobs=-1)
        clf.fit(X_train, y_train)
    elif clf_name == "MonteCarloRandomForest":
        clf = FairTTTSMonteCarloRandomForestClassifier(
            prob_type=PROB_TYPE,
            protected_attribute_name=protected_attribute_index,
            privileged_group=dataset.privileged_group,
            unprivileged_group=dataset.unprivileged_group,
            favorable_class=dataset.favorable_class,
            unfavorable_class=dataset.unfavorable_class,
            n_estimators=100,
            random_state=111,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
    elif clf_name == "RandomForest_RejectedOption":
        clf = RandomForestClassifier(random_state=111, n_jobs=-1)
        clf.fit(X_train, y_train)
    elif clf_name == "ThresholdOptimizer":
        base_estimator = RandomForestClassifier(random_state=111, n_jobs=-1)
        base_estimator.fit(X_train, y_train)
        clf = ThresholdOptimizer(
            estimator=base_estimator,
            constraints="equalized_odds",
            prefit=True,
            predict_method='predict_proba'
        )
        clf.fit(X_train, y_train, sensitive_features=protected_attribute_train)
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

    start_time = time.time()

    if clf_name == "ThresholdOptimizer":
        preds = clf.predict(X_test, sensitive_features=protected_attribute_test)
    else:
        pred_probs = clf.predict_proba(X_test)
        if len(clf.classes_) > 1:
            if clf.classes_[1] == dataset.favorable_class:
                favorable_class_index = 1
            else:
                favorable_class_index = 0
        else:
            favorable_class_index = 0

        if clf_name == "RandomForest_RejectedOption":
            preds_probs = pred_probs[:, favorable_class_index]
            preds = apply_reject_option_classification(
                preds_probs,
                protected_attribute_test,
                threshold=0.5,
                uncertainty_range=0.05,
                privileged_group=dataset.privileged_group,
                favorable_class=dataset.favorable_class
            )
        else:
            preds = np.argmax(pred_probs, axis=1)

    runtime = time.time() - start_time
    accuracy = accuracy_score(y_test, preds)

    privileged_indices = np.where(protected_attribute_test == dataset.privileged_group)[0]
    unprivileged_indices = np.where(protected_attribute_test == dataset.unprivileged_group)[0]

    if len(privileged_indices) == 0:
        print(f"No privileged samples in the test set for {dataset_name} with classifier {clf_name}.")
        fpr_priv, tpr_priv, acc_priv = np.nan, np.nan, np.nan
    else:
        fpr_priv, tpr_priv, acc_priv = compute_metrics(y_test[privileged_indices], preds[privileged_indices])

    if len(unprivileged_indices) == 0:
        print(f"No unprivileged samples in the test set for {dataset_name} with classifier {clf_name}.")
        fpr_unpriv, tpr_unpriv, acc_unpriv = np.nan, np.nan, np.nan
    else:
        fpr_unpriv, tpr_unpriv, acc_unpriv = compute_metrics(y_test[unprivileged_indices], preds[unprivileged_indices])

    equal_opportunity_difference = opportunity_diff_tpr(tpr_priv, tpr_unpriv)
    average_odds_difference = odds_diff(tpr_priv, tpr_unpriv, fpr_priv, fpr_unpriv)

    if not np.isnan(np.mean(preds[privileged_indices])) and not np.isnan(np.mean(preds[unprivileged_indices])):
        statistical_parity_difference = np.mean(preds[unprivileged_indices]) - np.mean(preds[privileged_indices])
        if np.mean(preds[privileged_indices]) > 0:
            disparate_impact = np.mean(preds[unprivileged_indices]) / np.mean(preds[privileged_indices])
        else:
            disparate_impact = np.inf
    else:
        statistical_parity_difference = np.nan
        disparate_impact = np.nan

    return [
        dataset_name,
        clf_name,
        accuracy,
        disparate_impact,
        statistical_parity_difference,
        equal_opportunity_difference,
        runtime
    ]

classifiers = {
    "RandomForest_baseline": RandomForestClassifier(),
    "RandomForest_RejectedOption": RandomForestClassifier(),
    "MonteCarloRandomForest": FairTTTSMonteCarloRandomForestClassifier(),
    "ThresholdOptimizer": ThresholdOptimizer(),
}

all_tasks = []
for dataset_name, dataset in datasets:
    print(f"Processing dataset: {dataset_name}")
    X, y = dataset.data.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32), dataset.target.fillna(0).values
    for clf_name, clf in classifiers.items():
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            task = delayed(evaluate_classifier)(
                dataset_name, dataset, clf_name, clf, train_index, test_index
            )
            all_tasks.append(task)

results = Parallel(n_jobs=-1)(all_tasks)
filtered_results = [result for result in results if not all(pd.isna(result[2:]))]

results_df = pd.DataFrame(
    filtered_results,
    columns=["Dataset", "Classifier", "Accuracy", "Disparate Impact", "Statistical Parity Difference", "Equalized Odds Difference", "Runtime"],
)

summary_df = results_df.groupby(['Dataset','Classifier']).agg({
    'Accuracy': ['mean', 'std'],
    'Disparate Impact': ['mean', 'std'],
    'Statistical Parity Difference': ['mean', 'std'],
    'Equalized Odds Difference': ['mean', 'std'],
    'Runtime': ['mean', 'std']
}).reset_index()

print(summary_df)
summary_df.to_csv(f'FairTTTSRFMethod_{PROB_TYPE}_{ALPHA}_{S}_RandomForest.csv', index=False)
