import argparse
import warnings
import pandas as pd
import numpy as np
import time
import os
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.utils import Bunch
from fairlearn.postprocessing import ThresholdOptimizer
import category_encoders as ce

#######################################################################
# Configuration and Argument Parsing
#######################################################################
ALPHA = 2
PROB_TYPE = "distance"
S = 250

parser = argparse.ArgumentParser(description="Run experiment with FairTTTS on a Decision Tree and compare methods.")
parser.add_argument("--alpha", type=float, default=2, help="Alpha")
parser.add_argument("--probtype", type=str, default="distance", help="Probtype")
args = parser.parse_args()
ALPHA = args.alpha
PROB_TYPE = args.probtype

#######################################################################
# FairTTTS MonteCarloDecisionTreeClassifier
#######################################################################
class FairTTTSMonteCarloDecisionTreeClassifier(DecisionTreeClassifier):
    """
    A Decision Tree classifier that applies Monte Carlo simulations during prediction
    to mitigate bias towards unprivileged groups, based on the FairTTTS method.

    Parameters:
    - prob_type (str): Probability type used at each node.
    - n_simulations (int): Number of simulations for averaging predictions.
    - protected_attribute_name (str or int): Name or index of protected attribute.
    - privileged_group (int): Value representing the privileged group.
    - unprivileged_group (int): Value representing the unprivileged group.
    - favorable_class (int): Favorable class label.
    - unfavorable_class (int): Unfavorable class label.
    """
    def __init__(self, prob_type='distance', n_simulations=S, protected_attribute_name=None,
                 privileged_group=1, unprivileged_group=0, favorable_class=1, unfavorable_class=0,
                 criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                 min_samples_leaf=20, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=111, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 class_weight=None, ccp_alpha=0.0):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha
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
        X, y = check_X_y(X, y, dtype=np.float32)
        super().fit(X, y, sample_weight=sample_weight)
        self.class_label_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        # Determine protected attribute index
        if hasattr(self, 'feature_names_in_'):
            if isinstance(self.protected_attribute_name, str):
                if self.protected_attribute_name not in self.feature_names_in_:
                    raise ValueError("Protected attribute name not found in feature names.")
                self.protected_attribute_index_in_tree = np.where(self.feature_names_in_ == self.protected_attribute_name)[0][0]
            elif isinstance(self.protected_attribute_name, int):
                self.protected_attribute_index_in_tree = self.protected_attribute_name
            else:
                raise ValueError("Protected attribute must be a name or an index.")
        else:
            n_features = X.shape[1]
            if isinstance(self.protected_attribute_name, int):
                if 0 <= self.protected_attribute_name < n_features:
                    self.protected_attribute_index_in_tree = self.protected_attribute_name
                else:
                    raise ValueError("Protected attribute index out of range.")
            else:
                raise ValueError("For NumPy input, protected_attribute_name should be an integer index.")
        return self

    def _get_more_negative_direction(self, node_id, tree):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if left_child == _tree.TREE_LEAF and right_child == _tree.TREE_LEAF:
            return None
        unfavorable_class_index = self.class_label_to_index[self.unfavorable_class]
        left_unfavorable = tree.value[left_child][0][unfavorable_class_index] if left_child != _tree.TREE_LEAF else 0
        right_unfavorable = tree.value[right_child][0][unfavorable_class_index] if right_child != _tree.TREE_LEAF else 0
        if left_unfavorable > right_unfavorable:
            return 'left'
        elif right_unfavorable > left_unfavorable:
            return 'right'
        else:
            return 'left'

    def predict_proba(self, X, n_simulations=None):
        if n_simulations is None:
            n_simulations = self.n_simulations
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)
        tree = self.tree_

        # Precompute stats if needed
        if self.prob_type == 'distance':
            valid_thresholds = tree.threshold[tree.threshold != _tree.TREE_UNDEFINED]
            max_distance = np.max(np.abs(valid_thresholds)) if len(valid_thresholds) > 0 else 1.0
        else:
            max_distance = None

        proba = []
        # For 'confidence' we need global stats
        if self.prob_type == 'confidence':
            feature_avgs = np.mean(X, axis=0)
            feature_stds = np.std(X, axis=0) + 1e-9
        else:
            feature_avgs = None
            feature_stds = None

        for x in X:
            simulation_results = []
            for _ in range(n_simulations):
                current_node = 0
                depth = 0
                while True:
                    feature_index = tree.feature[current_node]
                    if feature_index == _tree.TREE_UNDEFINED:
                        # Leaf node
                        simulation_results.append(tree.value[current_node][0])
                        break

                    threshold = tree.threshold[current_node]
                    if x[feature_index] <= threshold:
                        direction = 'left'
                        go_left = True
                    else:
                        direction = 'right'
                        go_left = False

                    # Compute probability p for flipping direction
                    if self.prob_type == 'fixed':
                        p = 0.2
                    elif self.prob_type == 'depth':
                        p = min(0.05 * depth, 0.2)
                    elif self.prob_type == 'certainty':
                        node_values = tree.value[current_node].flatten()
                        total = np.sum(node_values)
                        distribution = node_values / total if total > 0 else np.zeros_like(node_values)
                        max_certainty = np.max(distribution)
                        p = min(1 - max_certainty, 0.5)
                    elif self.prob_type == 'agreement':
                        node_values = tree.value[current_node].flatten()
                        majority_class_ratio = np.max(node_values) / np.sum(node_values) if np.sum(node_values) > 0 else 0
                        p = min(1 - majority_class_ratio, 0.5)
                    elif self.prob_type == 'distance':
                        distance = abs(x[feature_index] - threshold)
                        p = min(0.1 - min((distance / max_distance), 0.1), 0.5)
                    elif self.prob_type == 'confidence':
                        avg = feature_avgs[feature_index]
                        std = feature_stds[feature_index]
                        dist = abs(x[feature_index] - avg)
                        p = max(0.1 - (dist / std), 0)
                        p = min(p, 0.5)
                    elif self.prob_type == 'bayes':
                        p = 0  # Not implemented
                    else:
                        raise ValueError('Invalid prob_type')

                    more_negative_direction = self._get_more_negative_direction(current_node, tree)
                    if (direction == more_negative_direction) and \
                       (x[self.protected_attribute_index_in_tree] == self.unprivileged_group) and \
                       (feature_index == self.protected_attribute_index_in_tree):
                        p = min(p * ALPHA, 0.5)

                    rnd = np.random.rand()
                    if go_left:
                        if rnd > p:
                            current_node = tree.children_left[current_node]
                        else:
                            current_node = tree.children_right[current_node]
                    else:
                        if rnd > p:
                            current_node = tree.children_right[current_node]
                        else:
                            current_node = tree.children_left[current_node]
                    depth += 1

            mean_proba = np.mean(simulation_results, axis=0)
            proba.append(mean_proba)

        return np.array(proba)


#######################################################################
# Data Loading and Preprocessing
#######################################################################
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
        return (file_name, dataset_bunch)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

datasets = Parallel(n_jobs=-1)(
    delayed(load_dataset)(file_name) for file_name in dataset_list
)
datasets = [dataset for dataset in datasets if dataset is not None]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#######################################################################
# Metrics and Evaluation Functions
#######################################################################
def compute_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan, np.nan, np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size != 4:
        # Handle irregular confusion matrix sizes gracefully
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
    """
    Adjust predictions by applying a reject option in uncertain regions:
    If probability falls within (threshold ± uncertainty_range), 
    check if protected_attribute is unprivileged and prediction is unfavorable, 
    and flip if needed.
    """
    corrected_predictions = np.zeros(predictions.shape)
    decision_boundary_lower = threshold - uncertainty_range
    decision_boundary_upper = threshold + uncertainty_range
    for i, pred in enumerate(predictions):
        pred_class = int(pred > threshold)
        if decision_boundary_lower < pred < decision_boundary_upper:
            # In uncertainty region
            if protected_attribute[i] != privileged_group:
                # Unprivileged group
                if pred_class != favorable_class:
                    corrected_predictions[i] = 1 - pred_class
                else:
                    corrected_predictions[i] = pred_class
            else:
                corrected_predictions[i] = pred_class
        else:
            # Outside uncertainty region
            corrected_predictions[i] = pred_class
    return corrected_predictions

def evaluate_classifier(dataset_name, dataset, clf_name, clf, train_index, test_index):
    X = dataset.data.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
    y = dataset.target.values
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    protected_attribute_train = dataset.data[dataset.privileged_feature].values[train_index].astype(np.int32)
    protected_attribute_test = dataset.data[dataset.privileged_feature].values[test_index].astype(np.int32)

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

    # Train the classifier as per clf_name
    if clf_name == "DecisionTree_baseline":
        base_clf = DecisionTreeClassifier(random_state=111,min_samples_leaf=20)
        base_clf.fit(X_train, y_train)
        clf = base_clf
    elif clf_name == "MonteCarloDecisionTree":
        monte_clf = FairTTTSMonteCarloDecisionTreeClassifier(
            random_state=111,min_samples_leaf=20,
            prob_type=PROB_TYPE,
            protected_attribute_name=protected_attribute_index,
            privileged_group=dataset.privileged_group,
            unprivileged_group=dataset.unprivileged_group,
            favorable_class=dataset.favorable_class,
            unfavorable_class=dataset.unfavorable_class
        )
        monte_clf.fit(X_train, y_train)
        clf = monte_clf
    elif clf_name == "DecisionTree_RejectedOption":
        base_clf = DecisionTreeClassifier(random_state=111,min_samples_leaf=20)
        base_clf.fit(X_train, y_train)
        clf = base_clf
    elif clf_name == "ThresholdOptimizer":
        base_clf = DecisionTreeClassifier(random_state=111,min_samples_leaf=20)
        base_clf.fit(X_train, y_train)
        threshold_clf = ThresholdOptimizer(
            estimator=base_clf,
            constraints="equalized_odds",
            prefit=True,
            predict_method='predict_proba'
        )
        threshold_clf.fit(X_train, y_train, sensitive_features=protected_attribute_train)
        clf = threshold_clf
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

    start_time = time.time()

    # Prediction
    if clf_name == "ThresholdOptimizer":
        # Threshold optimizer outputs class labels directly
        preds = clf.predict(X_test, sensitive_features=protected_attribute_test)
    else:
        pred_probs = clf.predict_proba(X_test)
        # Identify favorable class index
        # Ensure classes_ is sorted or find the favorable class index
        if dataset.favorable_class in clf.classes_:
            favorable_class_index = np.where(clf.classes_ == dataset.favorable_class)[0][0]
        else:
            # If not found, default to the second class (assuming binary)
            favorable_class_index = 1 if len(clf.classes_) > 1 else 0

        if clf_name == "DecisionTree_RejectedOption":
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
            # Regular decision: predict highest probability class
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

#######################################################################
# Run Experiments
#######################################################################
classifiers = {
    "DecisionTree_baseline": DecisionTreeClassifier(),
    "DecisionTree_RejectedOption": DecisionTreeClassifier(),
    "MonteCarloDecisionTree": FairTTTSMonteCarloDecisionTreeClassifier(),
    "ThresholdOptimizer": ThresholdOptimizer(),
}

all_tasks = []
for dataset_name, dataset in datasets:
    print(f"Processing dataset: {dataset_name}")
    X, y = dataset.data.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32), dataset.target.fillna(0).values
    for clf_name, clf in classifiers.items():
        for train_index, test_index in skf.split(X, y):
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

summary_filename = f'FairTTTSDTMethod_{PROB_TYPE}_{ALPHA}_{S}_DecisionTree.csv'
summary_df.to_csv(summary_filename, index=False)
print(f"Results saved to {summary_filename}")
