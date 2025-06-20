import os
import time
import numpy as np
import pandas as pd
import joblib
import contextlib
import warnings
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.exceptions import ConvergenceWarning

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# --- Suppress specific warnings from scikit-learn ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to display a tqdm progress bar for joblib parallel jobs.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class ModelTrainer:
    """
    A class to train, evaluate, and compare multiple machine learning models.
    Includes logic to skip retraining if models are already saved.
    """

    def __init__(self, train_data_path, test_data_path, categories_path, output_dir, n_jobs=-1):
        """
        Initializes the ModelTrainer.
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.categories_path = categories_path
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, 'models')
        self.cm_dir = os.path.join(output_dir, 'confusion_matrices')
        self.n_jobs = n_jobs

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)

        # Initialize placeholders
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.class_labels = None
        self.category_mapping = None
        self.pipelines = {}
        self.param_grids = {}
        self.best_models = {}
        self.cv_scores = {}
        self.test_results = {}

    def _load_and_prepare_data(self):
        """
        Loads feature data, prepares it for scikit-learn, and loads category descriptions.
        """
        print("1. Loading and preparing data...")
        train_df = pd.read_pickle(self.train_data_path)
        test_df = pd.read_pickle(self.test_data_path)
        self.X_train = np.vstack(train_df['fv'].values)
        self.X_test = np.vstack(test_df['fv'].values)
        self.y_train = train_df['occupation_code'].values - 1
        self.y_test = test_df['occupation_code'].values - 1
        self.class_labels = [str(c) for c in sorted(train_df['occupation_code'].unique())]

        try:
            categories_df = pd.read_csv(self.categories_path, sep=':', header=None, names=['code', 'description'])
            self.category_mapping = pd.Series(categories_df.description.values, index=categories_df.code).to_dict()
            print("Successfully loaded category descriptions for plot legends.")
        except FileNotFoundError:
            self.category_mapping = None
        print("-" * 30)

    def _define_models_and_pipelines(self):
        """
        Defines models, pipelines, and extensive hyperparameter grids for tuning.
        """
        print("2. Defining model pipelines and hyperparameter grids...")

        # --- Logistic Regression ---
        self.pipelines['Logistic Regression'] = Pipeline(
            [('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42))])
        self.param_grids['Logistic Regression'] = [
            {'scaler': [StandardScaler(), MinMaxScaler(), 'passthrough'], 'classifier__solver': ['saga'],
             'classifier__penalty': ['elasticnet'], 'classifier__l1_ratio': np.linspace(0.1, 0.9, 3),
             'classifier__max_iter': [1000, 2500]},
            {'scaler': [StandardScaler(), MinMaxScaler(), 'passthrough'], 'classifier__solver': ['saga'],
             'classifier__penalty': ['l1', 'l2'], 'classifier__max_iter': [1000, 2500]},
            {'scaler': [StandardScaler(), MinMaxScaler(), 'passthrough'], 'classifier__solver': ['lbfgs'],
             'classifier__penalty': ['l2'], 'classifier__max_iter': [1000, 2500]}
        ]

        # --- Random Forest ---
        self.pipelines['Random Forest'] = Pipeline(
            [('scaler', 'passthrough'), ('classifier', RandomForestClassifier(random_state=42, n_jobs=self.n_jobs))])
        self.param_grids['Random Forest'] = {'classifier__n_estimators': [100, 300, 500, 1000],
                                             'classifier__max_depth': [10, 20, 40, None],
                                             'classifier__max_features': ['sqrt', 'log2'],
                                             'classifier__min_samples_leaf': [1, 2, 4],
                                             'classifier__class_weight': [None, 'balanced_subsample']}

        # --- XGBoost ---
        self.pipelines['XGBoost'] = Pipeline([('scaler', 'passthrough'), ('classifier',
                                                                          XGBClassifier(objective='multi:softprob',
                                                                                        random_state=42,
                                                                                        use_label_encoder=False,
                                                                                        eval_metric='mlogloss'))])
        self.param_grids['XGBoost'] = {'classifier__n_estimators': [100, 200, 300, 400],
                                       'classifier__learning_rate': [0.01, 0.05, 0.1],
                                       'classifier__max_depth': [3, 5, 7], 'classifier__reg_alpha': [0, 0.1, 1],
                                       'classifier__reg_lambda': [0.1, 1, 10]}

        # --- SVM ---
        self.pipelines['SVM'] = Pipeline(
            [('scaler', StandardScaler()), ('classifier', SVC(random_state=42, probability=True))])
        self.param_grids['SVM'] = [
            {'scaler': [StandardScaler(), MinMaxScaler()], 'classifier__kernel': ['rbf'],
             'classifier__C': np.logspace(-2, 2, 5), 'classifier__gamma': np.logspace(-2, 2, 5),
             'classifier__class_weight': [None, 'balanced']},
            {'scaler': [StandardScaler(), MinMaxScaler()], 'classifier__kernel': ['linear'],
             'classifier__C': np.logspace(-2, 2, 5), 'classifier__class_weight': [None, 'balanced']},
            {'scaler': [StandardScaler(), MinMaxScaler()], 'classifier__kernel': ['poly'],
             'classifier__C': np.logspace(-2, 2, 5), 'classifier__degree': [2, 3, 4],
             'classifier__class_weight': [None, 'balanced']}
        ]

        # --- MLP ---
        self.pipelines['MLP (Neural Network)'] = Pipeline([('scaler', StandardScaler()), (
        'classifier', MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=10, max_iter=1000))])
        self.param_grids['MLP (Neural Network)'] = [
            {'scaler': [StandardScaler(), MinMaxScaler()], 'classifier__solver': ['adam'],
             'classifier__activation': ['relu', 'tanh'],
             'classifier__hidden_layer_sizes': [(64,), (128,), (256,), (64, 128), (128, 64), (128, 256), (256, 128),
                                                (64, 128, 64), (128, 256, 128)],
             'classifier__alpha': [0.0001, 0.001, 0.01], 'classifier__learning_rate_init': [0.001, 0.01, 0.1]},
            {'scaler': [StandardScaler(), MinMaxScaler()], 'classifier__solver': ['sgd'],
             'classifier__activation': ['relu', 'tanh'],
             'classifier__hidden_layer_sizes': [(64,), (128,), (256,), (64, 128), (128, 64), (128, 256), (256, 128),
                                                (64, 128, 64), (128, 256, 128)],
             'classifier__alpha': [0.0001, 0.001, 0.01], 'classifier__learning_rate': ['constant', 'adaptive'],
             'classifier__learning_rate_init': [0.001, 0.01, 0.1], 'classifier__momentum': [0.9, 0.95, 0.99]}
        ]
        print("Definitions complete.")
        print("-" * 30)

    def _load_saved_models(self):
        """
        Attempts to load pre-trained models from the output directory.
        Returns True if all models were successfully loaded, False otherwise.
        """
        print("Attempting to load saved models...")
        all_models_loaded = True
        for name in self.pipelines.keys():
            model_filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.joblib"
            model_path = os.path.join(self.models_dir, model_filename)

            if os.path.exists(model_path):
                try:
                    self.best_models[name] = joblib.load(model_path)
                    print(f"- Found and loaded: {name}")
                except Exception as e:
                    print(f"Error loading {model_path}: {e}. Will retrain.")
                    all_models_loaded = False
                    self.best_models = {}
                    break
            else:
                print(f"- Model not found: {name}. Full training pipeline will run.")
                all_models_loaded = False
                self.best_models = {}
                break

        if all_models_loaded:
            print("All models successfully loaded from disk.")
        return all_models_loaded

    def train_models(self):
        """
        Runs GridSearchCV for each defined model pipeline.
        """
        print("\n3. Starting hyperparameter tuning for all models...")
        for name in self.pipelines.keys():
            print(f"\n========== Tuning Model: {name} ==========")
            start_time = time.time()
            pipeline = self.pipelines[name]
            param_grid = self.param_grids[name]
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='f1_macro',
                                       n_jobs=self.n_jobs, verbose=0)

            if isinstance(param_grid, list):
                n_combinations = sum(
                    np.prod([len(v) for v in d.values() if isinstance(v, (list, np.ndarray))]) for d in param_grid)
            else:
                n_combinations = np.prod([len(v) for v in param_grid.values()])
            total_fits = n_combinations * grid_search.cv

            print(f"Searching {total_fits} combinations...")
            with tqdm_joblib(tqdm(desc=f"{name} Grid Search", total=total_fits)):
                grid_search.fit(self.X_train, self.y_train)

            end_time = time.time()
            self.best_models[name] = grid_search.best_estimator_
            self.cv_scores[name] = grid_search.best_score_
            print(f"\n--- {name} CV Results ---")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Best CV F1-Macro Score: {grid_search.best_score_:.4f}")
            print(f"Total Search Time: {end_time - start_time:.2f} seconds")
            print("=" * 60)
        print("\nAll models have been tuned.")
        print("-" * 30)

    def evaluate_and_save_models(self):
        """
        Evaluates models on the test set, saves models, plots, and a summary CSV.
        """
        print("\n4. Evaluating best models on the test set and saving all artifacts...")
        results_for_csv = []
        for name, model in self.best_models.items():
            print(f"\n--- Evaluating: {name} ---")
            y_pred = model.predict(self.X_test)
            test_accuracy = round(accuracy_score(self.y_test, y_pred), 4)
            test_f1_macro = round(f1_score(self.y_test, y_pred, average='macro', zero_division=0), 4)
            test_f1_weighted = round(f1_score(self.y_test, y_pred, average='weighted', zero_division=0), 4)
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=self.class_labels, zero_division=0))

            cm_path = os.path.join(self.cm_dir,
                                   f"cm_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png")
            try:
                # Use a layout that is more square to better fit the matrix
                fig, ax = plt.subplots(figsize=(10, 8))
                fig.suptitle(f"Confusion Matrix: {name}", fontsize=16)

                disp = ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred, display_labels=self.class_labels, ax=ax)

                # Use tight_layout to automatically adjust plot params
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make space for suptitle

                plt.savefig(cm_path)
                plt.close(fig)
                print(f"Confusion Matrix saved to: {cm_path}")
            except Exception as e:
                print(f"Could not save confusion matrix for {name}: {e}")

            # Only save the model if it was newly trained
            if self.cv_scores:
                model_path = os.path.join(self.models_dir,
                                          f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.joblib")
                joblib.dump(model, model_path)
                print(f"Model saved to: {model_path}")

            results_for_csv.append({
                'Model Name': name,
                'Test Accuracy': test_accuracy,
                'Test F1-Macro': test_f1_macro,
                'Test F1-Weighted': test_f1_weighted,
            })
            print("-" * 55)

        summary_df = pd.DataFrame(results_for_csv).sort_values(by='Test Accuracy', ascending=False)
        print("\n\n========== FINAL MODEL PERFORMANCE SUMMARY ==========")
        print(summary_df)
        csv_path = os.path.join(self.output_dir, 'performance_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"\nFinal summary saved to: {csv_path}")
        print("-" * 30)

    def run(self, force_retrain=False):
        """
        Executes the entire training and evaluation pipeline.

        Args:
            force_retrain (bool): If True, forces retraining even if models exist.
                                  If False, skips training if models are found.
        """
        self._load_and_prepare_data()
        self._define_models_and_pipelines()

        if force_retrain or not self._load_saved_models():
            self.train_models()
        else:
            print("\nSkipping training as all models were loaded from disk.")

        self.evaluate_and_save_models()
        print("\nModel pipeline has completed successfully.")


if __name__ == '__main__':
    # --- Configuration ---
    BASE_DATA_DIR = '../data'
    RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, 'raw')
    EXTRACTED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'extracted')
    OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'output')

    train_file_path = os.path.join(EXTRACTED_DATA_DIR, 'train.pkl')
    test_file_path = os.path.join(EXTRACTED_DATA_DIR, 'test.pkl')
    categories_file_path = os.path.join(RAW_DATA_DIR, 'major_categories')

    # Set this to True to force the grid search and retraining of all models.
    # Set to False to use saved models if they exist.
    FORCE_RETRAIN_MODELS = False

    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        print(f"Error: Training or test data not found.")
        print(f"Please run the preprocessing and feature extraction scripts first.")
    else:
        trainer = ModelTrainer(
            train_data_path=train_file_path,
            test_data_path=test_file_path,
            categories_path=categories_file_path,
            output_dir=OUTPUT_DIR,
            n_jobs=16
        )
        trainer.run(force_retrain=FORCE_RETRAIN_MODELS)
