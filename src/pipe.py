import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    label_binarize
)
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score, roc_curve, auc
) 
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from typing import Tuple, List

import config
from prep import fetch_data, clean

# Disable chained assignment warning (false positive)
pd.options.mode.chained_assignment = None

class ClassifierPipeline:
    def __init__(self, config) -> None:
        self.classifier_models = {
            'RandomForestClassifier': RandomForestClassifier(),
            'XGBClassifier': XGBClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'SVC': SVC(),
            'GradientBoostingClassifier': GradientBoostingClassifier() 
        }
        self.config = config

        self.transformers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'OneHotEncoder': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'LabelEncoder': LabelEncoder(),
            'OrdinalEncoder': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        }

        self.x_cols = [
            'Daily Rainfall Total (mm)',
            'Highest 30 Min Rainfall (mm)',
            'Highest 60 Min Rainfall (mm)',
            'Highest 120 Min Rainfall (mm)',
            'Min Temperature (deg C)',
            'Maximum Temperature (deg C)',
            'Min Wind Speed (km/h)',
            'Max Wind Speed (km/h)',
            'Sunshine Duration (hrs)',
            'Cloud Cover (%)',
            'Wet Bulb Temperature (deg F)',
            'Relative Humidity (%)',
            'Air Pressure (hPa)',
            'pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central',
            'psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central',
            'temp_range',
            'wind_speed_range',
            'Dew Point Category',
            'Wind Direction',
            'Daily Rainfall',
            'Min Temperature',
            'Maximum Temperature',
            'Relative Humidity',
            'PM25 North', 'PM25 South', 'PM25 East', 'PM25 West', 'PM25 Central',
            'PSI North', 'PSI South', 'PSI East', 'PSI West', 'PSI Central',
            'Temperature Range',
            'Wind Speed Range'
        ]
        self.y_col = 'Daily Solar Panel Efficiency'

        self.num_cols = [
            'Daily Rainfall Total (mm)',
            'Highest 30 Min Rainfall (mm)',
            'Highest 60 Min Rainfall (mm)',
            'Highest 120 Min Rainfall (mm)',
            'Min Temperature (deg C)',
            'Maximum Temperature (deg C)',
            'Min Wind Speed (km/h)',
            'Max Wind Speed (km/h)',
            'Sunshine Duration (hrs)',
            'Cloud Cover (%)',
            'Wet Bulb Temperature (deg F)',
            'Relative Humidity (%)',
            'Air Pressure (hPa)',
            'pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central',
            'psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central',
            'temp_range',
            'wind_speed_range'
        ]
        self.cat_cols = [
            'Dew Point Category',
            'Wind Direction',
            'Daily Rainfall',
            'Min Temperature',
            'Maximum Temperature',
            'Relative Humidity',
            'PM25 North', 'PM25 South', 'PM25 East', 'PM25 West', 'PM25 Central',
            'PSI North', 'PSI South', 'PSI East', 'PSI West', 'PSI Central',
            'Temperature Range',
            'Wind Speed Range'
        ]

        self.test_split = getattr(self.config, 'test_split', 0.2)
        self.n_jobs = getattr(self.config, 'n_jobs', -1)
        self.cv = getattr(self.config, 'cv', 5)
        self.scoring = getattr(self.config, 'scoring', 'accuracy')

    def _config_steps(self, config, pipe_var) -> List:
        if len(config) == 0:
            raise Exception(f'No steps in config file for {pipe_var}.')
        steps = []
        for key, value in config.items():
            if value not in self.transformers:
                raise Exception(f"Transformer {value} not supported.")
            steps.append((key, self.transformers[value]))
        return steps

    def _get_classifier_model(self, classifer_model):
        if classifer_model not in self.classifier_models:
            raise Exception(f"Classifier model {classifer_model} not supported.")
        return self.classifier_models[classifer_model]

    def prepare_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            df1 = fetch_data(self.config.filepath_to_db1, 'SELECT * FROM weather')
            df2 = fetch_data(self.config.filepath_to_db2, 'SELECT * FROM air_quality')
            print('Data fetched successfully')
            df = pd.merge(df1, df2, on=['data_ref', 'date'])
            df = clean(df)
            print('Data cleaned successfully')

            X = df[self.x_cols]
            y = df[self.y_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_split
            )

            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise Exception(f"Error in preparing data: {e}")

    def create_pipeline(self) -> Pipeline:
        
        num_pipeline = Pipeline(
            steps=self._config_steps(
                self.config.numerical_pipeline, "numerical_pipeline"
            )
        )

        if self.config.classifier == "RandomForestClassifier":
            cat_pipeline = Pipeline(
                steps=self._config_steps(
                    self.config.categorical_pipeline_ordinal, "categorical_pipeline"
                )
            )
        else:
            cat_pipeline = Pipeline(
                steps=self._config_steps(
                    self.config.categorical_pipeline_onehot, "categorical_pipeline"
                )
            )

        col_transformer = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, self.num_cols),
                ("cat_pipeline", cat_pipeline, self.cat_cols),
            ],
            remainder="drop",
            n_jobs=self.n_jobs,
        )
        classifier = self._get_classifier_model(self.config.classifier)
        final_pipeline = Pipeline(
            steps=[("col_transformer", col_transformer), ("classifier",  OneVsRestClassifier(classifier))]
        )

        return final_pipeline

    def train_and_evaluate(
        self, X_train, X_test, y_train, y_test, final_pipeline
    ) -> None: 
        grid_search = GridSearchCV(
            estimator=final_pipeline,
            param_grid=self.config.parameters,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        grid_search.fit(X_train, y_train)
        print("Model trained successfully")
        best_classifier = grid_search.best_estimator_
        y_pred = best_classifier.predict(X_test)
        y_pred_proba = best_classifier.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        y_test_bin = label_binarize(y_test, classes=best_classifier.classes_)
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
        print(f"ROC AUC Score: {roc_auc}")

        plt.figure()
        for i in range(len(best_classifier.classes_)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'Class {best_classifier.classes_[i]} (area = {auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig(self.config.filepath_to_roc_auc_curve)
        print (f"ROC AUC curve saved sucessfully to {self.config.filepath_to_roc_auc_curve}")

    def run_pipeline(self) -> None:
        X_train, X_test, y_train, y_test = self.prepare_data()
        final_pipeline = self.create_pipeline()
        self.train_and_evaluate(X_train, X_test, y_train, y_test, final_pipeline)


if __name__ == "__main__":
    pipeline = ClassifierPipeline(config)
    pipeline.run_pipeline()
