import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from typing import Tuple, List

import config
from prep import fetch_data, clean


class ClassifierPipeline:
    def __init__(self, config) -> None:
        self.classifier_models = {
            "RandomForestClassifier": RandomForestClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "SVC": SVC(),
        }
        self.config = config

        self.transformers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "OneHotEncoder": OneHotEncoder(sparse_output=False),
        }

        self.x_cols = [
            "ID",
            "Call Duration",
            "Call Frequency",
            "Financial Loss",
            "Flagged by Carrier",
            "Is International",
            "Previous Contact Count",
            "Country Prefix",
            "Call Type",
        ]
        self.y_col = "Scam Call"

        self.num_cols = [
            "Call Duration",
            "Call Frequency",
            "Financial Loss",
            "Previous Contact Count",
        ]
        self.cat_cols = [
            "Flagged by Carrier",
            "Is International",
            "Country Prefix",
            "Call Type",
        ]

        self.test_split = getattr(self.config, "test_split", 0.2)
        self.n_jobs = getattr(self.config, "n_jobs", -1)
        self.cv = getattr(self.config, "cv", 5)
        self.scoring = getattr(self.config, "scoring", "accuracy")

    def _config_steps(self, config, pipe_var) -> List:
        if len(config) == 0:
            raise Exception(f"No steps in config file for {pipe_var}.")
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
            df = fetch_data(self.config.filepath_to_db, "SELECT * FROM calls")
            print("Data fetched successfully")
            df = clean(df)
            print("Data cleaned successfully")

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
        cat_pipeline = Pipeline(
            steps=self._config_steps(
                self.config.categorical_pipeline, "categorical_pipeline"
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
            steps=[("col_transformer", col_transformer), ("classifier", classifier)]
        )
        return final_pipeline

    def train_and_evaluate(
        self, X_train, X_test, y_train, y_test, final_pipeline
    ) -> None:
        param_grid = self.config.parameters
        grid_search = GridSearchCV(
            estimator=final_pipeline,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        grid_search.fit(X_train, y_train)
        print("Model trained successfully")
        best_classifier = grid_search.best_estimator_
        y_pred = best_classifier.predict(X_test)
        print(grid_search.best_score_)
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)

    def run_pipeline(self) -> None:
        X_train, X_test, y_train, y_test = self.prepare_data()
        final_pipeline = self.create_pipeline()
        self.train_and_evaluate(X_train, X_test, y_train, y_test, final_pipeline)


if __name__ == "__main__":
    pipeline = ClassifierPipeline(config)
    pipeline.run_pipeline()
