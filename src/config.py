# File paths to databases
filepath_to_db1 = "../data/weather.db"
filepath_to_db2 = "../data/air_quality.db"

# Configuration for model training and evaluation
test_split = 0.2
n_jobs = -1
cv = 5
scoring = "accuracy"

# File path to ROC AUC curve image
filepath_to_roc_auc_curve = "chart.png"

# Pipelines for preprocessing
numerical_pipeline = {
    'scale': 'StandardScaler'
}

categorical_pipeline_onehot = {
    'encode': 'OneHotEncoder'
}

categorical_pipeline_ordinal = {
    'encode': 'OrdinalEncoder'
}

# Uncomment the desired classifier and parameter grid

# RandomForestClassifier parameters
classifier = 'RandomForestClassifier'
parameters = {
    'classifier__estimator__n_estimators': [50, 100, 150],
    'classifier__estimator__max_depth': [None, 10, 20, 30],
    'classifier__estimator__min_samples_split': [2, 5, 10],
    'classifier__estimator__min_samples_leaf': [1, 2, 4]
}

# XGBClassifier parameters
# classifier = 'XGBClassifier'
# parameters = {
#     'classifier__estimator__n_estimators': [50, 100, 150],
#     'classifier__estimator__max_depth': [3, 6, 9],
#     'classifier__estimator__learning_rate': [0.01, 0.1, 0.2],
#     'classifier__estimator__subsample': [0.6, 0.8, 1.0],
#     'classifier__estimator__colsample_bytree': [0.6, 0.8, 1.0]
# }

# KNeighborsClassifier parameters
# classifier = 'KNeighborsClassifier'
# parameters = {
#     'classifier__estimator__n_neighbors': [3, 5, 7, 9],
#     'classifier__estimator__weights': ['uniform', 'distance'],
#     'classifier__estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'classifier__estimator__p': [1, 2]
# }

# SVC parameters
# classifier = 'SVC'
# parameters = {
#     'classifier__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'classifier__estimator__C': [0.1, 1, 10, 100],
#     'classifier__estimator__gamma': ['scale', 'auto'],
#     'classifier__estimator__degree': [3, 4, 5],
#     'classifier__estimator__probability': [True]
# }

# GradientBoostingClassifier parameters
# classifier = 'GradientBoostingClassifier'
# parameters = {
#     'classifier__estimator__n_estimators': [50, 100, 150],
#     'classifier__estimator__learning_rate': [0.01, 0.1, 0.2],
#     'classifier__estimator__max_depth': [3, 5, 7],
#     'classifier__estimator__min_samples_split': [2, 5, 10],
#     'classifier__estimator__min_samples_leaf': [1, 2, 4],
#     'classifier__estimator__subsample': [0.7, 0.85, 1.0]
# }