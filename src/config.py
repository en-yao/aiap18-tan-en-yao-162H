filepath_to_db = "../data/calls.db"
test_split = 0.2
n_jobs = -1
cv = 5
scoring = "accuracy"

numerical_pipeline = {
    'scale': 'StandardScaler'
}

categorical_pipeline = {
    'encode': 'OneHotEncoder'
}

classifier = 'RandomForestClassifier'

parameters = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}
'''
classifier = 'DecisionTreeClassifier'

parameters = {
    'classifier__criterion': ["gini", "entropy"],
    'classifier__max_depth': [None, 5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

classifier = 'SVC'

parameters = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ["linear", "rbf"],
    'classifier__gamma': [0.1, 0.01]
}'''