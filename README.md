#### Name and email
Name: Tan En Yao

Email: t.enyao93@gmail.com

#### Overview of folder structure
```
AIAP
│
├── .github
├── src/                  (Source code directory)
│   ├── pipe.py           (Main Python script)
│   ├── prep.py           (Helper functions)      
│   ├── config.py         (Configuration settings)
│   ├── query.py          (Dataset query)
├── README.md             (Project documentation)
├── eda.ipynb             (Jupyter notebook)
├── requirements.txt      (Python dependencies)
├── run.sh                (Shell script)
```

#### Instruction for executing the pipeline and modifying parameters
Double click on the the bash script (run.sh) to run it.

Go to *config.py* to modify parameters. Pipieline processes can be modified under `numerical_pipeline` and `categorical_pipeline`. Algorithms and parameters can be modified under `classifier` and `parameters` respectively.

#### Description of flow of the pipeline
Data is loaded and processed using steps from the exploratory data analysis and includes removing duplicates, imputing missing values and category consolidation.

The processed data is then split into train and test data.

The overall pipeline consists of a numerical pipeline, a categorical pipeline and a classifier. Optimization is done with grid search using parameters from *config.py*.

Modifications to the classifiers and their parameters, as well as the processes in numerical and categorical pipelines can be made in *config.py*.

#### Overview of key findings from the EDA 
Duplicate rows were found and removed. Unexpected negative values were found and removed. Categories that represent the same concept but are expressed differently were found and merged together.

A mix of categorical and numerical features in the data requires that the pipeline consists of pipelines that can transform each.

#### Describe how features in the dataset are processed

| Process                    | Description                                                                                       |
|----------------------------|---------------------------------------------------------------------------------------------------|
| Removing Duplicates        | Find and remove duplicated values in features.                                                    |
| Handling Missing Values    | Find and replace missing values in features with either mean, median or mode.                     |
| Removing Unexpected Values | Remove values in features that are not logical, e.g. negative values for duration of an event     |
| Category consolidation     | Find and merge categories that represent the same concept but are expressed differently           |
| Encoding Categorical       | Encode categorical features into numerical representations.                                       |
| Scaling Numerical          | Scale numerical features to a common range.                                                       |


#### Explanation of your choice of models 
The explanatory data analysis shows that the classes for *Scam* and *Not Scam* are not linearly separable. The following models can be used to solve this type of non-linear classification.

*Support Vector Machines*
This classifier typically uses linear support vectors to separate classes in the best possible way. However, for cases where classes are inseparable by linear vectors, the classifier uses kernel trick to transform the input space to a higher dimensional space such that the classes can easily be separated by linear vectors.

*Decision Tree Classifier*
This classifier uses the features in the dataset to form a complex decision boundary in the form of a flowchart. This approach is useful in cases where the decision boundary is not as simple as a straight line.

*Random Forest*
This classifier is much like the Decision Tree Classifier where features in the dataset form complex decision boundaries. However it uses randomness to generate several other trees and aggregate the results from across the different trees to make a final classification. This step helps to increase the accuracy of predicting out of sample data.

#### Evaluation of the models developed
Accuracy, precision and recall were used to evaluate model performance. Accuracy was around 0.8-0.85 for all the models.

In the context of scam calls, recall takes precedence over precision as the aim of the model is to accurately identify as many scam calls as possible. Failure to identify a scam call may potentially result in undesirable consequences such as financial loss.

With this in mind, the Support Vector Machine is the preferred model as it has the highest recall score of 0.7 for the Scam class out of the 3 models. While Random Forest had the lowest recall score of 0.63.

#### Other considerations for deploying the models developed
No other considerations
