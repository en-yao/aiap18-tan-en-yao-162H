### Name and email
Name: Tan En Yao

Email: t.enyao93@gmail.com

### Overview of folder structure
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

### Instruction for executing the pipeline and modifying parameters
To run the pipeline, double click on the the bash script (run.sh).

To modify parameters, go to *src/config.py*. In the file there are model training configurations, numerical and categorical pipelines classifiers and parameter grids that are used to configure the pipeline. Change the model training configurations by changing their values. Change the desired classifier and parameter grid by uncommenting.

### Description of flow of the pipeline
<br> **Step 1: Defining variables** <br>
- `classifier_models` - list of models that can be configured in the pipeline
- `config` - the config file that stores model training and evaluation parameters
- `transformers` - list of transformers that can be configured in the pipeline
- `x_cols` - list of feature columns
- `y_col` - target variable
- `num_cols` - list of numerical feature columns
- `cat_cols` - list of categorical feature columns
- `test_split` - train test split ratio
- `n_jobs` - number of parallel jobs to run in GridSearchCV
- `cv` - number of cross validations folds to be used in GridSearchCV
- `scoring` - scoring method to be used in GridSearvhCV

<br> **Step 2: Data preprocessing** <br>
`prepare_data()` calls the functions `fetch_data()` and `clean()` imported from `prep.py` for data processing. 
`fetch_data()` is called twice for weather.db and air_quality.db and the two resultant dataframes are merged to form the main dataframe. The main dataframe is processed using `clean()` before it is split into train and test datasets, X_train, X_test, y_train, y_test.

`fetch_data()` queries a database file and stores the result in a dataframe.

`clean()` processes the input data in the following steps:
1. Convert object columns to numeric
2. Convert 'date' column to datetime
3. Clean and standardize categorical columns with the goal of merging several categories into broader groups
4. Convert negative values to positive for the column 'Max Wind Speed'<br>
This is based on the assumption that negative values represent opposite wind direction
5. Remove duplicate rows
6. Impute missing values with the median. Missing values in a continuous numerical dataset can be imputed with either the mean or the median. The median is often preferred over the mean for imputing missing values in datasets because it is more robust to outliers and skewed distributions. If there are extreme values in the dataset, the mean can be pulled in their direction, leading to a skewed imputation value, however the median, as the middle value of a sorted dataset, is not influenced by outliers. It represents the central tendency without being affected by extreme values. Data visualization for numerical features in weather.db and air_quality.db shows numerous outliers are present, hence to avoid skewed imputations, missing values are imputed with the median.
7. Bin numeric features (to be elaborated in Step 3: Feature Engineering)
8. Feature engineering (to be elaborated in Step 3: Feature Engineering)
9. Convert target feature to numeric
10. Remove outliers that may be due to instrument error. Most of the features except "Wet Bulb Temperature (deg F)" have outliers that can be naturally occurring. Wet Bulb Temperature values typically range between about 0°F and 100°F. A value of -60 could be the result of a measurement or sensor error. The outlier values that fall outside the typical Wet Bulb Temperature range between about 0°F and 100°F should be excluded from the dataset

<br> **Step 3: Feature engineering** <br>
Feature engineering is part of `clean()` that is mentioned in Step 2. Some of the features are derived from binning and others are derived from domain knowledge. Binning might simplify non-linear relationships between the variable and the target. The following are binned features:

- Daily Rainfall - The impact of rainfall on solar panel efficiency might not be linear; light rainfall might have little to no effect, medium rainfall might clean the panels and improve efficiency, while heavy rainfall might drastically reduce efficiency

- Min Temperature & Maximum Temperature - Solar panels might operate optimally within a certain temperature range and become less efficient at higher temperatures due to overheating or at lower temperatures due to implied low sunlight availability.

- Relative Humidity - The impact of relative humidity on solar panel efficiency is likely to be non-linear; low humidity may have minimal impact, moderate humidity might slightly reduce efficiency due to moisture absorption, while high humidity levels can significantly reduce efficiency due to increased cloud cover, moisture on the panels, and potential for more frequent weather events like rain.

- PM25 (all regions) - The impact of PM25 on solar panel efficiency may be non-linear; low PM2.5 levels may have minimal impact, moderate PM2.5 levels might cause noticeable decrease in efficiency, while high PM2.5 levels may significantly reduce efficiency due to heavy smog or haze.

- PSI (all regions) - The impact of PSI on solar panel efficiency may be non-linear; low PSI levels migh have minimal impact, 
moderate PSI levels might cause slight decrease in efficiency due to some pollutants in the air. while high PSI levels can drop efficiency significantly as heavy pollution reduces the amount of sunlight reaching the panels.

The following are features derived from applying domain knowledge on the operating conditions of solar panels:

- Daily Temperature Range - A large diurnal temperature range can cause thermal stress on solar panels, affecting their efficiency over time.

- Wind Speed Range - Variability in wind speed could indicate periods of cooling which might enhance performance of the solar panels.

<br> **Step 4: Model prediction** <br>
Classification models are defined earlier in the `classifier_models` variable and includes the following models:

`RandomForestClassifier`
The Random Forest classifier natively supports multiclass classification. The aggregation of multiple decision trees reduces reduces variance and prevents overfitting. It is also able to model complex and non-linear decision boundaries, which is useful for multiclass problems where the boundaries between classes may not be linear or easily separable. 

`XGBClassifier`
The Xtreme Gradient Boosting classifier uses a boosting framework to correct the errors of previous models in an iterative process. It is able to adjust class weights to handle imbalanced class distributions. This is particularly useful in multiclass classification where some classes may have fewer instances than others. It also incorporates L1 and L2 regularization to help prevent overfitting and improve generalization.

`KNeighborsClassifier`
The K-Nearest Neighbor classifier classifies a new instance based on the majority class among its K nearest neighbors. It can handle multiclass classification by assigning the class that is most common among the K nearest neighbors. This means it can naturally work with multiple classes without needing modifications. The classifier is also able to handle complex, non-linear decision boundaries by considering the data structure. This is particularly useful in multiclass problems where the class boundaries may not be linearly separable.

`SVC`
The Support Vector classifier builds multiple binary classifiers using One-vs-Rest or One-vs-One strategies. Each classifier is trained to distinguish between different classes or pairs of classes, hence making it very effective for high-dimensional datasets. It is suitable for complex multiclass classification problems where features may be numerous and varied. It is also able to handle complex relationships and non-linear decision boundaries, which is valuable for multiclass problems with complex class distributions.

`GradientBoostingClassifier`
The Gradient Boosting classifier uses a boosting framework to correct the errors of previous models in an iterative process. It is able to model complex, non-linear relationships between features and target classes. This is especially useful in multiclass classification tasks where the decision boundaries between classes are not easily separable by linear models. The classifier also incorporates regularization techniques that helps prevent overfitting.

<br> **Step 5: Categorical Variable Encoding** <br>
`create_pipeline()` combines a classifier and pipelines for numerical and categorical features using `ColumnTransformer()`. Pipelines for categorical features include `categorical_pipeline_ordinal` and `categorical_pipeline_onehot` as configured in config.py. As the names suggest, `categorical_pipeline_ordinal` is configured with `OrdinalEncoder()` and `categorical_pipeline_onehot` is configured with `OneHotEncoder()`.

During data preprocessing in earlier steps, `LabelEncoder()` is used to encode the categorical data in the target variable.

`OneHotEncoder()` converts categorical variables into a series of binary columns. Each category is represented by a column with a 1 or 0 indicating the presence or absence of that category. For a categorical variable with n unique values, one-hot encoding creates n binary columns.

`OrdinalEncoder()` converts categorical variables into integers based on a defined order. Each unique category is assigned a unique integer value.

`LabelEncoder()` is used for target variable and can be used for all mentioned models as they expect single integer labels for classification

- For SVC, one-hot encoding helps to provide a clear separation between categories, enabling it to find the optimal hyperplane without misinterpreting categorical data.
- For KNN, one-hot encoding ensures that distances between data points with categorical features are caulculated accurately, leading to better classification performance.
- For XGBoost & Gradient Boosting, one-hot encoding enables accurate splits in decision trees based on binary features, which improves their ability to handle categorical data and capture complex interactions.
- For Random Forest Classifier, ordinal encoding may be preferred to preserve meaningful ordinal relationships. For example, a feature called Wind Speed may be categorized as Low, Medium, and High. Ordinal encoding preserves this order and allows the RandomForest model to recognize the increasing impact of wind speed on solar panel efficiency. 

**Step 6: Evaluation Metric Selection** <br>
`train_and_evaluate()` takes a parameter grid of a model and uses `GridSearchCV()` to find the best parameters. The best performing model is used for evaluation using precision, recall, f1 score and ROC AUC.

Precision - the ratio of correctly predicted positive observations to the total predicted positives. It measures how many of the items identified as positive are actually positive.

Recall - the ratio of correctly predicted positive observations to the all observations in the actual class. It measures how many of the actual positives were identified.

F1 Score - The F1 Score is the harmonic mean of precision and recall. It provides a single metric that balances the trade-off between precision and recall, especially useful when you need to balance both metrics. It is particularly useful in cases of imbalanced datasets where you want to ensure that both false positives and false negatives are minimized.

ROC AUC - AUC-ROC measures the performance of a classification model at various threshold settings. It is typically used for binary classifications. For multiclass problems where the target variable has more than two categories such as "Low", "Medium", "High", `OneVsRestClassifier()` is used for evaluating the ROC AUC curve for each category. Additionally, `multi_class='ovr'` will be passed to `roc_auc_score()`. The ROC AUC curve is saved in the file`chart.png` in the *src* folder

### Overview of key findings from the EDA 
- Data cleaning is necessary to remove duplicate rows, impute missing values and merge categories into broader groups for some categorical features
- There is slight imbalance in the target variable.
- There is a significant number of outliers across several numerical features, but only the outliers from "Wet Bulb Temperature (deg F)" are removed as they are outside the normal instrument measuring range between about 0°F and 100°F.

### Describe how features in the dataset are processed

| Process                    | Description                                                                                       |
|----------------------------|---------------------------------------------------------------------------------------------------|
| Convert column dtype       | Convert object columns to numeric to enable mathematical calculations                             |
| Convert column to datetime | Convert date to datetime to enable timeseries analysis                                            |
| Clean categorical columns  | Merge categories into broader groups to help simplify analysis                                    |
| Convert negative values    | Convert negative values to positive for the column 'Max Wind Speed'                               |
| Removing duplicate rows    | Find and remove duplicated values in features                                                     |
| Impute Missing Values      | Find and replace missing values in features with median to preserve data integrity                |
| Bin numeric features       | Bin numerical features to simplify non-linear relationships between the variable and the target   |
| Feature engineering        | Create more features that might provide additional predictive power                               |
| Convert target feature     | Convert target feature to numeric for model compatibility                                         |
| Handle outliers            | Identify and handle outliers in the dataset. (Remove if necessary)                                |

### Explanation of your choice of models 
`RandomForestClassifier`
The Random Forest classifier natively supports multiclass classification. The aggregation of multiple decision trees reduces reduces variance and prevents overfitting. It is also able to model complex and non-linear decision boundaries, which is useful for multiclass problems where the boundaries between classes may not be linear or easily separable. 

`XGBClassifier`
The Xtreme Gradient Boosting classifier uses a boosting framework to correct the errors of previous models in an iterative process. It is able to adjust class weights to handle imbalanced class distributions. This is particularly useful in multiclass classification where some classes may have fewer instances than others. It also incorporates L1 and L2 regularization to help prevent overfitting and improve generalization.

`KNeighborsClassifier`
The K-Nearest Neighbor classifier classifies a new instance based on the majority class among its K nearest neighbors. It can handle multiclass classification by assigning the class that is most common among the K nearest neighbors. This means it can naturally work with multiple classes without needing modifications. The classifier is also able to handle complex, non-linear decision boundaries by considering the data structure. This is particularly useful in multiclass problems where the class boundaries may not be linearly separable.

`SVC`
The Support Vector classifier builds multiple binary classifiers using One-vs-Rest or One-vs-One strategies. Each classifier is trained to distinguish between different classes or pairs of classes, hence making it very effective for high-dimensional datasets. It is suitable for complex multiclass classification problems where features may be numerous and varied. It is also able to handle complex relationships and non-linear decision boundaries, which is valuable for multiclass problems with complex class distributions.

`GradientBoostingClassifier`
The Gradient Boosting classifier uses a boosting framework to correct the errors of previous models in an iterative process. It is able to model complex, non-linear relationships between features and target classes. This is especially useful in multiclass classification tasks where the decision boundaries between classes are not easily separable by linear models. The classifier also incorporates regularization techniques that helps prevent overfitting.

### Evaluation of the models developed
- In terms of Accuracy, XGBoost has the highest accuracy at 0.847, indicating that 84.74% of the predictions were correct. KNN has the lowest accuracy at 0.729.
- In terms of Precision, XGBoost has the highest precision score of 0.850, meaning it has the highest proportion of true positive predictions among all positive predictions. SVC has the lowest precision score of 0.748, which means it has more false positives compared to XGBoost.
- In terms of Recall, XGBoost has the highest recall score of 0.793 and Gradient Boosting has a recall score slightly lower at 0.795. This shows that both models correctly identified a large proportion of actual positive cases. KNN has the lowest recall score of 0.636, which means it missed a significant number of true positive cases.
- In terms of F1 score, XGBoost has the highest F1 score at 0.815, showing good balance between precision and recall. KNN has the lowest F1 score at 0.665, which shows poor overall balance of precision and recall.
- In terms of ROC AUC score, Gradient Boosting has the highest ROC AUC score of 0.844, indicating excellent performance in distinguishing between positive and negative classes. KNN has the lowest ROC AUC score of 0.794, it is less effective at distinguishing between classes compared to the other models.
- Overall, XGBoost and Gradient Boosting are the two best performing models across most metrics, XGBoost has slightly better precision and F1 scores and Gradient Boosting has a better ROC AUC score.

### Other considerations for deploying the models developed
