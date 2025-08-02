# Hands-on Machine Learning with Scikit-learn, Keras, and TensorFlow

My study notes and worked exercises from
[Hands-on Machine Learning with Scikit-learn, Keras, and TensorFlow (3rd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/).

## Data Sets

* [Geron - California Housing](https://github.com/ageron/data/tree/main/housing) (Chpater 2)
* [OpenML - mnist_784](https://www.openml.org/search?type=data&status=active&id=554) (Chapter 3)
* [Geron - Titanic](https://github.com/ageron/data/tree/main/titanic) (Chpater 3)
* [Spamassassin - Spam/Ham](https://spamassassin.apache.org/old/publiccorpus/) (Chapter 3)

## Glossary

Search the code-base for these terms to find usages and examples.

### joblib

* [`joblib.dump`](https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html) - Persist an arbitrary Python object into one file.
* [`joblib.load`](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html) - Reconstruct a Python object from a file persisted with `joblib.dump`.

### pandas

* [`pandas.cut`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html) - Bin values into discrete intervals.
* [`pandas.DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html) - Foundational data holding object for pandas.
* [`pandas.get_dummies`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) - Convert categorical variable into dummy/indicator variables.
* [`pandas.plotting.scatter_matrix`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html) - Draw a matrix of scatter plots.
* [`pandas.read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) - Read a comma-separated values (csv) file into DataFrame.
* [`pandas.Series`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) - One-dimensional ndarray with axis labels (including time series).

### scikit-learn

* [`sklearn.base.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) - Base class for all estimators in scikit-learn.
* [`sklearn.base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html) - Construct a new unfitted estimator with the same parameters.
* [`sklearn.base.TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html) - Mixin class for all transformers in scikit-learn.
* [`sklearn.compose.ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) - Applies transformers to columns of an array or pandas DataFrame.
* [`sklearn.compose.make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html) - Create a callable to select columns to be used with `ColumnTransformer`.
* [`sklearn.compose.TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) - Meta-estimator to regress on a transformed target.
* [`sklearn.datasets.fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) - Fetch dataset from openml by name or dataset id.
* [`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - A random forest classifier.
* [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) - A random forest regressor.
* [`sklearn.feature_selection.SelectFromModel`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) - Meta-transformer for selecting features based on importance weights.
* [`sklearn.impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) - Univariate imputer for completing missing values with simple strategies.
* [`sklearn.linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - Ordinary least squares Linear Regression.
* [`sklearn.linear_model.SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) - Linear classifiers (SVM, logistic regression, etc.) with SGD (stochastic gradient descent) training.
* [`sklearn.metrics.confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) - Compute confusion matrix to evaluate the accuracy of a classification.
* [`sklearn.metrics.ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) - Confusion Matrix visualization.
* [`sklearn.metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) - Compute the F1 score, also known as balanced F-score or F-measure.
* [`sklearn.metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - Mean squared error regression loss.
* [`sklearn.metrics.pairwise.rbf_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html) - Compute the rbf (gaussian) kernel between X and Y.
* [`sklearn.metrics.precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) - Compute precision-recall pairs for different probability thresholds.
* [`sklearn.metrics.precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) - Compute the precision.
* [`sklearn.metrics.recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) - Compute the recall.
* [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
* [`sklearn.metrics.roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) - Compute Receiver operating characteristic (ROC).
* [`sklearn.metrics.root_mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html) - Root mean squared error regression loss.
* [`sklearn.model_selection.cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) - Generate cross-validated estimates for each input data point.
* [`sklearn.model_selection.cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) - Evaluate a score by cross-validation.
* [`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) - Exhaustive search over specified parameter values for an estimator.
* [`sklearn.model_selection.HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html) - Randomized search on hyper parameters.
* [`sklearn.model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) - Randomized search on hyper parameters.
* [`sklearn.model_selection.StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) - Class-wise stratified K-Fold cross-validator.
* [`sklearn.model_selection.StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) - Class-wise stratified ShuffleSplit cross-validator.
* [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - Split arrays or matrices into random train and test subsets.
* [`sklearn.multiclass.OneVsRestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) - One-vs-the-rest (OvR) multiclass strategy.
* [`sklearn.multioutput.ClassifierChain`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html) - A multi-label model that arranges binary classifiers into a chain.
* [`sklearn.neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) - Classifier implementing the k-nearest neighbors vote.
* [`sklearn.neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) - Regression based on k-nearest neighbors.
* [`sklearn.pipeline.make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) - Construct a `Pipeline` from the given estimators.
* [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) - A sequence of data transformers with an optional final predictor.
* [`sklearn.preprocessing.add_dummy_feature`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.add_dummy_feature.html) - Augment dataset with an additional dummy feature.
* [`sklearn.preprocessing.FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) - Constructs a transformer from an arbitrary callable.
* [`sklearn.preprocessing.MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) - Transform features by scaling each feature to a given range.
* [`sklearn.preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) - Encode categorical features as a one-hot numeric array.
* [`sklearn.preprocessing.OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) - Encode categorical features as an integer array.
* [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) - Standardize features by removing the mean and scaling to unit variance.
* [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - C-Support Vector Classification.
* [`sklearn.svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) - Epsilon-Support Vector Regression.
* [`sklearn.tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) - A decision tree regressor.
* [`sklearn.utils.estimator_checks.check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html) - Check if estimator adheres to scikit-learn conventions.
* [`sklearn.utils.validation.check_array`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_array.html) - Input validation on an array, list, sparse matrix or similar.
* [`sklearn.utils.validation.check_is_fitted`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html) - Perform is_fitted validation for estimator.
* [`sklearn.utils.validation.validate_data`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.validate_data.html) - Validate input data and set or check feature names and counts of the input.

### SciPy

* [`scipy.stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) - A normal continuous random variable.
* [`scipy.stats.randint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html) - A uniform discrete random variable.
* [`scipy.stats.sem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html) - Compute standard error of the mean.
* [`scipy.stats.t`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) - A Student's t continuous random variable.
