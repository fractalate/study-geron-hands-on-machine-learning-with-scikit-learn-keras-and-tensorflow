# Hands-on Machine Learning with Scikit-learn, Keras, and TensorFlow

My study notes and worked exercises from
[Hands-on Machine Learning with Scikit-learn, Keras, and TensorFlow (3rd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/).

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

* [`sklearn.neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) - Regression based on k-nearest neighbors.
* [`sklearn.linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - Ordinary least squares Linear Regression.
* [`sklearn.base.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) - Base class for all estimators in scikit-learn.
* [`sklearn.base.TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html) - Mixin class for all transformers in scikit-learn.
* [`sklearn.base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html) - Construct a new unfitted estimator with the same parameters.
* [`sklearn.compose.ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) - Applies transformers to columns of an array or pandas DataFrame.
* [`sklearn.compose.make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html) - Create a callable to select columns to be used with `ColumnTransformer`.
* [`sklearn.compose.TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) - Meta-estimator to regress on a transformed target.
* [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) - A random forest regressor.
* [`sklearn.feature_selection.SelectFromModel`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) - Meta-transformer for selecting features based on importance weights.
* [`sklearn.impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) - Univariate imputer for completing missing values with simple strategies.
* [`sklearn.linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - Ordinary least squares Linear Regression.
* [`sklearn.metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - Mean squared error regression loss.
* [`sklearn.metrics.root_mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html) - Root mean squared error regression loss.
* [`sklearn.metrics.pairwise.rbf_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html) - Compute the rbf (gaussian) kernel between X and Y.
* [`sklearn.model_selection.cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) - Evaluate a score by cross-validation.
* [`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) - Exhaustive search over specified parameter values for an estimator.
* [`sklearn.model_selection.HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html) - Randomized search on hyper parameters.
* [`sklearn.model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) - Randomized search on hyper parameters.
* [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - Split arrays or matrices into random train and test subsets.
* [`sklearn.model_selection.StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) - Class-wise stratified ShuffleSplit cross-validator.
* [`sklearn.neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) - Regression based on k-nearest neighbors.
* [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) - A sequence of data transformers with an optional final predictor.
* [`sklearn.pipeline.make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) - Construct a `Pipeline` from the given estimators.
* [`sklearn.preprocessing.FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) - Constructs a transformer from an arbitrary callable.
* [`sklearn.preprocessing.MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) - Transform features by scaling each feature to a given range.
* [`sklearn.preprocessing.OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) - Encode categorical features as an integer array.
* [`sklearn.preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) - Encode categorical features as a one-hot numeric array.
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
