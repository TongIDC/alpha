import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-114267.48174869406
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=10.0, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.1)),
    VarianceThreshold(threshold=0.0001),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=1, min_samples_split=2, n_estimators=100)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.2, min_samples_leaf=6, min_samples_split=13, n_estimators=100)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.6000000000000001, min_samples_leaf=9, min_samples_split=17, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
