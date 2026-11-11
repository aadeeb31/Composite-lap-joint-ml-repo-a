import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from catboost import CatBoostRegressor, CatBoostClassifier

META = ['Row_ID', 'Paper_ID', 'Value_Type', 'Target_Is_Derived']
SEED = 42

TASKS = {
    'CFRP_CFRP__MaxLoad': 'CFRP_CFRP',
    'Composite_Composite__MaxLoad': 'Composite_Composite',
    'CFRP_CFRP__LSS': 'CFRP_CFRP',
    'Composite_Composite__LSS': 'Composite_Composite',
    'CFRP_CFRP__FailureMode': 'CFRP_CFRP',
    'Composite_Composite__FailureMode': 'Composite_Composite',
}

FAMILY = {'OLS': 'Linear', 'Ridge': 'Linear', 'Lasso': 'Linear', 'ElasticNet': 'Linear', 'LogReg-L2': 'Linear', 'LogReg-L1': 'Linear',
          'SVR': 'Kernel', 'SVC': 'Kernel', 'MLP': 'Neural net',
          'DT': 'Tree', 'RF': 'Forest', 'ET': 'Forest',
          'GB': 'Boosting', 'XGB': 'Boosting', 'LGBM': 'Boosting', 'AB': 'Boosting', 'CatBoost': 'Boosting',
          'KNN': 'Nearest neighbour', 'BayesRidge': 'Bayesian', 'GPR': 'Bayesian', 'GPC': 'Bayesian', 'GNB': 'Bayesian'}


def reg_models():
    # name: (estimator, grid, impute missing values, scale the target)
    kernels = [C(1.0) * Matern(1.0, nu=1.5) + WhiteKernel(0.1),
               C(1.0) * Matern(1.0, nu=2.5) + WhiteKernel(0.1),
               C(1.0) * RBF(1.0) + WhiteKernel(0.1)]
    return {
        'OLS': (LinearRegression(), {'fit_intercept': [True]}, True, False),
        'Ridge': (Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}, True, False),
        'Lasso': (Lasso(max_iter=50000), {'alpha': [1e-4, 1e-3, 1e-2, 1e-1]}, True, False),
        'ElasticNet': (ElasticNet(max_iter=50000), {'alpha': [1e-3, 1e-2, 1e-1], 'l1_ratio': [0.2, 0.5, 0.8]}, True, False),
        'SVR': (SVR(epsilon=0.05), {'C': [1, 10, 100], 'gamma': [0.01, 0.05, 0.2]}, True, True),
        'MLP': (MLPRegressor(max_iter=2000, early_stopping=True, random_state=0),
                {'hidden_layer_sizes': [(64,), (128, 64)], 'alpha': [1e-4, 1e-2]}, True, True),
        'DT': (DecisionTreeRegressor(random_state=0), {'max_depth': [6, 12, None], 'min_samples_leaf': [1, 3]}, True, False),
        'RF': (RandomForestRegressor(random_state=0, n_jobs=1),
               {'n_estimators': [200, 400], 'max_depth': [None, 12], 'min_samples_leaf': [1, 2]}, True, False),
        'ET': (ExtraTreesRegressor(random_state=0, n_jobs=1),
               {'n_estimators': [200, 400], 'max_depth': [None, 12], 'min_samples_leaf': [1, 2]}, True, False),
        'GB': (GradientBoostingRegressor(random_state=0), {'n_estimators': [200, 400], 'max_depth': [3, 5]}, True, False),
        'XGB': (xgb.XGBRegressor(random_state=0, n_jobs=1, subsample=0.8, colsample_bytree=0.8),
                {'n_estimators': [300, 600], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.1]}, False, False),
        'LGBM': (lgb.LGBMRegressor(random_state=0, n_jobs=1, verbose=-1),
                 {'n_estimators': [300, 600], 'num_leaves': [15, 31], 'learning_rate': [0.05, 0.1]}, False, False),
        'AB': (AdaBoostRegressor(random_state=0), {'n_estimators': [100, 300], 'learning_rate': [0.1, 0.5, 1.0]}, True, False),
        'CatBoost': (CatBoostRegressor(random_seed=0, verbose=0, thread_count=1, allow_writing_files=False),
                     {'iterations': [300, 600], 'depth': [4, 6], 'learning_rate': [0.05, 0.1]}, False, False),
        'KNN': (KNeighborsRegressor(), {'n_neighbors': [3, 4, 5, 6, 7], 'weights': ['uniform', 'distance']}, True, False),
        'BayesRidge': (BayesianRidge(), {'alpha_1': [1e-6, 1e-4], 'lambda_1': [1e-6, 1e-4]}, True, True),
        'GPR': (GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=0, random_state=0), {'kernel': kernels}, True, False),
    }


def clf_models():
    kernels = [C(1.0) * Matern(1.0, nu=1.5), C(1.0) * RBF(1.0)]
    return {
        'LogReg-L2': (LogisticRegression(max_iter=5000), {'C': [0.1, 1, 10, 100]}, True, False),
        'LogReg-L1': (LogisticRegression(penalty='l1', solver='saga', max_iter=5000), {'C': [0.1, 1, 10, 100]}, True, False),
        'SVC': (SVC(probability=True, random_state=0), {'C': [1, 10, 100], 'gamma': [0.01, 0.05, 0.2]}, True, False),
        'MLP': (MLPClassifier(max_iter=2000, early_stopping=True, random_state=0),
                {'hidden_layer_sizes': [(64,), (128, 64)], 'alpha': [1e-4, 1e-2]}, True, False),
        'DT': (DecisionTreeClassifier(random_state=0), {'max_depth': [6, 12, None], 'min_samples_leaf': [1, 3]}, True, False),
        'RF': (RandomForestClassifier(random_state=0, n_jobs=1),
               {'n_estimators': [200, 400], 'max_depth': [None, 12], 'min_samples_leaf': [1, 2]}, True, False),
        'ET': (ExtraTreesClassifier(random_state=0, n_jobs=1),
               {'n_estimators': [200, 400], 'max_depth': [None, 12], 'min_samples_leaf': [1, 2]}, True, False),
        'GB': (GradientBoostingClassifier(random_state=0), {'n_estimators': [100, 200], 'max_depth': [3, 5]}, True, False),
        'XGB': (xgb.XGBClassifier(random_state=0, n_jobs=1, subsample=0.8, colsample_bytree=0.8),
                {'n_estimators': [300, 600], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.1]}, False, False),
        'LGBM': (lgb.LGBMClassifier(random_state=0, n_jobs=1, verbose=-1),
                 {'n_estimators': [300, 600], 'num_leaves': [15, 31], 'learning_rate': [0.05, 0.1]}, False, False),
        'AB': (AdaBoostClassifier(random_state=0), {'n_estimators': [100, 300], 'learning_rate': [0.1, 0.5, 1.0]}, True, False),
        'CatBoost': (CatBoostClassifier(random_seed=0, verbose=0, thread_count=1, allow_writing_files=False),
                     {'iterations': [300, 600], 'depth': [4, 6], 'learning_rate': [0.05, 0.1]}, False, False),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 4, 5, 6, 7], 'weights': ['uniform', 'distance']}, True, False),
        'GNB': (GaussianNB(), {'var_smoothing': [1e-9, 1e-6, 1e-3, 1e-1]}, True, False),
        'GPC': (GaussianProcessClassifier(random_state=0), {'kernel': kernels}, True, False),
    }


def make_pipeline(model, impute=True, scale_target=False, classifier=False):
    if scale_target and not classifier:
        model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    steps = [('imputer', SimpleImputer(strategy='median'))] if impute else []
    return Pipeline(steps + [('scaler', MinMaxScaler()), ('model', model)])


def grid_prefix(scale_target=False, classifier=False):
    return 'model__regressor__' if (scale_target and not classifier) else 'model__'


def load_task(task, root='../data/processed'):
    import pandas as pd
    df = pd.read_csv(f'{root}/{TASKS[task]}/{task}.csv')
    target = df.columns[-1]
    X = df.drop(columns=META + [target])
    return df, X, df[target].values, df['Paper_ID'].values, target == 'Failure_Mode', target


def make_folds(groups, k, seed):
    # whole studies are assigned to folds; the number of records per fold is balanced greedily
    rng = np.random.RandomState(seed)
    studies = np.unique(groups)
    rng.shuffle(studies)
    sizes = {s: (groups == s).sum() for s in studies}
    order = sorted(studies, key=lambda s: -sizes[s] + rng.rand() * 0.5)
    load = np.zeros(k)
    fold = {}
    for s in order:
        j = int(np.argmin(load + rng.rand(k) * 1e-3))
        fold[s] = j
        load[j] += sizes[s]
    return np.array([fold[g] for g in groups])
