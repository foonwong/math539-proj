gridParams = {
    'learning_rate': [0.1, 0.05, 0.005],
    'max_depth': [3, 5, 7],
    'num_leaves': [30, 80],
    'min_child_samples': [300, 700, 1500],
    # 'bagging_fraction' : np.logspace(-1, 0, 3),
    # 'feature_fraction' : np.logspace(-1, 0, 3), 
    # 'lambda_l2' : np.logspace(-3, 2, 10) 
}

lgb_reg = LGBMRegressor(task='train',
    random_state= seed, 
    boosting_type= 'gbdt',
    device='gpu',
    n_jobs= 4, 
    verbose= [-1]
)

lgb_grid = GridSearchCV(lgb_reg, gridParams,
    verbose=10,
    cv=5, n_jobs=4
)