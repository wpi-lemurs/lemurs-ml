'''
Dictionaries to save the optimal tuned hyperparameters for different models.
'''

# 70/30 temporal split, sleep from screentime
TEMPORAL_SLEEP_PARAMS = {
    'logistic_regression': {
        'C': 100.0,
        'class_weight': None,
        'solver': 'lbfgs'
    },
    'random_forest': {
        'n_estimators': 500,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.5,
        'scale_pos_weight': 1,
        'reg_alpha': 0,
        'n_estimators': 400,
        'min_child_weight': 10,
        'max_depth': 4,
        'learning_rate': 0.05,
        'gamma': 5,
        'colsample_bytree': 0.9
    }
}

# 70/30 random split, sleep from screentime
RANDOM_SLEEP_PARAMS = {
    'logistic_regression': {
        'C': 0.01,
        'class_weight': 'balanced',
        'solver': 'lbfgs'
    },
    'random_forest': {
        'n_estimators': 500,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 4,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.7,
        'scale_pos_weight': 1,
        'reg_alpha': 0,
        'n_estimators': 600,
        'min_child_weight': 3,
        'max_depth': 5,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 0.7
    }
}

# 70/30 temporal split, self harm from screentime
TEMPORAL_SELF_HARM_PARAMS = {
    'logistic_regression': {
        'C': 0.01,
        'class_weight': 'balanced',
        'solver': 'liblinear'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 4,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.5,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 0,
        'n_estimators': 800,
        'min_child_weight': 10,
        'max_depth': 4,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 0.9
    }
}


# 70/30 random split, self harm from screentime
RANDOM_SELF_HARM_PARAMS = {
    'logistic_regression': {
        'C': 0.01,
        'class_weight': 'balanced',
        'solver': 'liblinear'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 4,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.5,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 1,
        'n_estimators': 400,
        'min_child_weight': 10,
        'max_depth': 5,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 0.7
    }
}

# 70/30 temporal split, suicide from screentime
TEMPORAL_SUICIDE_PARAMS = {
    'logistic_regression': {
        'C': 0.01,
        'class_weight': 'balanced',
        'solver': 'liblinear'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.5,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 1,
        'n_estimators': 400,
        'min_child_weight': 10,
        'max_depth': 5,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 0.7
    }
}

# 70/30 random split, suicide from screentime
RANDOM_SUICIDE_PARAMS = {
    'logistic_regression': {
        'C': 0.01,
        'class_weight': 'balanced',
        'solver': 'liblinear'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.9,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 1,
        'n_estimators': 600,
        'min_child_weight': 5,
        'max_depth': 6,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 0.5
    }
}

# 70/30 random split, phq9 from screentime
RANDOM_PHQ9_PARAMS = {
    'logistic_regression': {
        'C': 0.01,
        'class_weight': 'balanced',
        'solver': 'liblinear'
    },
    'random_forest': {
        'n_estimators': 500,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 4,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.5,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 0,
        'n_estimators': 400,
        'min_child_weight': 5,
        'max_depth': 5,
        'learning_rate': 0.05,
        'gamma': 1,
        'colsample_bytree': 0.5
    }

}

# 70/30 temporal split, phq9 from screentime 144 HOURS
TEMPORAL_PHQ9_PARAMS = {
    'logistic_regression': {
        'C': 0.1,
        'class_weight': 'balanced',
        'solver': 'liblinear'
    },
    'random_forest': {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_split': 8,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.5,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 0,
        'n_estimators': 400,
        'min_child_weight': 5,
        'max_depth': 5,
        'learning_rate': 0.05,
        'gamma': 1,
        'colsample_bytree': 0.5
    }
}

# 70/30 random split, social connection from screentime
RANDOM_SOCIAL_CONNECTION_PARAMS = {
    'logistic_regression': {
        'C': 0.01,
        'class_weight': None,
        'solver': 'lbfgs'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'class_weight': None
    },
    'xgboost': {
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'reg_alpha': 0,
        'n_estimators': 400,
        'min_child_weight': 1,
        'max_depth': 5,
        'learning_rate': 0.1,
        'gamma': 5,
        'colsample_bytree': 0.9
    }
}

# 70/30 temporal split, social connection from screentime
TEMPORAL_SOCIAL_CONNECTION_PARAMS = {
    'logistic_regression': {
        'C': 0.01,
        'class_weight': None,
        'solver': 'lbfgs'
    },
    'random_forest': {
        'n_estimators': 300,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'subsample': 0.7,
        'scale_pos_weight': 1,
        'reg_alpha': 1,
        'n_estimators': 600,
        'min_child_weight': 3,
        'max_depth': 5,
        'learning_rate': 0.1,
        'gamma': 5,
        'colsample_bytree': 0.5
    }
}