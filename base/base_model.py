import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from abc import abstractmethod

class BaseModel:
    def __init__(self, X, y, model, param_config):
        self.X = X
        self.y = y
        self.model_class = model
        self.model = None
        self.param_config = param_config
        self.best_params = self.study()
    
    def build_params(self, trial):
        params = {}
        for key, val in self.param_config.items():
            if val[0] == "int":
                params[key] = trial.suggest_int(key, val[1], val[2])
            elif val[0] == "float":
                params[key] = trial.suggest_float(key, val[1], val[2])
            elif val[0] == "fixed":
                params[key] = val[1]
            else:
                raise ValueError(f"Unknown param type for {key}: {val[0]}")
        return params
    
    def objective(self, trial):
        params = self.build_params(trial)
        self.model = self.model_class(**params)
        score = cross_val_score(self.model, self.X, self.y, cv=5, scoring='accuracy').mean()

        return score
    
    def study(self):
        rf_study = optuna.create_study(direction='maximize')
        rf_study.optimize(self.objective, n_trials=100)

        self.best_params = rf_study.best_params
        print(f'{self.model_class.__name__}\'s best params', self.best_params)
        self.model = self.model_class(**self.best_params)
        return self.best_params
