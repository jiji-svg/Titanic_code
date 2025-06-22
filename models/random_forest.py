import json
from sklearn.ensemble import RandomForestClassifier
from base import BaseModel


class RandomForest(BaseModel):
    def __init__(self, X, y):
        with open('config.json') as f:
            config = json.load(f)
        super().__init__(X, y, RandomForestClassifier, config['rf_params'])
