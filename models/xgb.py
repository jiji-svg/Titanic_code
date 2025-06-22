import json
from xgboost import XGBClassifier
from base import BaseModel

class XGB(BaseModel):
    def __init__(self, X, y):
        with open('config.json') as f:
            config = json.load(f)

        super().__init__(X, y, XGBClassifier, config['xgb_params'])
