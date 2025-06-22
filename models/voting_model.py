from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from .random_forest import RandomForest
from .xgb import XGB

class VoteModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_valid, self.y_train, self.y_valid = \
        train_test_split(X, y, test_size=0.2, random_state=1201)
        self.numerical_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        self.rf_model = RandomForest(X, y).model
        self.xgb_model = XGB(X, y).model

        self.voting_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('xgb', self.xgb_model)
            ],
            voting='soft'
        )
        self.voting_pipeline()

    def voting_pipeline(self):
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', num_transformer, self.numerical_features),
            ('cat', cat_transformer, self.categorical_features)
        ])

        self.final_pipeline = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('model', self.voting_model)
        ])

    def train(self):
        self.final_pipeline.fit(self.X_train, self.y_train)
        acc = self.final_pipeline.score(self.X_valid, self.y_valid)

        return acc
    
    def test(self, X):
        preds = self.final_pipeline.predict(X)
        return preds
    
