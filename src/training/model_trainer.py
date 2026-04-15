import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


class ModelTrainer:
    def __init__(self):
        self.model = None

    def build_pipeline(self,
                       preprocessor: ColumnTransformer,
                       model_params: dict
                       ):
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(**model_params))
        ])

    def train_model(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    preprocessor: ColumnTransformer,
                    model_params: dict
                    ):
        if self.model is None:
            self.build_pipeline(
                preprocessor=preprocessor,
                model_params=model_params
            )
        self.model.fit(X_train, y_train)

    def save_model(self, filename: str):
        if self.model:
            joblib.dump(self.model, filename)
            print(f'Model saved to {filename}.')
        else:
            print('No model to save.')

    def load_model(self, filename: str):
        try:
            self.model = joblib.load(filename)
            print(f'Model loaded from {filename}.')
        except FileNotFoundError:
            print(f'File {filename} not found.')
