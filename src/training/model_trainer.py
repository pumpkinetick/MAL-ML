import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


class ModelTrainer:
    """
    Trains, persists, and restores a Random Forest regression pipeline.

    The pipeline consists of a preprocessing ``ColumnTransformer``
    followed by a ``RandomForestRegressor``.

    :ivar model: The ``scikit-learn`` ``Pipeline`` currently held by the
        trainer. ``None`` until ``train_model()`` is invoked.
    """

    def __init__(self):
        self.model = None

    def _build_pipeline(self,
                        preprocessor: ColumnTransformer,
                        model_params: dict
                        ):
        """
        Builds a new ``Pipeline`` combining the given preprocessor with a
        ``RandomForestRegressor`` and stores it in ``model``.

        :param ColumnTransformer preprocessor: Column transformer used
            as the first step.
        :param dict model_params: Keyword arguments forwarded to
            ``RandomForestRegressor``.
        """
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
        """
        Fits the pipeline on the provided training data, building it
        first if ``model`` has not yet been initialized.

        :param pd.DataFrame X_train: Training feature matrix.
        :param pd.Series y_train: Training target values.
        :param ColumnTransformer preprocessor: Column transformer used
            when the pipeline needs to be built.
        :param dict model_params: Keyword arguments forwarded to
            ``RandomForestRegressor`` when the pipeline is built.
        """
        if self.model is None:
            self._build_pipeline(
                preprocessor=preprocessor,
                model_params=model_params
            )
        self.model.fit(X_train, y_train)

    def save_model(self, filename: str):
        """
        Serializes ``model`` to disk using ``joblib``.

        :param str filename: Path where the model should be written.
        """
        if self.model:
            joblib.dump(self.model, filename)
            print(f'Model saved to {filename}.')
        else:
            print('No model to save.')

    def load_model(self, filename: str):
        """
        Loads a previously saved model from the disk into ``model``.

        :param str filename: Path to the ``joblib`` file to load.
        """
        try:
            self.model = joblib.load(filename)
            print(f'Model loaded from {filename}.')
        except FileNotFoundError:
            print(f'File {filename} not found.')
