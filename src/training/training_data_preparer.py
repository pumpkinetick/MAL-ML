import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)


class TrainingDataPreparer:
    def __init__(self,
                 numerical_features: list,
                 ordinal_features: list,
                 ordinal_categories: list,
                 categorical_features: list,
                 list_features: list,
                 historical_features: list
                 ):
        self.numerical_features = numerical_features
        self.ordinal_features = ordinal_features
        self.ordinal_categories = ordinal_categories
        self.categorical_features = categorical_features
        self.list_features = list_features
        self.historical_features = historical_features

        self.train_dataset = None
        self.X_train = None
        self.y_train = None

        self.test_dataset = None
        self.X_test = None
        self.y_test = None

        self.preprocessor = self.get_preprocessor()

    def get_preprocessor(self) -> ColumnTransformer:
        transformers = [
            ('num', Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
            ]), self.numerical_features),

            ('ord', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OrdinalEncoder(
                    categories=self.ordinal_categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ))
            ]), self.ordinal_features),

            ('cat', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), self.categorical_features),

            ('hist', Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value=-1)),
                ('scale', StandardScaler())
            ]), self.historical_features)
        ]

        for col in self.list_features:
            transformers.append(
                (f'list_{col}', Pipeline([
                    ('impute', SimpleImputer(strategy='constant', fill_value='')),
                    ('flatten', FunctionTransformer(lambda x: x.flatten())),
                    ('vec', CountVectorizer(
                        tokenizer=lambda x: x.split(', '),
                        token_pattern=None,
                        binary=True,
                        stop_words=None,
                        min_df=2
                    ))
                ]), [col])
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

    def prepare_datasets(self,
                         expanded_dataset: pd.DataFrame,
                         split_year: int
                         ):
        self.train_dataset = expanded_dataset[
            expanded_dataset['Released_Year'] < split_year
        ]
        self.test_dataset = expanded_dataset[
            expanded_dataset['Released_Year'] >= split_year
        ]

        features = (self.numerical_features +
                    self.ordinal_features +
                    self.categorical_features +
                    self.list_features +
                    self.historical_features)

        self.X_train, self.y_train = self.train_dataset[features], self.train_dataset['Score']
        self.X_test, self.y_test = self.test_dataset[features], self.test_dataset['Score']
