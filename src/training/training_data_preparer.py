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
    """
    Builds the preprocessing pipeline and splits the dataset
    into training and test sets for model training.

    The preparer groups features by type
    (numerical, ordinal, categorical, list-like, and historical)
    and assembles a ``ColumnTransformer`` that imputes and
    encodes each group appropriately. Splitting is performed
    chronologically by a cutoff year.

    :param list numerical_features: Names of numeric feature columns.
    :param list ordinal_features: Names of ordinal feature columns.
    :param list ordinal_categories: List of category orderings, one per
        entry in ``ordinal_features``, passed to ``OrdinalEncoder``.
    :param list categorical_features: Names of nominal categorical columns.
    :param list list_features: Names of columns containing comma-separated
        lists to be vectorized with ``CountVectorizer``.
    :param list historical_features: Names of engineered historical feature
        columns.

    :ivar preprocessor: Assembled ``ColumnTransformer``.
    :ivar train_dataset: Full training subset (with metadata columns).
    :ivar X_train: Training feature matrix.
    :ivar y_train: Training target (``Score``).
    :ivar test_dataset: Full test subset (with metadata columns).
    :ivar X_test: Test feature matrix.
    :ivar y_test: Test target (``Score``).
    """

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
        """
        Builds a ``ColumnTransformer`` that applies the proper imputation
        and encoding strategy to each feature group:

        - numerical: median imputation and standard scaling,
        - ordinal: most-frequent imputation and ordinal encoding,
        - categorical: most-frequent imputation and one-hot encoding,
        - historical: constant (``-1``) imputation and standard scaling,
        - list-like: constant (``''``) imputation and binary count
          vectorization per column.

        :return: A configured ``ColumnTransformer`` with ``remainder='drop'``.
        """
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
        """
        Splits ``expanded_dataset`` chronologically into training and
        test sets and populates the ``X_*`` / ``y_*`` attributes.

        Rows with ``'Released_Year' < split_year`` form the training set;
        the remaining rows form the test set. Only the configured
        feature columns are kept in the feature matrices; the target
        is the ``Score`` column.

        :param pd.DataFrame expanded_dataset: Dataset produced by ``FeatureEngineer``.
        :param int split_year: Year used as the train/test cutoff
            (exclusive upper bound for training).
        """
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
