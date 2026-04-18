import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score
)
from sklearn.pipeline import Pipeline


class Evaluator:
    """
    Computes evaluation metrics and diagnostics for a trained
    regression pipeline.

    Supports overall regression metrics (MAE, RMSE, R²),
    per-season NDCG ranking scores, per-source error breakdowns,
    side-by-side score comparisons for a specific season,
    and feature importance extraction from the underlying Random Forest.

    :param Pipeline model: Trained ``scikit-learn`` ``Pipeline`` with a
        ``'preprocessor'`` step and a ``'regressor'`` step.

    :ivar model: The wrapped pipeline.
    """

    def __init__(self, model: Pipeline):
        self.model = model

    def get_overall_metrics(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series
                            ) -> pd.DataFrame:
        """
        Computes MAE, RMSE, and R² on both the training and test sets.

        :param pd.DataFrame X_train: Training feature matrix.
        :param pd.Series y_train: Training target values.
        :param pd.DataFrame X_test: Test feature matrix.
        :param pd.Series y_test: Test target values.

        :return: ``DataFrame`` with one row per dataset
            (``'Train'`` and ``'Test'``) and columns
            ``'MAE'``, ``'RMSE'``, ``'R$^2$'``.
        """
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        metrics = [
            {'Dataset': 'Train',
             'MAE': mean_absolute_error(y_train, y_train_pred),
             'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
             'R$^2$': r2_score(y_train, y_train_pred)},

            {'Dataset': 'Test',
             'MAE': mean_absolute_error(y_test, y_test_pred),
             'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
             'R$^2$': r2_score(y_test, y_test_pred)}
        ]

        return pd.DataFrame(metrics)

    def get_seasonal_ndcg(self,
                          test_dataset: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_test: pd.Series,
                          target_year: int | None = None
                          ) -> pd.DataFrame:
        """
        Computes the NDCG ranking score for each season of each year
        (or optionally for ``target_year`` only),
        treating true scores as relevance values.

        Seasons with fewer than two test items are skipped.

        :param pd.DataFrame test_dataset: Test subset including
            the ``'Premiered'`` column.
        :param pd.DataFrame X_test: Test feature matrix aligned
            with ``test_dataset``.
        :param pd.Series y_test: Test target values aligned
            with ``test_dataset``.
        :param int | None target_year: Year whose four seasons should be scored.
            If ``None``, the entire dataset is considered.

        :return: ``DataFrame`` with columns ``'Year'``, ``'Season'``, and ``'NDCG'``.
        """
        y_pred = self.model.predict(X_test)

        valid_years = test_dataset['Released_Year'].unique()
        if target_year and target_year not in valid_years:
            raise ValueError(f'Invalid target year: {target_year}')

        seasonal_ndcg = list()

        def append_ndcg_score(t_year: int, t_season: str):
            mask = test_dataset['Premiered'] == f'{t_season} {t_year}'
            if mask.sum() > 1:
                true_rel = y_test[mask].values
                pred_scores = y_pred[np.where(mask)[0]]

                ndcg = ndcg_score([true_rel], [pred_scores])
                seasonal_ndcg.append(
                    {'Year': t_year, 'Season': t_season, 'NDCG': ndcg}
                )

        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        for season in seasons:
            if target_year:
                append_ndcg_score(t_year=target_year, t_season=season)
            else:
                for year in valid_years:
                    append_ndcg_score(t_year=year, t_season=season)

        return pd.DataFrame(seasonal_ndcg)

    def get_score_comparison(self,
                             test_dataset: pd.DataFrame,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             target_season: str
                             ) -> pd.DataFrame:
        """
        Returns the actual and predicted scores (and their difference)
        for all test items that premiered in a specific season.

        :param pd.DataFrame test_dataset: Test subset including
            the ``'Premiered'`` column.
        :param pd.DataFrame X_test: Test feature matrix aligned
            with ``test_dataset``.
        :param pd.Series y_test: Test target values aligned
            with ``test_dataset``.
        :param str target_season: Season label to filter by,
            e.g. ``'Fall 2024'``.

        :return: ``DataFrame`` with columns ``'Actual Score'``,
            ``'Predicted Score'``, and ``'Difference'``. An empty
            ``DataFrame`` is returned if no rows match.
        """
        y_pred = self.model.predict(X_test)

        if target_season not in test_dataset['Premiered'].unique():
            raise ValueError(f'Invalid season: {target_season}')

        mask = test_dataset['Premiered'] == target_season
        if mask.any():
            return pd.DataFrame({
                'Actual Score': y_test[mask].values,
                'Predicted Score': y_pred[np.where(mask)[0]],
                'Difference': y_pred[np.where(mask)[0]] - y_test[mask].values
            })
        return pd.DataFrame()

    def get_metrics_by_source(self,
                              test_dataset: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_test: pd.Series
                              ) -> pd.DataFrame:
        """
        Computes MAE, RMSE, and row count for each ``Source`` value in
        the test set, sorted ascending by MAE. The ``'Other'`` bucket
        is excluded.

        :param pd.DataFrame test_dataset: Test subset including
            the ``'Source'`` column.
        :param pd.DataFrame X_test: Test feature matrix aligned
            with ``test_dataset``.
        :param pd.Series y_test: Test target values aligned
            with ``test_dataset``.

        :return: ``DataFrame`` with columns ``'Source'``, ``'MAE'``,
            ``'RMSE'``, and ``'Count'``.
        """
        y_pred = self.model.predict(X_test)

        source_metrics = list()
        for source in test_dataset['Source'].unique():
            if source == 'Other':
                continue

            mask = test_dataset['Source'] == source
            if mask.any():
                s_true = y_test[mask]
                s_pred = y_pred[np.where(mask)[0]]

                source_metrics.append({
                    'Source': source,
                    'MAE': mean_absolute_error(s_true, s_pred),
                    'RMSE': np.sqrt(mean_squared_error(s_true, s_pred)),
                    'Count': mask.sum()
                })

        return pd.DataFrame(source_metrics).sort_values(by='MAE')

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Extracts feature importances from the Random Forest regressor,
        resolving feature names through the preprocessing
        ``ColumnTransformer`` (including nested pipelines).

        :return: ``DataFrame`` with columns ``'Feature'`` and
            ``'Importance'``, sorted descending by importance.
        """
        preprocessor = self.model.named_steps['preprocessor']

        feature_names = list()
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder' and transformer == 'drop':
                continue

            if isinstance(transformer, Pipeline):
                current_features = list(columns)

                for step_name, step_estimator in transformer.steps:
                    if hasattr(step_estimator, 'get_feature_names_out'):
                        current_features = list(step_estimator.get_feature_names_out(
                            input_features=current_features
                        ))
                feature_names.extend(current_features)

            elif hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(input_features=columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)

        feature_importances = self.model.named_steps['regressor'].feature_importances_

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        return importance_df

    def get_cumulative_score_metrics(self,
                                     test_dataset: pd.DataFrame,
                                     X_test: pd.DataFrame,
                                     y_test: pd.Series,
                                     step: float = 1.0
                                     ) -> pd.DataFrame:
        """
        Computes MAE, average seasonal NDCG, and a random NDCG baseline for
        cumulative score thresholds (e.g., Score >= 1, >= 2, ..., >= 9).

        :param pd.DataFrame test_dataset: Test subset including
            the ``'Score'`` column.
        :param pd.DataFrame X_test: Test feature matrix aligned
            with ``test_dataset``.
        :param pd.Series y_test: Test target values aligned
            with ``test_dataset``.
        :param float step: The increment between thresholds. Defaults to 1.0.

        :return: ``DataFrame`` with columns ``'Threshold'``, ``'MAE'``,
            ``'NDCG'``, ``'Random NDCG'``, and ``'Count'``.
        """
        y_pred = self.model.predict(X_test)

        thresholds = np.arange(1.0, 10.0, step)

        cumulative_results = list()
        for t in thresholds:
            mask = y_test >= t
            if mask.sum() > 1:
                current_mae = mean_absolute_error(y_test[mask], y_pred[np.where(mask)[0]])

                relevant_seasons = test_dataset[mask]['Premiered'].unique()

                seasonal_ndcg = list()
                random_ndcg = list()
                for season in relevant_seasons:
                    s_mask = (test_dataset['Premiered'] == season) & mask
                    if s_mask.sum() > 1:
                        s_true = y_test[s_mask].values
                        s_pred = y_pred[np.where(s_mask)[0]]
                        seasonal_ndcg.append(ndcg_score([s_true], [s_pred]))

                        s_pred_random = np.random.permutation(s_pred)
                        random_ndcg.append(ndcg_score([s_true], [s_pred_random]))

                cumulative_results.append({
                    'Threshold': f'{t:.1f}≤',
                    'MAE': current_mae,
                    'NDCG': np.mean(seasonal_ndcg) if seasonal_ndcg else np.nan,
                    'Random NDCG': np.mean(random_ndcg) if random_ndcg else np.nan,
                    'Count': mask.sum()
                })

        return pd.DataFrame(cumulative_results)
