from typing import Optional

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
    :param pd.DataFrame train_dataset: Full training subset (with metadata columns).
    :param pd.DataFrame X_train: Training feature matrix.
    :param pd.Series y_train: Training target (``Score``).
    :param pd.DataFrame test_dataset: Full test subset (with metadata columns).
    :param pd.DataFrame X_test: Test feature matrix.
    :param pd.Series y_test: Test target (``Score``).

    :ivar pd.Series y_train_pred: Predicted scores for the training set.
    :ivar pd.Series y_test_pred: Predicted scores for the test set.
    """

    def __init__(self,
                 model: Pipeline,
                 train_dataset: pd.DataFrame,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 test_dataset: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_test: pd.Series
                 ):
        self.model = model

        self.train_dataset = train_dataset
        self.y_train = y_train

        self.test_dataset = test_dataset
        self.y_test = y_test

        self.y_train_pred = self.model.predict(X_train)
        self.y_test_pred = self.model.predict(X_test)

    def get_overall_metrics(self) -> pd.DataFrame:
        """
        Computes MAE, RMSE, and R² on both the training and test sets.

        :return: ``DataFrame`` with one row per dataset
            (``'Train'`` and ``'Test'``) and columns
            ``'MAE'``, ``'RMSE'``, ``'R$^2$'``.
        """
        metrics = [
            {'Dataset': 'Train',
             'MAE': mean_absolute_error(self.y_train, self.y_train_pred),
             'RMSE': np.sqrt(mean_squared_error(self.y_train, self.y_train_pred)),
             'R$^2$': r2_score(self.y_train, self.y_train_pred)},

            {'Dataset': 'Test',
             'MAE': mean_absolute_error(self.y_test, self.y_test_pred),
             'RMSE': np.sqrt(mean_squared_error(self.y_test, self.y_test_pred)),
             'R$^2$': r2_score(self.y_test, self.y_test_pred)}
        ]

        return pd.DataFrame(metrics)

    def get_seasonal_ndcg(self,
                          target_year: Optional[int] = None
                          ) -> pd.DataFrame:
        """
        Computes the NDCG ranking score for each season of each year
        (or optionally for ``target_year`` only),
        treating true scores as relevance values.

        Seasons with fewer than two test items are skipped.

        :param Optional[int] target_year: Year whose four seasons should be scored.
            If ``None``, all test items are included.

        :return: ``DataFrame`` with columns ``'Year'``, ``'Season'``, and ``'NDCG'``.
        """
        valid_years = self.test_dataset['Released_Year'].unique()
        if target_year and target_year not in valid_years:
            raise ValueError(f'Invalid target year: {target_year}')

        seasonal_ndcg = list()

        def append_ndcg_score(t_year: int, t_season: str):
            mask = self.test_dataset['Premiered'] == f'{t_season} {t_year}'
            if mask.sum() > 5:
                true_rel = self.y_test[mask].values
                pred_scores = self.y_test_pred[np.where(mask)[0]]

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
                             target_season: Optional[str] = None
                             ) -> pd.DataFrame:
        """
        Returns the actual and predicted scores (and their difference)
        for items in a specific season, or for the entire test set if
        no season is specified.

        :param Optional[str] target_season: Optional season label (e.g. ``'Fall 2024'``).
            If ``None``, all test items are included.

        :return: ``DataFrame`` with columns ``'Title'``, ``'Actual Score'``,
            ``'Predicted Score'``, and ``'Difference'``, sorted by
            absolute error ascending.
        """
        if target_season:
            if target_season not in self.test_dataset['Premiered'].unique():
                raise ValueError(f'Invalid season: {target_season}')
            mask = self.test_dataset['Premiered'] == target_season
        else:
            mask = pd.Series(data=True, index=self.test_dataset.index)

        if mask.any():
            comparison_df = pd.DataFrame({
                'Title': self.test_dataset[mask]['title'].values,
                'Actual Score': self.y_test[mask].values,
                'Predicted Score': self.y_test_pred[np.where(mask)[0]],
                'Difference': self.y_test_pred[np.where(mask)[0]] - self.y_test[mask].values
            })

            comparison_df['Abs_Error'] = comparison_df['Difference'].abs()
            comparison_df.sort_values(by='Abs_Error', ascending=True, inplace=True)
            comparison_df.drop(columns=['Abs_Error'], inplace=True)

            comparison_df.reset_index(drop=True, inplace=True)
            comparison_df.index += 1

            return comparison_df

        return pd.DataFrame()

    def get_metrics_by_source(self) -> pd.DataFrame:
        """
        Computes MAE, RMSE, and row count for each ``Source`` value in
        the test set, sorted ascending by MAE. The ``'Other'`` bucket
        is excluded.

        :return: ``DataFrame`` with columns ``'Source'``, ``'MAE'``,
            ``'RMSE'``, and ``'Count'``.
        """
        source_metrics = list()
        for source in self.test_dataset['Source'].unique():
            if source == 'Other':
                continue

            mask = self.test_dataset['Source'] == source
            if mask.any():
                s_true = self.y_test[mask]
                s_pred = self.y_test_pred[np.where(mask)[0]]

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
                                     step: float = 1.0,
                                     random_runs: int = 10,
                                     seed: int = 42
                                     ) -> pd.DataFrame:
        """
        Computes MAE, average seasonal NDCG, and a stable random NDCG baseline.
        The baseline is the mean of multiple seeded random permutations.

        :param float step: The increment between thresholds. Defaults to 1.0.
        :param int random_runs: Number of random attempts to average for
            the baseline.
        :param int seed: Seed for reproducibility.

        :return: ``DataFrame`` with columns ``'Threshold'``, ``'MAE'``,
            ``'NDCG'``, ``'Random NDCG'``, and ``'Count'``.
        """
        np.random.seed(seed)

        thresholds = np.arange(1.0, 10.0, step)

        cumulative_results = list()
        for t in thresholds:
            mask = self.y_test >= t
            if mask.sum() > 5:
                current_mae = mean_absolute_error(
                    self.y_test[mask], self.y_test_pred[np.where(mask)[0]]
                )

                relevant_seasons = self.test_dataset[mask]['Premiered'].unique()

                seasonal_ndcg = list()
                seasonal_random_ndcg = list()
                for season in relevant_seasons:
                    s_mask = (self.test_dataset['Premiered'] == season) & mask
                    if s_mask.sum() > 5:
                        s_true = self.y_test[s_mask].values
                        s_pred = self.y_test_pred[np.where(s_mask)[0]]

                        seasonal_ndcg.append(ndcg_score([s_true], [s_pred]))

                        run_scores = list()
                        for _ in range(random_runs):
                            s_pred_rand = np.random.permutation(s_pred)
                            run_scores.append(ndcg_score([s_true], [s_pred_rand]))

                        seasonal_random_ndcg.append(np.mean(run_scores))

                cumulative_results.append({
                    'Threshold': f'{t:.1f}≤',
                    'MAE': current_mae,
                    'NDCG': np.mean(seasonal_ndcg) if seasonal_ndcg else np.nan,
                    'Random NDCG': np.mean(seasonal_random_ndcg) if seasonal_random_ndcg else np.nan,
                    'Count': mask.sum()
                })

        return pd.DataFrame(cumulative_results)
