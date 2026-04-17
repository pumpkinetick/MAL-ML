from typing import Callable

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Builds historical, time-aware features from the cleaned anime dataset.

    For each row, the engineer uses only data from earlier years
    (optionally restricted to the past ``k_years``)
    to compute metrics such as past performance, momentum, consistency, and experience
    for multivalued attributes like studios, producers, genres, and themes.

    :param pd.DataFrame cleaned_dataset: Already-cleaned dataset produced by ``DataCleaner``.

    :ivar expanded_dataset: Dataset extended with historical feature columns.
    :ivar historical_features: Names of the columns added by the engineer.
    """

    def __init__(self, cleaned_dataset: pd.DataFrame):
        self.expanded_dataset = cleaned_dataset.copy()

        self.historical_features = list()

    def expand_dataset(self, k_years: int):
        """
        Adds historical feature columns to ``expanded_dataset``.

        For ``Studios``, ``Producers``, ``Genres``, and ``Themes``,
        a past-performance average and a momentum slope are computed.
        For ``Studios`` and ``Producers``,
        score consistency and total experience are additionally computed.

        :param int k_years: Size of the rolling window (in years) used when
            aggregating historical performance and momentum. Consistency
            uses the same window; experience uses the full history.
        """
        for col in ['Studios', 'Producers', 'Genres', 'Themes']:
            self._add_historical_metric(
                source_col=col,
                metric_name='Past_Perf',
                k_years=k_years,
                func=(
                    lambda valid: np.mean([np.mean([s for yr, s in v]) for v in valid])
                    if valid else -1
                )
            )
            self._add_historical_metric(
                source_col=col,
                metric_name='Momentum',
                k_years=k_years,
                func=self._calculate_momentum
            )

        for col in ['Studios', 'Producers']:
            self._add_historical_metric(
                source_col=col,
                metric_name='Consistency',
                k_years=k_years,
                func=(
                    lambda valid: np.mean([
                        np.std([s for yr, s in v]) for v in valid
                        if len(v) >= 2
                    ])
                    if any(len(v) >= 2 for v in valid) else -1
                )
            )
            self._add_historical_metric(
                source_col=col,
                metric_name='Experience',
                k_years=None,
                func=lambda valid: np.mean([len(v) for v in valid]) if valid else 0
            )

    def _add_historical_metric(self,
                               source_col: str,
                               metric_name: str,
                               func: Callable,
                               k_years: int | None = None
                               ):
        """
        Computes a single historical metric and appends it as a new
        column ``"{source_col}_{metric_name}"`` to ``expanded_dataset``.

        The dataset is expected to be sorted chronologically so that
        only past entries contribute to each row's metric. For every
        row, the method collects the history of each entity listed in
        ``source_col`` (comma-separated) that is older than the row's
        release year (optionally within a ``k_years`` window) and
        applies ``func`` to aggregate those histories into a single
        score. Afterward, the current row's score is added to each
        entity's history.

        :param str source_col: Name of the column containing a
            comma-separated list of entities (e.g. ``'Studios'``).
        :param str metric_name: Suffix used to name the generated column.
        :param Callable func: Aggregation function that takes a list of valid
            per-entity history lists and returns a numeric score.
        :param int | None k_years: Optional rolling window size in years.
            If ``None``, the entire past is considered.
        """
        history = dict()

        def get_metric(row: pd.Series) -> float:
            entities = str(row[source_col]).split(', ')
            current_year = row['Released_Year']

            valid_histories = list()
            for ent in entities:
                if ent in history:
                    if k_years:
                        valid = [
                            (yr, s) for yr, s in history[ent]
                            if current_year - k_years <= yr < current_year
                        ]
                    else:
                        valid = [
                            (yr, s) for yr, s in history[ent]
                            if yr < current_year
                        ]

                    if valid:
                        valid_histories.append(valid)

            for ent in entities:
                if ent not in history:
                    history[ent] = list()
                history[ent].append((current_year, row['Score']))

            return func(valid_histories)

        new_col = f'{source_col}_{metric_name}'
        self.expanded_dataset[new_col] = self.expanded_dataset.apply(get_metric, axis=1)
        self.historical_features.append(new_col)

    @staticmethod
    def _calculate_momentum(valid_histories: list) -> float:
        """
        Estimates a momentum score by comparing the average score of
        the most recent half of each history with the average of the
        older half, then averaging these differences across entities.

        :param list valid_histories: List of per-entity history lists,
            each containing ``(year, score)`` tuples. Entities with
            fewer than two past entries are skipped.

        :return: Mean of the newer-minus-older averages across all
            eligible entities, or ``0`` if none are eligible.
        """
        slopes = list()
        for valid in valid_histories:
            if len(valid) >= 2:
                valid.sort()
                mid = len(valid) // 2
                old_half = [s for yr, s in valid[:mid]]
                new_half = [s for yr, s in valid[mid:]]
                slopes.append(np.mean(new_half) - np.mean(old_half))

        return np.mean(slopes) if slopes else 0
