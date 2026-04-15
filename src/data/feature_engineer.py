from typing import Callable

import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, cleaned_dataset: pd.DataFrame):
        self.expanded_dataset = cleaned_dataset.copy()

        self.historical_features = list()

    def expand_dataset(self, k_years: int):
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
        slopes = list()
        for valid in valid_histories:
            if len(valid) >= 2:
                valid.sort()
                mid = len(valid) // 2
                old_half = [s for yr, s in valid[:mid]]
                new_half = [s for yr, s in valid[mid:]]
                slopes.append(np.mean(new_half) - np.mean(old_half))

        return np.mean(slopes) if slopes else 0
