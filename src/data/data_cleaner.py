import re

import numpy as np
import pandas as pd


class DataCleaner:
    """
    Cleans the raw MyAnimeList dataset.

    Filters out irrelevant rows and columns, fixes data types
    (notably parsing the ``Duration`` field into minutes),
    groups rare source values into a single ``'Other'`` bucket,
    and orders the dataset chronologically by release year and season.

    :param pd.DataFrame dataset: Raw anime dataset to be cleaned.

    :ivar cleaned_dataset: Working copy of the dataset that is updated
        in place by each cleaning step.
    """

    def __init__(self, dataset: pd.DataFrame):
        self.cleaned_dataset = dataset.copy()

    def clean_dataset(self):
        """
        Runs the full cleaning pipeline on ``cleaned_dataset``.
        """
        self._filter_dataset()
        self._fix_dtypes()
        self._sort_dataset()

    def _filter_dataset(self):
        """
        Selects relevant columns, drops rows with missing key values,
        keeps only finished-airing TV entries, and groups rare sources
        (fewer than 10 occurrences) under ``'Other'``.
        """
        management_cols = [
            'myanimelist_id', 'title', 'Type', 'Status',
            'Premiered', 'Released_Season', 'Released_Year'
        ]
        feature_cols = [
            'Source', 'Genres', 'Themes',
            'Studios', 'Producers', 'Demographic',
            'Duration', 'Rating'
        ]
        target_col = ['Score']

        cols_to_keep = management_cols + feature_cols + target_col
        self.cleaned_dataset = self.cleaned_dataset[cols_to_keep].copy()

        self.cleaned_dataset.dropna(
            subset=management_cols + target_col,
            inplace=True
        )

        self.cleaned_dataset = self.cleaned_dataset[
            (self.cleaned_dataset['Type'] == 'TV')
            &
            (self.cleaned_dataset['Status'] == 'Finished Airing')
        ].copy()

        # Group rare sources
        source_counts = self.cleaned_dataset['Source'].value_counts()
        rare_sources = source_counts[source_counts < 10].index
        self.cleaned_dataset['Source'] = self.cleaned_dataset['Source'].apply(
            lambda x: 'Other' if x in rare_sources else x
        )

    def _fix_dtypes(self):
        """
        Converts columns to their proper types:
        ``Released_Year`` to ``int``, trims whitespace from ``Rating``,
        and parses ``Duration`` strings (e.g. ``"24 min"`` or ``"30 sec"``)
        into a numeric number of minutes.
        """
        self.cleaned_dataset['Released_Year'] = (
            self.cleaned_dataset['Released_Year'].astype(int)
        )
        self.cleaned_dataset['Rating'] = self.cleaned_dataset['Rating'].str.strip()

        def convert_duration_to_minutes(duration_str: str) -> float:
            match = re.search(pattern=r'(\d+)\s*(min|sec)', string=str(duration_str))
            if match:
                value = float(match.group(1))
                unit = match.group(2)

                if unit == 'sec':
                    return value / 60.0
                return value
            return np.nan

        self.cleaned_dataset['Duration'] = (
            self.cleaned_dataset['Duration']
            .apply(convert_duration_to_minutes)
        )

    def _sort_dataset(self):
        """
        Sorts the dataset chronologically by ``Released_Year`` and then
        by ``Released_Season`` (Winter → Spring → Summer → Fall).
        """
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']

        self.cleaned_dataset['Released_Season'] = pd.Categorical(
            self.cleaned_dataset['Released_Season'],
            categories=season_order,
            ordered=True
        )
        self.cleaned_dataset.sort_values(
            by=['Released_Year', 'Released_Season'],
            inplace=True
        )
