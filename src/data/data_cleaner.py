import re

import numpy as np
import pandas as pd


class DataCleaner:
    def __init__(self, dataset: pd.DataFrame):
        self.cleaned_dataset = dataset.copy()

    def clean_dataset(self):
        self.filter_dataset()
        self.fix_dtypes()
        self.sort_dataset()

    def filter_dataset(self):
        management_cols = [
            'myanimelist_id', 'Type', 'Status',
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

    def fix_dtypes(self):
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

    def sort_dataset(self):
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
