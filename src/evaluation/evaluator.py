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
    def __init__(self, model: Pipeline):
        self.model = model

    def get_overall_metrics(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series
                            ) -> pd.DataFrame:
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
                          target_year: int
                          ) -> pd.DataFrame:
        y_pred = self.model.predict(X_test)

        seasons = ['Winter', 'Spring', 'Summer', 'Fall']

        seasonal_ndcg = list()
        for season in seasons:
            mask = test_dataset['Premiered'] == f'{season} {target_year}'
            if mask.any():
                true_rel = y_test[mask].values
                pred_scores = y_pred[np.where(mask)[0]]

                if len(true_rel) > 1:
                    ndcg = ndcg_score([true_rel], [pred_scores])
                    seasonal_ndcg.append({'Season': season, 'NDCG': ndcg})

        return pd.DataFrame(seasonal_ndcg)

    def get_score_comparison(self,
                             test_dataset: pd.DataFrame,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             target_season: str
                             ) -> pd.DataFrame:
        y_pred = self.model.predict(X_test)

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
