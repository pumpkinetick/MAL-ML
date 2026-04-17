import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualizer:
    """
    Collection of static plotting helpers for evaluation results.

    Each method takes a ``DataFrame`` produced by ``Evaluator``
    and renders a figure.
    """

    @staticmethod
    def plot_overall_metrics(metrics_df: pd.DataFrame):
        """
        Plots train-versus-test bar charts for
        MAE, RMSE, and R² side by side.

        :param pd.DataFrame metrics_df: ``DataFrame`` as returned by
            ``Evaluator.get_overall_metrics()``, with columns
            ``'MAE'``, ``'RMSE'``, and ``'R$^2$'``.
        """
        melted_df = metrics_df.melt(
            id_vars='Dataset', var_name='Metric', value_name='Value'
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 8))

        for i, metric in enumerate(['MAE', 'RMSE', 'R$^2$']):
            sns.barplot(
                data=melted_df[melted_df['Metric'] == metric],
                x='Dataset', y='Value',
                ax=axes[i],
                palette={'Train': '#d4a368', 'Test': '#8FB8CF'},
                hue='Dataset',
                legend=False
            )
            axes[i].set_title(f'Overall {metric}')
            axes[i].set_ylabel('Score')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_seasonal_ndcg(seasonal_ndcg_df: pd.DataFrame,
                           target_year: int
                           ):
        """
        Plots NDCG scores across the four seasons of a given year.

        The y-axis lower bound is adjusted to zoom in on the score
        range while staying capped at 1.

        :param pd.DataFrame seasonal_ndcg_df: ``DataFrame`` as returned by
            ``Evaluator.get_seasonal_ndcg()``, with columns
            ``'Season'`` and ``'NDCG'``.
        :param int target_year: Year represented by the data;
            used only in the plot title.
        """
        plt.figure(figsize=(8, 6))
        sns.barplot(data=seasonal_ndcg_df, x='Season', y='NDCG')
        plt.title(f'Seasonal NDCG for {target_year}')
        y_min = max(0, min(seasonal_ndcg_df['NDCG']) - 0.1)
        plt.ylim(y_min, 1)
        plt.show()

    @staticmethod
    def plot_mae_by_source(source_metrics_df: pd.DataFrame):
        """
        Plots MAE per source as a bar chart, annotating each bar with
        the number of samples in that source.

        :param pd.DataFrame source_metrics_df: ``DataFrame`` as returned by
            ``Evaluator.get_metrics_by_source()``, with columns
            ``'Source'``, ``'MAE'``, and ``'Count'``.
        """
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(data=source_metrics_df, x='Source', y='MAE')

        for i, p in enumerate(ax.patches):
            count = int(source_metrics_df.iloc[i]['Count'])
            ax.annotate(
                text=f'{count}',
                xy=(p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )

        plt.xticks(rotation=45)
        plt.title('MAE by Source')
        plt.show()

    @staticmethod
    def plot_feature_importances(importance_df: pd.DataFrame,
                                 top_n: int = 20
                                 ):
        """
        Plots the most important ``top_n`` number of features
        from a feature-importance ``DataFrame`` as a horizontal bar chart.

        :param pd.DataFrame importance_df: ``DataFrame`` as returned by
            ``Evaluator.get_feature_importances()``, with columns
            ``'Feature'`` and ``'Importance'`` (sorted descending).
        :param int top_n: Number of top features to display. Defaults to ``20``.
        """
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Features by Importance')
        plt.show()
