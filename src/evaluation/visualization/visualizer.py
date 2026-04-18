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

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))

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
    def plot_seasonal_ndcg(seasonal_ndcg_df: pd.DataFrame):
        """
        Plots the NDCG score distribution for each of the four seasons
        as a box plot, or optionally as a bar chart in the case of
        only one year of data.

        The y-axis lower bound is adjusted to zoom in on the score
        range while staying capped at 1.

        :param pd.DataFrame seasonal_ndcg_df: ``DataFrame`` as returned by
            ``Evaluator.get_seasonal_ndcg()``, with columns
            ``'Year'``, ``'Season'``, and ``'NDCG'``.
        """
        plt.figure(figsize=(8, 6))

        years = seasonal_ndcg_df['Year'].unique()
        if len(years) == 1:
            sns.barplot(
                data=seasonal_ndcg_df,
                x='Season', y='NDCG',
                palette={'Winter': '#8FB8CF',
                         'Spring': '#98B486',
                         'Summer': '#d4a368',
                         'Fall': '#B07D62'},
                hue='Season'
            )
            plt.title(f'Seasonal NDCG for {years[0]}')
            y_min = max(0, min(seasonal_ndcg_df['NDCG']) - 0.1)
        else:
            sns.boxplot(
                data=seasonal_ndcg_df,
                x='Season', y='NDCG',
                palette={'Winter': '#8FB8CF',
                         'Spring': '#98B486',
                         'Summer': '#d4a368',
                         'Fall': '#B07D62'},
                hue='Season',
                whis=(0.0, 100.0)
            )
            plt.title(f'Seasonal NDCG Distribution')
            y_min = max(0, min(seasonal_ndcg_df['NDCG']) - 0.01)

        plt.ylim(y_min, 1.0)
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
        sns.barplot(
            data=importance_df.head(top_n),
            x='Importance', y='Feature'
        )
        plt.title(f'Top {top_n} Features by Importance')
        plt.show()

    @staticmethod
    def plot_cumulative_score_performance(cumulative_df: pd.DataFrame):
        """
        Plots MAE and NDCG metrics across cumulative score thresholds,
        comparing model NDCG against a random baseline.

        :param pd.DataFrame cumulative_df: ``DataFrame`` as returned by
            ``Evaluator.get_cumulative_score_metrics()``, with columns
            ``'Threshold'``, ``'MAE'``, ``'NDCG'``, ``'Random NDCG'``,
            and ``'Count'``.
        """
        fig, ax_ndcg = plt.subplots(figsize=(12, 6))

        sns.lineplot(
            data=cumulative_df, x='Threshold', y='NDCG', ax=ax_ndcg,
            color='#8FB8CF', marker='s', label='Model NDCG',
            markeredgewidth=0, markeredgecolor='none'
        )
        sns.lineplot(
            data=cumulative_df, x='Threshold', y='Random NDCG', ax=ax_ndcg,
            color='#8FB8CF', linestyle='--', alpha=0.5, label='Random Baseline'
        )
        ax_ndcg.set_ylabel('Average Seasonal NDCG')

        all_ndcg_vals = pd.concat([cumulative_df['NDCG'], cumulative_df['Random NDCG']])
        y_min = max(0, min(all_ndcg_vals.dropna()) - 0.01)
        ax_ndcg.set_ylim(y_min, 1.0)

        ax_mae = ax_ndcg.twinx()
        sns.lineplot(
            data=cumulative_df, x='Threshold', y='MAE', ax=ax_mae,
            color='#d4a368', marker='o', label='MAE',
            markeredgewidth=0, markeredgecolor='none'
        )
        ax_mae.set_ylabel('MAE')
        ax_mae.set_xlabel('Score Threshold')

        plt.title('Model Performance based on Anime Quality (Cumulative)')

        lines1, labels1 = ax_mae.get_legend_handles_labels()
        lines2, labels2 = ax_ndcg.get_legend_handles_labels()
        ax_mae.get_legend().remove()
        ax_ndcg.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        fig.tight_layout()
        plt.show()
