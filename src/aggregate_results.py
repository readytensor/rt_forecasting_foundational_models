import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chardet
import sys

INPUT_DIR = "./../inputs"
OUTPUT_DIR = "./../outputs"
SCORES_FNAME = "forecasting-model-scores.csv"
DATASET_FNAME = "forecasting-benchmark-datasets.csv"
MODELS_FNAME = "forecasting-models.csv"
METRIC = "rmsse_fcst"


AGGREGATED_SCORES_FNAME = "aggregate-scores.csv"

def read_files():
    scores = pd.read_csv(os.path.join(INPUT_DIR, SCORES_FNAME))
    datasets = pd.read_csv(os.path.join(INPUT_DIR, DATASET_FNAME))
    models = pd.read_csv(os.path.join(INPUT_DIR, MODELS_FNAME), encoding="ISO-8859-1")
    return scores, datasets, models

def aggregate_scores(scores, datasets, models):
    filtered = scores[scores["metric_external_id"] == METRIC]
    # filtered["final_metric_value"] = -1 * filtered["final_metric_value"]
    filtered.loc[:, "final_metric_value"] = -1 * filtered["final_metric_value"]

    average_rmsse_per_model = filtered.groupby('model_name')['final_metric_value']\
        .mean().reset_index().rename(columns={'final_metric_value': 'overall'})


    cleaned = filtered.drop(
        ['model_external_id', 'model_category_id', 'metric_external_id', 
         'dataset_id', 'preview_metric_value'], axis=1)
    # print(scores.shape, filtered.shape)

    merged = cleaned.merge(datasets, on='dataset_external_id', how='left')
    aggregated_data = merged.groupby(['model_name', 'frequency'])\
        .agg(rmsse_fcst=('final_metric_value', 'mean')).reset_index()


    filtered2 = aggregated_data.merge(models, on='model_name', how='inner')
    used_models_data = filtered2[filtered2['use?'] == 1].drop(columns=['use?']).reset_index()

    # Pivot the table to have the frequency as columns and the rmsse values as the values in those columns
    pivoted_data = used_models_data.pivot_table(index=['model_name', 'category'],
                                                columns='frequency',
                                                values='rmsse_fcst',
                                                aggfunc='first').reset_index()

    pivoted_data_with_average = pivoted_data.merge(average_rmsse_per_model, on='model_name', how='left')

    category_order = [
        'naïve (baseline)', 'statistical', 'statistical / machine-learning',
        'machine-learning', 'neural network', 'foundational model']

    pivoted_data_with_average['category'] = pd.Categorical(
        pivoted_data_with_average['category'], categories=category_order, ordered=True)

    # Sort the dataframe first by category in the specified order, then within each category
    # by the overall rmsse score in descending order
    final_data = pivoted_data_with_average.sort_values(by=['category', 'overall'], ascending=[True, False])


    final_data = final_data[[
        'model_name', 'category', 
        'hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'other', 'overall']]
    return final_data

def aggregate_and_save_scores():
    scores, datasets, models = read_files()
    aggregated_scores = aggregate_scores(scores, datasets, models)
    aggregated_scores.to_csv(
        os.path.join(OUTPUT_DIR, AGGREGATED_SCORES_FNAME),
        encoding="ISO-8859-1",
        index=False,
        float_format='%.2f'
        )


if __name__ == "__main__":
    aggregate_and_save_scores()