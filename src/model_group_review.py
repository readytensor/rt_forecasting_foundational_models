import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chardet
import sys
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

from aggregate_results import read_files
from vars import (
    INPUT_DIR,
    OUTPUT_DIR,
    SCORES_FNAME,
    DATASET_FNAME,
    MODELS_FNAME,
    AGGREGATED_SCORES_FNAME,
    METRIC,
    METRIC_DISPLAY_NAME
)


def extract_model_group_scores(scores, datasets, models):
    model_scores = scores[scores["metric_external_id"] == METRIC].copy()
    print(model_scores.head())
    # filtered["final_metric_value"] = -1 * filtered["final_metric_value"]
    model_scores.loc[:, "final_metric_value"] = -1 * model_scores["final_metric_value"]

    model_scores = model_scores.drop(
        ['model_external_id', 'model_category_id', 'metric_external_id', 
         'dataset_id', 'preview_metric_value'], axis=1)

    model_names = [list(m.keys())[0] for m in models]
    names_to_display_names = {list(m.keys())[0]: list(m.values())[0] for m in models}
    ordered_display_names = [list(m.values())[0] for m in models]
    model_scores = model_scores[model_scores['model_name'].isin(model_names)]
    model_scores['model_name'] = model_scores['model_name'].map(names_to_display_names)

    model_scores.rename(columns={
        "dataset_external_id": "dataset",
        "final_metric_value": METRIC_DISPLAY_NAME,
        }, inplace=True)

    pivoted_models_scores = model_scores.pivot_table(
        index=['dataset'],
        columns='model_name',
        values=METRIC_DISPLAY_NAME,
        aggfunc='first').reset_index()
    
    pivoted_models_scores = pivoted_models_scores[
        ["dataset"] + ordered_display_names
    ]
    return pivoted_models_scores



def identify_and_plot_hallucinations(models_scores):
    model_columns = models_scores.columns[1:]  # Columns with model names
    diff_df = models_scores[model_columns].pct_change(axis='columns') * 100
    hallucination_mask = diff_df > 35  # True for hallucinations
    
    # Create a custom color map: yellow for hallucinations, white for others
    cmap = sns.color_palette(["white", "yellow"], as_cmap=True)
    white_cmap = mcolors.ListedColormap(['white'])

    data_to_display = models_scores[model_columns]
    # Convert the dataframe values to strings with commas for thousands
    formatted_values = data_to_display.map(lambda x: f"{x:,.2f}")
    
    # Plotting
    plt.figure(figsize=(12, 10))
    # Plot the base heatmap with white color
    ax = sns.heatmap(
        models_scores[model_columns],
        annot=True,
        fmt=".2f",
        cmap=white_cmap,
        cbar=False,
        linewidths=0.5,
        linecolor='#1db1c1',
        annot_kws={"size": 11}
    )

    # Overlay with hallucination highlights, masking non-hallucinations
    sns.heatmap(
        models_scores[model_columns],
        mask=~hallucination_mask,
        cmap=mcolors.ListedColormap(['yellow']),
        cbar=False,
        annot=False,
        linewidths=0.5,
        linecolor='#1db1c1',
    )

    plt.xticks(np.arange(len(model_columns)) + 0.5, model_columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(models_scores)) + 0.5, models_scores.iloc[:, 0], rotation=0)
    plt.title('RMSSE Scores by Dataset for Chronos Models', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Datasets', fontsize=14)

    ax = plt.gca()

    # Add a rectangle around the heatmap
    num_datasets = len(models_scores)
    num_models = len(model_columns)
    ax.add_patch(Rectangle((0, 0), num_models, num_datasets, fill=False, edgecolor='#1db1c1', lw=0.5, clip_on=False))

    plt.tight_layout(pad=1.5)

    # Save the plot
    chart_fpath = os.path.join(OUTPUT_DIR, f"{model_group_name}_model_scores_by_dataset.png")
    plt.savefig(chart_fpath, dpi=300)
    plt.show()



def extract_and_save_model_group_scores(model_group_name, models):
    scores, datasets, _ = read_files()
    model_group_scores = extract_model_group_scores(scores, datasets, models)
    fname = f"{model_group_name}_model_scores_by_dataset.csv"
    # model_group_scores.to_csv(
    #     os.path.join(OUTPUT_DIR, fname),
    #     encoding="ISO-8859-1",
    #     index=False,
    #     float_format='%.2f'
    #     )

    identify_and_plot_hallucinations(model_group_scores)


if __name__ == "__main__":
    model_group_name = "chronos"
    models = [
        {"Amazon Chronos-T5-Tiny Forecasting Model": "chronos-t5-tiny"},
        {"Amazon Chronos-T5-Mini Forecasting Model": "chronos-t5-mini"},
        {"Amazon Chronos-T5-Small Forecasting Model": "chronos-t5-small"},
        {"Amazon Chronos-T5-Base Forecasting Model": "chronos-t5-base"},
        {"Amazon Chronos-T5-Large Forecasting Model": "chronos-t5-large"},
    ]
    extract_and_save_model_group_scores(model_group_name, models)