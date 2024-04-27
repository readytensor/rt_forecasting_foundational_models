import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def read_files(project):
    metrics_df = pd.read_csv(
        f'./../inputs/{project}/{project}_metrics.csv'
    )
    models = pd.read_csv(
        f'./../inputs/{project}/{project}-forecasting-models.csv',
        encoding="ISO-8859-1"
    )
    datasets = pd.read_csv(
        f'./../inputs/{project}/{project}-forecasting-benchmark-datasets.csv'
    )
    return metrics_df, models, datasets


def filter_and_save_metrics(project):
    metrics_df, models, datasets = read_files(project)
    # Filter out models that are not used
    models = models[models['use?'] == 1]
    # Filter out datasets that are not used
    datasets = datasets[datasets['use?'] == 1].copy()
    # Filter the metrics dataframe
    metrics_df = metrics_df[metrics_df['model_name'].isin(models['model_name'])]
    metrics_df = metrics_df[metrics_df['dataset_name'].isin(datasets['dataset_name'])]
    metrics_df.drop(columns=["benchmark_external_id"], inplace=True)

    
    agg_scores = pd.read_csv(
        f"./../outputs/{project}/{project}-aggregate-scores.csv",
        encoding="ISO-8859-1"
    )
    model_order = agg_scores['model_name'].tolist()
    metrics_df.set_index('model_name', inplace=True)
    # Step 3: Reindex the filtered_metrics_df according to the model_order
    metrics_df = metrics_df.reindex(model_order)

    # Step 4: Reset the index if desired
    metrics_df.reset_index(inplace=True)

    metrics_df.rename(
        columns={
            "train_execution_time_seconds": "train_time_seconds",
            "predict_execution_time_seconds": "predict_time_seconds",
        },
        inplace=True
    )

    metrics_df.to_csv(
        f"./../outputs/{project}/{project}_filtered_metrics.csv",
        index=False,
        encoding="ISO-8859-1")


def create_heatmap(project):
    perf_metrics = pd.read_csv(
        f"./../outputs/{project}/{project}_filtered_metrics.csv",
        encoding="ISO-8859-1"
    )
    perf_metrics.drop(columns=["dataset_name"], inplace=True)
    # Assuming 'sorted_filtered_metrics_df' is your sorted dataframe
    data_to_display = perf_metrics.set_index('model_name')
    print(data_to_display)
    
    # Convert the dataframe values to strings with commas for thousands
    formatted_values = data_to_display.applymap(lambda x: f"{x:,.1f}")

    # Create a mask of the same shape as your dataframe, but with all False values
    # since you want to show all numbers in the heatmap
    mask = np.zeros_like(data_to_display, dtype=bool)
        
    # Define a colormap that is white regardless of the value
    white_cmap = mcolors.ListedColormap(['white'])

    # Generate the heatmap using white for all the values,
    # but the text will show the dataframe's values
    fig = plt.figure(figsize=(10, 10))  # Adjust as needed
    ax = sns.heatmap(
        data_to_display,
        mask=mask,
        annot=formatted_values.values,
        fmt="",
        cmap=white_cmap,
        cbar=False,
        linewidths=0.5,
        linecolor='#1db1c1',
        annot_kws={"size": 11}
    )
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 10)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 10)


    # Using suptitle for the main title above everything
    fig.suptitle('Execution Times and Memory Usage', fontsize=16, y=0.99)

    # Using title for what you're calling the subtitle, which will now appear just above the heatmap
    ax.text(0.25, 1.19, "Air Quality-2018 Dataset", ha='center', va='bottom', 
        transform=ax.transAxes, fontsize=14)

    # Move the x-axis labels to the top
    ax.xaxis.tick_top()  # This moves the x-axis labels to the top
    ax.xaxis.set_label_position('top')  # This moves the x-axis title to the top
    plt.xticks(rotation=45, ha="left")

    # Adjust layout to add more space on the right
    plt.subplots_adjust(right=0.5)  # Adjust the right margin to add whitespace

    # Manually add lines to the right and bottom edges
    num_columns = len(data_to_display.columns)
    num_rows = len(data_to_display.index)
    # ax.hlines(y=np.arange(num_rows+1), xmin=0, xmax=num_columns, color='#1db1c1', linewidth=1.0)
    # ax.vlines(x=np.arange(num_columns+1), ymin=0, ymax=num_rows, color='#1db1c1', linewidth=1.0)
    ax.hlines(y=[num_rows], xmin=0, xmax=num_columns, color='#1db1c1', linewidth=2.0)
    ax.vlines(x=[num_columns], ymin=0, ymax=num_rows, color='#1db1c1', linewidth=2.0)

    # Remove the y-axis label
    plt.ylabel('')

    plt.tight_layout()
    plt.savefig(
        f'./../outputs/{project}/model_exec_durations_and_memory.png',
        dpi=300
    )



def create_durations_and_memory_chart(project):
    filter_and_save_metrics(project)
    create_heatmap(project)


if __name__ == "__main__":
    project = "moirai"
    create_durations_and_memory_chart(project)