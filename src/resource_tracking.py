import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def read_files():
    metrics_df = pd.read_csv('./../inputs/chronos_project_metrics.csv')
    models = pd.read_csv('./../inputs/forecasting-models.csv', encoding="ISO-8859-1")
    datasets = pd.read_csv('./../inputs/forecasting-benchmark-datasets.csv')
    return metrics_df, models, datasets


def filter_and_save_metrics():
    metrics_df, models, datasets = read_files()
    # Filter out models that are not used
    models = models[models['use?'] == 1]
    # Filter out datasets that are not used
    datasets = datasets[datasets['use?'] == 1]
    # Filter the metrics dataframe
    metrics_df = metrics_df[metrics_df['model_name'].isin(models['model_name'])]
    metrics_df = metrics_df[metrics_df['dataset_name'].isin(datasets['dataset_name'])]
    metrics_df.drop(columns=["benchmark_name", "benchmark_external_id"], inplace=True)

    
    agg_scores = pd.read_csv("./../outputs/aggregate-scores.csv", encoding="ISO-8859-1")
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
        "./../outputs/filtered_metrics.csv",
        index=False,
        encoding="ISO-8859-1")


def get_perf_metrics():
    agg_scores = pd.read_csv("./../outputs/aggregate-scores.csv", encoding="ISO-8859-1")
    perf_metrics = pd.read_csv("./../outputs/filtered_metrics.csv", encoding="ISO-8859-1")
    model_order = agg_scores['model_name'].tolist()
    perf_metrics.set_index('model_name', inplace=True)
    
    # Step 3: Reindex the filtered_metrics_df according to the model_order
    perf_metrics = perf_metrics.reindex(model_order)

    # Step 4: Reset the index if desired
    perf_metrics.reset_index(inplace=True)

    # print(agg_scores.head())
    # print(perf_metrics.head())
    return perf_metrics


def create_heatmap():
    perf_metrics = pd.read_csv("./../outputs/filtered_metrics.csv", encoding="ISO-8859-1")
    perf_metrics.drop(columns=["dataset_name"], inplace=True)
    # Assuming 'sorted_filtered_metrics_df' is your sorted dataframe
    data_to_display = perf_metrics.set_index('model_name')
    
    # Convert the dataframe values to strings with commas for thousands
    formatted_values = data_to_display.map(lambda x: f"{x:,.1f}")

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
    plt.savefig('./../outputs/model_exec_durations_and_memory.png')


# def create_perf_chart(perf_metrics):
    
#     # Prepare the Data
#     # Assuming 'sorted_filtered_metrics_df' is your sorted dataframe and you want to
#     # display all its columns
#     data_to_display = perf_metrics
#     width, height = 18, 7

#     # Create the Figure and Axes
#     fig, ax = plt.subplots(figsize=(width, height))  # Adjust width and height as needed

#     # Remove Axes
#     ax.axis('off')

#     # Add the Table
#     # The 'cellText' argument takes a 2D list of table values, which you can get from
#     # the dataframe values
#     # 'colLabels' argument takes the column names for the header
#     table = ax.table(
#         cellText=data_to_display.values, colLabels=data_to_display.columns,
#         loc='center', cellLoc='center')

#     # Optionally adjust the layout
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)  # Adjust font size as needed
#     # table.auto_set_column_width(col=list(range(len(data_to_display.columns))))  # Adjust column widths

#     # Set custom widths for each column by column index (e.g., 0, 1, 2, ...)
#     # column_widths = {0: 0.1, 1: 0.2, 2: 0.15}  # Example widths for the first three columns

#     # for key, cell in table.get_celld().items():
#     #     col_index = key[1]  # Column index
#     #     if col_index in column_widths:
#     #         cell.set_width(column_widths[col_index])

#     # Adjust layout
#     plt.tight_layout()

#     # Save the Figure
#     plt.savefig("./../outputs/performance_metrics_table.png")

def create_durations_and_memory_chart():
    filter_and_save_metrics()
    create_heatmap()


if __name__ == "__main__":
    create_durations_and_memory_chart