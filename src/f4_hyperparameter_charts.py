import pandas as pd
import matplotlib.pyplot as plt
import sys


def read_experiments_data(project):
    file_path = f"./../inputs/{project}/{project}-hyperparameter-experiments.csv"
    df = pd.read_csv(file_path)
    return df


def create_hyperparameter_scenarios_dict(df, hyperparameters, scenario_col):
    # Initialize the dictionary with the base scenario for each hyperparameter
    hp_scenarios = {hp: ["base_scenario"] for hp in hyperparameters}
    # Iterate over each scenario in the dataframe
    for scenario in df[scenario_col].unique():
        if scenario == "base_scenario":
            continue
        for hp in hyperparameters:
            if hp in scenario:
                hp_scenarios[hp].append(scenario)
                break
    return hp_scenarios

def filter_data(df, column_name, filter_values):
    """
    Filter the dataframe based on a column name and a list of filter values.

    Parameters:
    - df: The pandas DataFrame containing the data.
    - column_name: A string representing the column name to filter by.
    - filter_values: A list of strings representing the values to filter for in the specified column.

    Returns:
    - A filtered pandas DataFrame containing only the rows where the specified column matches one of the filter values.
    """
    # Use the DataFrame's `isin` method to filter rows where the column value is in the list of filter values
    filtered_df = df[df[column_name].isin(filter_values)]
    return filtered_df


def average_metric_by_scenario(data, hyperparameter, hp_spec, scenario_values, scenario_col, metric_column):
    """
    Calculate the average metric value grouped by scenario values for a specific hyperparameter.

    Parameters:
    - data: The pandas DataFrame containing the data.
    - hyperparameter: A string representing the hyperparameter column name to filter by scenario values.
    - hp_spec: Dictionary indicating numeric or categorical type of hyperparameter. If categorical, 
               Also specifies the order of the hyperparameter values.
    - scenario_values: A list of strings representing the scenario values to filter for in the specified hyperparameter.
    - metric_column: A string representing the name of the metric column to calculate the average for.

    Returns:
    - A pandas DataFrame with hyperparameter values as the index and the average metric value for each scenario as columns.
    """
    # Filter the data for the given scenario values
    filtered_data = filter_data(data, scenario_col, scenario_values)
    
    # Group by the hyperparameter and calculate the average of the metric column
    avg_metric_df = filtered_data.groupby(hyperparameter)[metric_column].mean().reset_index()
    
    # Set the hyperparameter as the index
    avg_metric_df.set_index(hyperparameter, inplace=True)

    if hp_spec['type'] in ["int", "float"]:
        avg_metric_df.sort_index()
    elif hp_spec['type'] == 'categorical':
        avg_metric_df = avg_metric_df.reindex(hp_spec['categories'])
    return avg_metric_df



def generate_hyperparam_results(
        project_name, scenario_col, model_name,
        hyperparameters, metric_col, metric_name
    ):
    # read experiments CSV data
    data = read_experiments_data(project_name)
    
    # filter to a specific model
    data = filter_data(data, "model", [model_name])
    
    # generate scenarios dict
    hp_scenarios = create_hyperparameter_scenarios_dict(
        data, hyperparameters, scenario_col
    )
    hp_scenario_results = {}
    for hp, scenarios in hp_scenarios.items():
        avg_metric_df = average_metric_by_scenario(
            data, hp, hyperparameters[hp],
            scenarios, scenario_col, metric_col)
        avg_metric_df = avg_metric_df.round(2)
        hp_scenario_results[hp] = avg_metric_df
        print(avg_metric_df)
    
    return hp_scenario_results


def plot_hyperparameter_impacts(
        model_name, hp_scenario_results, hyperparameters,
        metric_col, metric_name, y_axis_range=None):
    """
    Plots the hyperparameter impact results using subplots, with an optional uniform y-axis range.
    
    Parameters:
    - hp_scenario_results: Dictionary with hyperparameters as keys and DataFrame results as values.
    - hyperparameters: Dictionary specifying the type and possibly categories of the hyperparameters.
    - metric_name: String representing the name of the metric to display in chart titles.
    - y_axis_range: Tuple (min, max) representing the range of the y-axis. If None, the range is calculated dynamically.
    """
    num_rows = len(hp_scenario_results) // 2 + len(hp_scenario_results) % 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))
    axs = axs.flatten()

    # Determine dynamic y-axis range if not provided
    if y_axis_range is None:
        all_values = pd.concat([df[metric_col] for df in hp_scenario_results.values()])
        min_val, max_val = all_values.min(), all_values.max()
        # Add a buffer to the max_val for annotations
        buffer = (max_val - min_val) * 0.05  # Adjust buffer size as needed
        y_axis_range = (min_val, max_val + buffer)
    
    
    for i, (hp, results_df) in enumerate(hp_scenario_results.items()):
        positions = range(len(results_df.index))
        axs[i].bar(positions, results_df[metric_col], color='#1db1c1')

        # Set the x-axis ticks to correspond to the category positions and labels
        axs[i].set_xticks(positions)
        axs[i].set_xticklabels(results_df.index, rotation=45, ha="right")
        
        # Annotate each bar with its value
        for x, y in zip(positions, results_df[metric_col]):
            axs[i].text(x, y, f'{y:.2f}', color='black', ha='center', va='bottom')
        
        axs[i].set_title(f"Impact of {hp}", fontsize=14)
        axs[i].set_xlabel(hp, fontsize=12)
        if i % 2 == 0:
            axs[i].set_ylabel(metric_name)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_ylim(y_axis_range)

            
    plt.suptitle(f"Hyperparameters Impact Analysis for `{model_name}`",
                 fontsize=16, y=0.98)    

    # Adjust layout
    plt.tight_layout(pad=1.5)
    
    # Save the figure
    plt.savefig(
        f"./../outputs/{project_name}/{project_name}_hyperparameter_impacts.png",
        dpi=300
    )
    plt.show()


def create_and_save_hp_impacts_chart(
        project_name, scenario_col, model_name, hyperparameters,
        metric_col, metric_name, y_range=None,
    ):
    hp_scenario_results = generate_hyperparam_results(
        project_name, scenario_col, model_name, hyperparameters,
        metric_col, metric_name
    )
    plot_hyperparameter_impacts(
        model_name,
        hp_scenario_results, hyperparameters, metric_col, metric_name, y_range
    )

if __name__ == "__main__":
    # project name
    # project_name = "chronos"
    project_name = "moirai"

    # name of the model to analyze
    model_name = {
        "moirai": "Moirai-large",
        "chronos": "chronos_t5_large",
    }
    # name of the column in data representing scenario name
    scenario_col = "scenario"
    # hyperparameters for the model
    hyperparameters = {
        "chronos": {
            "num_samples": {
                "type": "int"
            },
            "top_p": {
                "type": "float"
            },
            "top_k": {
                "type": "float"
            },
            "temperature": {
                "type": "float"
            },
        },
        "moirai": {
            "num_samples": {
                "type": "int"
            },
            "context_length": {
                "type": "int"
            },
        }
    }
    # metric column name in the data
    metric_col = "Root Mean Squared Scaled Error"
    # metric display name in charts
    metric_name = "RMSSE"

    y_range = [0.5, 1.0]
    create_and_save_hp_impacts_chart(
        project_name, scenario_col, model_name[project_name],
        hyperparameters[project_name],
        metric_col, metric_name, y_range,
    )

    