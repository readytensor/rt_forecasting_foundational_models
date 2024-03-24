import pandas as pd
import matplotlib.pyplot as plt
import sys


def read_docker_image_data(project):
    file_path = f"./../inputs/{project}_docker_image_sizes.csv"
    df = pd.read_csv(file_path)
    return df


def create_and_save_docker_image_chart(data, project_name):
    """
    Create a bar chart showing the sizes of Docker images for a given project.

    Parameters:
    - df: The pandas DataFrame containing the Docker image size data.
    - project: A string representing the name of the project.

    Returns:
    - A matplotlib figure object representing the bar chart.
    """
    # Reverse the DataFrame to display the first model at the top of the chart
    data_reversed = data.iloc[::-1]

    # Create a bar chart of the Docker image sizes
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        data_reversed['model_name'],
        data_reversed['docker_image_size_gb'], color='#1db1c1')
    plt.xlabel('Image Size (GB)')
    # plt.ylabel('Docker Image')
    plt.title('Model Image Sizes', fontsize=16, pad=10)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Find the maximum value for setting the x-axis range
    max_value = data_reversed['docker_image_size_gb'].max()
    margin = max_value * 0.1  # Add 10% of the max value as margin
    
    # Set the x-axis limits to include some extra space for the annotation
    plt.xlim(0, max_value + margin)

    
    # Annotate the end of each bar with its value
    for bar in bars:
        width = bar.get_width() 
        plt.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            fontsize=11,
            va='center')
    
    # Increase the font size of the y-axis ticks
    plt.tick_params(axis='y', labelsize=10)  # Adjust labelsize as needed
    
    # Adjust layout
    plt.tight_layout(pad=1.5)
    
    # Save the figure
    plt.savefig(f"./../outputs/{project_name}_docker_image_sizes.png")
    plt.show()


def create_docker_image_chart(project_name):
    # Read the Docker image data
    data = read_docker_image_data(project_name)

    # Create and save the Docker image chart
    create_and_save_docker_image_chart(data, project_name)


if __name__ == "__main__":
    project_name = "chronos"
    create_docker_image_chart(project_name)