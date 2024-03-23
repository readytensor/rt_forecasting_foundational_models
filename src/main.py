from aggregate_results import aggregate_and_save_scores
from create_heatmap_chart import create_and_save_heatmap
from resource_tracking import create_durations_and_memory_chart


def run_analysis():
    aggregate_and_save_scores()
    create_and_save_heatmap()
    create_durations_and_memory_chart()



if __name__ == "__main__":
    run_analysis()