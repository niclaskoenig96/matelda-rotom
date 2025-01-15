import os
import pickle
import matplotlib.pyplot as plt

def plot_metrics_grouped_by_metric_from_combined_pickle(combined_pickle_files, output_filename='grouped_metrics_plot.png'):
    """
    Generates and saves a plot where recall, precision, and f-score are grouped, with one line per implementation.

    Args:
        combined_pickle_files (list of str): List of paths to combined pickle files for each implementation.
        output_filename (str): The name of the file to save the plot.
    """
    # Load data from pickle files
    implementations_scores = []
    implementation_labels = []

    for pickle_file in combined_pickle_files:
        with open(pickle_file, 'rb') as f:
            scores_list = pickle.load(f)
            implementations_scores.append(scores_list)

        # Extract label from the filename
        base_name = os.path.basename(pickle_file)
        label = base_name.split('_combined')[0]
        implementation_labels.append(label)

    # Prepare data for plotting
    metric_names = ['Recall', 'Precision', 'F-score']
    metric_keys = ['total_recall', 'total_precision', 'total_fscore']

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[idx]
        for scores_list, label in zip(implementations_scores, implementation_labels):
            n_samples = [scores['n_samples'] for scores in scores_list]
            metric_values = [scores[metric_key] for scores in scores_list]
            ax.plot(n_samples, metric_values, label=label, marker='o')

        ax.set_title(metric_name, fontsize=14)
        ax.set_xlabel('Number of Samples (n_samples)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Value (0-1)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

# Example usage
if __name__ == "__main__":
    base_directory = "eval/combined"
    combined_pickle_files = [
        os.path.join(base_directory, file_name)
        for file_name in os.listdir(base_directory)
        if file_name.endswith('_combined.pkl')
    ]

    plot_metrics_grouped_by_metric_from_combined_pickle(combined_pickle_files)
