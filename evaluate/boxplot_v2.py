import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Model(str, Enum):
    CHATGPT = 'ChatGPT(GPT-4o)'
    NOTEBOOKLM = 'NotebookLM'
    CHATPDF = 'ChatPDF'
    LLMLOGANALYZER_LLMAMODEL3_70B = 'LLMLogAnalyzer(Llama-3-70B)'
    LLMLOGANALYZER_LLMAMODEL3_8B = 'LLMLogAnalyzer(Llama-3-8B)'


class Metric(str, Enum):
    ACCURACY = 'Cosine Similarity'
    ROUGE1_F1 = 'ROUGE-1 F1'


class Category(str, Enum):
    SUMMARIZATION = 'Summarization'
    PATTERN_EXTRACTION = 'Pattern Extraction'
    ANOMALY_DETECTION = 'Anomaly Detection'
    ROOT_CAUSE_ANALYSIS = 'Root Cause Analysis'
    PREDICTIVE_FAILURE_ANALYSIS = 'Predictive Failure Analysis'
    LOG_UNDERSTANDING_INTERPRETATION = 'Log Understanding and Interpretation'
    LOG_FILTERING_SEARCH = 'Log Filtering and Search'


# Define CSV file paths
CSV_FILES = [
    'evaluate/outputs/Apache/apache_metrics.csv',
    'evaluate/outputs/Linux/linux_metrics.csv',
    'evaluate/outputs/Mac/mac_metrics.csv',
    'evaluate/outputs/Windows/windows_metrics.csv'
]


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(file_path)


def filter_data(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filter dataframe by category."""
    return df[df['Category'] == category]


def create_boxplot(ax, values, positions, widths, labels):
    ax.boxplot(values, positions=positions, widths=widths)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)


def plot_combined_metrics(dfs: list[pd.DataFrame]) -> None:
    """Plot box plots for accuracy and ROUGE-1 F1 for all categories combined."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    models = [model.value for model in Model]
    metrics = [metric.value for metric in Metric]

    for j, metric in enumerate(metrics):
        values = []
        for model in models:
            column_name = f'{model} {metric}'
            values_model = np.concatenate([df[column_name].values for df in dfs])
            values.append(values_model)
        
        create_boxplot(axs[j], values, range(len(models)), 0.45, models)

    #axs[0].set_title('Cosine Similarity')
    #axs[1].set_title('ROUGE-1 F1')
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)

    axs[0].set_ylabel('Cosine Similarity', fontsize=14)
    axs[1].set_ylabel('ROUGE-1 F1', fontsize=14)

    fig.tight_layout(rect=[0.01, 0.01, 0.98, 0.98],w_pad=5)

    # Create a folder to save the boxplot images
    output_dir = 'evaluate/outputs/boxplots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Saving boxplot...')
    fig.savefig(f'{output_dir}/boxplot.png')

def calculate_iqr(df: pd.DataFrame, column_name: str) -> float:
    """
    Calculate Interquartile Range (IQR) for a given column.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column to calculate IQR for.

    Returns:
    float: IQR value.
    """
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    return iqr


def calculate_outliers_percentage(df: pd.DataFrame, column_name: str) -> float:
    """
    Calculate percentage of outliers for a given column.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column to calculate outliers for.

    Returns:
    float: Percentage of outliers.
    """
    iqr = calculate_iqr(df, column_name)
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    outliers_percentage = (len(outliers) / len(df)) * 100
    return outliers_percentage


def calculate_median(df: pd.DataFrame, column_name: str) -> float:
    """
    Calculate median value for a given column.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column to calculate median for.

    Returns:
    float: Median value.
    """
    median = df[column_name].median()
    return median

def main() -> None:
    
    # Save all print into a file
    sys.stdout = open('evaluate/outputs/boxplot.txt', 'w')
    dfs = [load_data(file_path) for file_path in CSV_FILES]
    
    dfs = [load_data(file_path) for file_path in CSV_FILES]

    # Combine dataframes for all categories
    combined_df = pd.concat(dfs, ignore_index=True)

    plot_combined_metrics([combined_df])
    
    # Calculate average improvement in IQR ratio
    models = [model.value for model in Model]
    metrics = [metric.value for metric in Metric]
    average_narrowing = {}
    for metric in metrics:
        total_narrowing = 0
        count = 0
        for model in models:
            if model != 'LLMLogAnalyzer(Llama-3-70B)':
                column_name = f'{model} {metric}'
                iqr_ratio = calculate_iqr(combined_df, column_name) / calculate_iqr(combined_df, f'LLMLogAnalyzer(Llama-3-70B) {metric}')
                total_narrowing += (iqr_ratio - 1)  # positive if LLMLogAnalyzer(Llama-3-70B) is narrower
                count += 1
        average_narrowing[metric] = (total_narrowing / count) * 100

    # Print average narrowing
    print("Average Narrowing in IQR (LLMLogAnalyzer(Llama-3-70B)): ")
    for metric, narrowing in average_narrowing.items():
        print(f"{metric}: {abs(narrowing):.2f}%")

    # Calculate median values
    median_values = {}
    for metric in metrics:
        median_values[metric] = {}
        for model in models:
            column_name = f'{model} {metric}'
            median_values[metric][model] = calculate_median(combined_df, column_name)

    # Print median values
    print("Median Values:")
    for metric, model_medians in median_values.items():
        print(f"Metric: {metric}")
        for model, median in model_medians.items():
            print(f"Model: {model}, Median: {median:.4f}")
        print("-" * 50)
        

    # Calculate outliers
    outliers = {}
    for metric in metrics:
        outliers[metric] = {}
        for model in models:
            column_name = f'{model} {metric}'
            iqr = calculate_iqr(combined_df, column_name)
            q1 = combined_df[column_name].quantile(0.25)
            q3 = combined_df[column_name].quantile(0.75)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_values = combined_df[(combined_df[column_name] < lower_bound) | (combined_df[column_name] > upper_bound)][column_name].values
            outliers[metric][model] = outlier_values

    # Print outliers
    print("Outliers:")
    for metric, model_outliers in outliers.items():
        print(f"Metric: {metric}")
        for model, outlier_values in model_outliers.items():
            print(f"Model: {model}, Outliers: {outlier_values}")
        print("-" * 50)

    # Calculate percentage of outliers
    outlier_percentages = {}
    for metric in metrics:
        outlier_percentages[metric] = {}
        for model in models:
            column_name = f'{model} {metric}'
            iqr = calculate_iqr(combined_df, column_name)
            q1 = combined_df[column_name].quantile(0.25)
            q3 = combined_df[column_name].quantile(0.75)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_count = len(combined_df[(combined_df[column_name] < lower_bound) | (combined_df[column_name] > upper_bound)])
            outlier_percentage = (outliers_count / len(combined_df)) * 100
            outlier_percentages[metric][model] = outlier_percentage

    # Print percentage of outliers
    print("Percentage of Outliers:")
    for metric, model_outlier_percentages in outlier_percentages.items():
        print(f"Metric: {metric}")
        for model, outlier_percentage in model_outlier_percentages.items():
            print(f"Model: {model}, Outlier Percentage: {outlier_percentage:.2f}%")
        print("-" * 50)


if __name__ == "__main__":
    main()