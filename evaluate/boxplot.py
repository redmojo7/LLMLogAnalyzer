import os
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


def create_boxplot(ax, values, positions, widths):
    ax.boxplot([values], positions=positions, widths=widths)

def plot_category_metrics(dfs: list[pd.DataFrame], category: str) -> None:
    """Plot box plots for accuracy and ROUGE-1 F1."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    models = [model.value for model in Model]
    metrics = [metric.value for metric in Metric]
    
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            column_name = f'{model} {metric}'
            values = np.concatenate([df[column_name].values for df in dfs])
            create_boxplot(axs[j], values, [i], 0.5)

    axs[0].set_title(f'{category} - Cosine Similarity')
    axs[1].set_title(f'{category} - ROUGE-1 F1')
    axs[0].set_ylim(0, 1)

    axs[0].set_ylabel('Cosine Similarity')
    axs[1].set_ylabel('ROUGE-1 F1')
    axs[1].set_ylim(0, 1)

    for ax in axs:
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)

    #fig.suptitle(f'{category} Metrics')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create a folder to save the boxplot images
    output_dir = 'evaluate/outputs/boxplots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f'Saving boxplot for {category}...')
    # lowercase the category and repace space with underscore 
    replaced_category = category.lower().replace(' ', '_')
    fig.savefig(f'{output_dir}/{list(Category).index(category)+1}_{replaced_category}_boxplot.png')


def main() -> None:
    dfs = [load_data(file_path) for file_path in CSV_FILES]

    for category in Category:
        category_dfs = [filter_data(df, category) for df in dfs]
        plot_category_metrics(category_dfs, category.value)


if __name__ == "__main__":
    main()