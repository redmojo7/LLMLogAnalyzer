import sys
import pandas as pd
import numpy as np
from enum import Enum

class Model(str, Enum):
    CHATGPT = 'ChatGPT(GPT-4o)'
    NOTEBOOKLM = 'NotebookLM'
    CHATPDF = 'ChatPDF'
    LLMLOGANALYZER_LLMAMODEL3_70B = 'LLMLogAnalyzer(Llama-3-70B)' 
    LLMLOGANALYZER_LLMAMODEL3_8B = 'LLMLogAnalyzer(Llama-3-8B)'  # Exclude LLMLogAnalyzer (Llama-3-8B) for IQR calculation


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


def calculate_iqr(dfs: pd.DataFrame, model: str, metric: str) -> float:
    """Calculate IQR for a given model and metric."""
    column_name = f'{model} {metric}'
    values = [df[column_name].values for df in dfs]
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    return iqr


def print_category_metrics(dfs: list[pd.DataFrame], category: str) -> None:
    """Print IQR for accuracy and ROUGE-1 F1."""
    models = [model.value for model in Model]
    metrics = [metric.value for metric in Metric]
    baseline_model = Model.LLMLOGANALYZER_LLMAMODEL3_70B.value
    
    print(f"\n{category} Metrics:")
    
    iqr_differences = {}
    all_iqr_values = {}
    
    for metric in metrics:
        print(f"\n{metric}:")
        
        iqr_values = {}
        for model in models:
            iqr = calculate_iqr(dfs, model, metric)
            iqr_values[model] = iqr
            all_iqr_values[f"{category} - {metric} - {model}"] = iqr
            print(f"{model}: IQR = {iqr:.4f}")
        
        # Calculate average IQR difference
        for model, iqr in iqr_values.items():
            if model != baseline_model:
                iqr_difference = iqr_values[baseline_model] - iqr
                iqr_differences[f"{metric} - {model}"] = iqr_difference
        
# Calculate overall average IQR difference for each metric
    rouge1_f1_iqr_differences = [value for key, value in iqr_differences.items() if 'ROUGE-1 F1' in key]
    cosine_similarity_iqr_differences = [value for key, value in iqr_differences.items() if 'Cosine Similarity' in key]

    rouge1_f1_iqr_values = [value for key, value in all_iqr_values.items() if 'ROUGE-1 F1' in key]
    cosine_similarity_iqr_values = [value for key, value in all_iqr_values.items() if 'Cosine Similarity' in key]

    percentage_narrower_rouge1_f1 = (np.mean(rouge1_f1_iqr_differences) / np.mean(rouge1_f1_iqr_values)) * 100
    percentage_narrower_cosine_similarity = (np.mean(cosine_similarity_iqr_differences) / np.mean(cosine_similarity_iqr_values)) * 100

    if np.mean(rouge1_f1_iqr_differences) > 0:
        print(f"\nLLMLogAnalyzer(Llama-3-70B) exhibits robustness with a {abs(percentage_narrower_rouge1_f1):.2f}% wider interquartile range compared to other methods for ROUGE-1 F1.")
    else:
        print(f"\nLLMLogAnalyzer(Llama-3-70B) exhibits robustness with a {abs(percentage_narrower_rouge1_f1):.2f}% narrower interquartile range compared to other methods for ROUGE-1 F1.")

    if np.mean(cosine_similarity_iqr_differences) > 0:
        print(f"\nLLMLogAnalyzer(Llama-3-70B) exhibits robustness with a {abs(percentage_narrower_cosine_similarity):.2f}% wider interquartile range compared to other methods for Cosine Similarity.")
    else:
        print(f"\nLLMLogAnalyzer(Llama-3-70B) exhibits robustness with a {abs(percentage_narrower_cosine_similarity):.2f}% narrower interquartile range compared to other methods for Cosine Similarity.")

    
def calculate_final_robustness(dfs: list[pd.DataFrame]) -> None:
    all_iqr_differences = {}
    all_iqr_values = {}
    
    for category in Category:
        category_dfs = [filter_data(df, category.value) for df in dfs]
        models = [model.value for model in Model]
        metrics = [metric.value for metric in Metric]
        baseline_model = Model.LLMLOGANALYZER_LLMAMODEL3_70B.value
        
        for metric in metrics:
            iqr_values = {}
            for model in models:
                iqr = calculate_iqr(category_dfs, model, metric)
                iqr_values[model] = iqr
                all_iqr_values[f"{category.value} - {metric} - {model}"] = iqr
            
            # Calculate average IQR difference
            for model, iqr in iqr_values.items():
                if model != baseline_model:
                    iqr_difference = iqr_values[baseline_model] - iqr
                    all_iqr_differences[f"{category.value} - {metric} - {model}"] = iqr_difference
    
    # Calculate final robustness for each criterion
    rouge1_f1_iqr_differences = [value for key, value in all_iqr_differences.items() if 'ROUGE-1 F1' in key]
    cosine_similarity_iqr_differences = [value for key, value in all_iqr_differences.items() if 'Cosine Similarity' in key]
    
    rouge1_f1_baseline_iqr = np.mean([value for key, value in all_iqr_values.items() if 'ROUGE-1 F1' in key and 'LLMLogAnalyzer(Llama-3-70B)' in key])
    cosine_similarity_baseline_iqr = np.mean([value for key, value in all_iqr_values.items() if 'Cosine Similarity' in key and 'LLMLogAnalyzer(Llama-3-70B)' in key])
    
    final_robustness_rouge1_f1 = abs(np.mean(rouge1_f1_iqr_differences) / rouge1_f1_baseline_iqr) * 100
    final_robustness_cosine_similarity = abs(np.mean(cosine_similarity_iqr_differences) / cosine_similarity_baseline_iqr) * 100
    
    print("\nFinal Robustness:\n")
    if np.mean(rouge1_f1_iqr_differences) > 0:
        print(f"Final Robustness (ROUGE-1 F1):\nLLMLogAnalyzer(Llama-3-70B) exhibits robustness with a {final_robustness_rouge1_f1:.2f}% wider interquartile range compared to other methods.")
    else:
        print(f"Final Robustness (ROUGE-1 F1):\nLLMLogAnalyzer(Llama-3-70B) exhibits robustness with a {final_robustness_rouge1_f1:.2f}% narrower interquartile range compared to other methods.")
    
    if np.mean(cosine_similarity_iqr_differences) > 0:
        print(f"\nFinal Robustness (Cosine Similarity):\nLLMLogAnalyzer(Llama-3-70B) exhibits robustness with a {final_robustness_cosine_similarity:.2f}% wider interquartile range compared to other methods.")
    else:
        print(f"\nFinal Robustness (Cosine Similarity):\nLLMLogAnalyzer(Llama-3-70B) exhibits robustness with a {final_robustness_cosine_similarity:.2f}% narrower interquartile range compared to other methods.")

def main() -> None:
    
    # Save all print into a file
    sys.stdout = open('evaluate/outputs/iqr.txt', 'w')
    dfs = [load_data(file_path) for file_path in CSV_FILES]

    for category in Category:
        category_dfs = [filter_data(df, category.value) for df in dfs]
        print_category_metrics(category_dfs, category.value)

    calculate_final_robustness(dfs)

if __name__ == "__main__":
    main()