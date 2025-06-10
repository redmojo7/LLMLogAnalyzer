import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load XLSX file
Metrics_Folder = 'evaluate'
dfs = pd.read_excel(Metrics_Folder + '/results.xlsx', sheet_name=None)

# Define output folder
Outputs_Folder = 'outputs'

# Define columns
question_column = 'Question'
expected_answer_column = 'Expected Answer'
category_column = 'Category'

# Define the order of categories
categories = [
    'Summarization',
    'Pattern Extraction',
    'Anomaly Detection',
    'Root Cause Analysis',
    'Predictive Failure Analysis',
    'Log Understanding and Interpretation',
    'Log Filtering and Search'
]

# Initialize metrics dictionaries
accuracy = {}
rouge1_scores = {}

# Create ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1'])

# Initialize vectorizer and lemmatizer
vectorizer = TfidfVectorizer(stop_words='english')
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    tokens = sent_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)

for sheet_name, df in dfs.items():
    
    # Skip sheets starting with "---", which are still in progress
    if sheet_name.startswith("---"):
        print(f"Skipping sheet: {sheet_name}")
        continue

    print(f"Processing sheet: {sheet_name}")
    
    # Get LLM columns
    llm_columns = [col for col in df.columns if col not in [question_column, category_column, expected_answer_column]]
    
    #print("LLM Models: " + str(llm_columns))
    
    # Initialize metrics dictionaries for this sheet
    accuracy[sheet_name] = {llm: [] for llm in llm_columns}
    rouge1_scores[sheet_name] = {llm: [] for llm in llm_columns}

    # Calculate metrics for each LLM model
    for index, row in df.iterrows():
        expected_answer = row[expected_answer_column]
        category = row[category_column]
        for llm in llm_columns:
            answer = row[llm]
            if pd.isnull(answer):
                continue
            
            # Preprocess text
            expected_answer_preprocessed = preprocess_text(expected_answer)
            answer_preprocessed = preprocess_text(answer)
            
            # Calculate cosine similarity
            vectors = vectorizer.fit_transform([expected_answer_preprocessed, answer_preprocessed])
            cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            accuracy[sheet_name][llm].append((category, cosine_sim))
            
            # Calculate ROUGE-1 score
            score = scorer.score(expected_answer, answer)
            rouge1_f1 = score['rouge1'].fmeasure
            rouge1_scores[sheet_name][llm].append((category, rouge1_f1))

    # Create a DataFrame to store average metrics per category
    average_metrics = []

    # Aggregate metrics by category
    for llm in llm_columns:
        accuracy_by_category = {}
        for category, value in accuracy[sheet_name][llm]:
            if category not in accuracy_by_category:
                accuracy_by_category[category] = []
            accuracy_by_category[category].append(value)

        rouge1_by_category = {}
        for category, value in rouge1_scores[sheet_name][llm]:
            if category not in rouge1_by_category:
                rouge1_by_category[category] = []
            rouge1_by_category[category].append(value)

        # Calculate averages for each category
        for category in accuracy_by_category.keys():
            accuracy_avg = sum(accuracy_by_category[category]) / len(accuracy_by_category[category]) if accuracy_by_category[category] else 0
            rouge1_avg = sum(rouge1_by_category[category]) / len(rouge1_by_category[category]) if rouge1_by_category[category] else 0

            average_metrics.append({
                'Sheet': sheet_name,
                'LLM Model': llm,
                'Category': category,
                'Cosine Similarity': accuracy_avg,
                'ROUGE-1 F1': rouge1_avg
            })

    # Convert to DataFrame
    average_metrics_df = pd.DataFrame(average_metrics)

    # Pivot the DataFrame to get categories as columns
    pivoted_df = average_metrics_df.pivot(index=['LLM Model'], columns='Category', 
                                           values=['Cosine Similarity', 'ROUGE-1 F1'])

    # Flatten the multi-level columns
    pivoted_df.columns = [f"{metric} ({category})" for metric, category in pivoted_df.columns]

    # Generate column names based on the order
    columns_order = []
    for category in categories:
        columns_order.append(f'Cosine Similarity ({category})')
        columns_order.append(f'ROUGE-1 F1 ({category})')

    # Reindex the DataFrame with the generated column order
    pivoted_df = pivoted_df.reindex(columns=columns_order)

    # Set display options to show all columns
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping

    # Print average metrics as a DataFrame
    #print("Average Metrics:")
    #print(pivoted_df.to_string())
    
    # Create output folder if it does not exist
    if not os.path.exists(f'{Metrics_Folder}/{Outputs_Folder}'):
        os.makedirs(f'{Metrics_Folder}/{Outputs_Folder}')
        
    # Create subfolder for the sheet
    if not os.path.exists(f'{Metrics_Folder}/{Outputs_Folder}/{sheet_name}'):
        os.makedirs(f'{Metrics_Folder}/{Outputs_Folder}/{sheet_name}')

    # Plotting the heatmaps for Accuracy and F1 scores side by side
    fig, axs = plt.subplots(1, 2, figsize=(24, 8))

    # Accuracy heatmap
    accuracy_data = pivoted_df.filter(like='Cosine Similarity')
    accuracy_data.columns = [col.split('(')[-1].strip(') ') for col in accuracy_data.columns]
    sns.heatmap(accuracy_data, ax=axs[0], annot=True, cmap='YlOrRd', cbar=True, vmin=0, vmax=1, linewidth=.5, fmt=".2f")
    axs[0].set_title(f'{sheet_name} Cosine Similarity Heatmap', fontsize=14, pad=20)
    axs[0].set_xlabel('Tasks', fontsize=12)
    axs[0].set_ylabel('LLM Model', fontsize=12)
    axs[0].tick_params(axis='x', rotation=45, labelsize=10)

    # F1 heatmap
    f1_data = pivoted_df.filter(like='F1')
    f1_data.columns = [col.split('(')[-1].strip(') ') for col in f1_data.columns]
    sns.heatmap(f1_data, ax=axs[1], annot=True, cmap='YlOrRd', cbar=True, vmin=0, vmax=1, linewidth=.5, fmt=".2f")
    axs[1].set_title(f'{sheet_name} ROUGE-1 F1 Heatmap', fontsize=14, pad=20)
    axs[1].set_xlabel('Tasks', fontsize=12)
    axs[1].set_ylabel('LLM Model', fontsize=12)
    axs[1].tick_params(axis='x', rotation=45, labelsize=10)

    plt.tight_layout()
    plt.savefig(f'{Metrics_Folder}/{Outputs_Folder}/{sheet_name}/{sheet_name.lower()}_heatmaps.png')
    

    # Melt the DataFrame to have 'Metric' column
    average_metrics_df = pd.melt(average_metrics_df, id_vars=['Sheet', 'LLM Model', 'Category'], 
                                 value_vars=['Cosine Similarity', 'ROUGE-1 F1'], 
                                 var_name='Metric', value_name='Value')
    # Pivot the DataFrame to get LLM models as columns
    pivoted_df = pd.pivot_table(average_metrics_df, index='Category', columns=['Sheet', 'LLM Model', 'Metric'], 
                                values='Value', aggfunc='mean')

    # Flatten the multi-level columns
    pivoted_df.columns = [f"{llm} {metric}" for sheet, llm, metric in pivoted_df.columns]

    # Specify column order
    llm_columns = ['ChatGPT(GPT-4o)', 'NotebookLM', 'ChatPDF', 'LLMLogAnalyzer(Llama-3-70B)', 'LLMLogAnalyzer(Llama-3-8B)']
    metrics = ['Cosine Similarity', 'ROUGE-1 F1']
    column_order = [f"{llm} {metric}" for llm in llm_columns for metric in metrics]

    # Reindex the DataFrame with the generated column order
    pivoted_df = pivoted_df.reindex(columns=column_order)

    # Reorder rows
    pivoted_df = pivoted_df.loc[categories]

    # Save average metrics to a CSV file
    pivoted_df.to_csv(f'{Metrics_Folder}/{Outputs_Folder}/{sheet_name}/{sheet_name.lower()}_metrics.csv', index=True, float_format='%.2f')