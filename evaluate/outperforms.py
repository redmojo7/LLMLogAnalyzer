import pandas as pd

# List of CSV files
csv_files = [
    'evaluate/outputs/Apache/apache_metrics.csv',
    'evaluate/outputs/Linux/linux_metrics.csv',
    'evaluate/outputs/Mac/mac_metrics.csv',
    'evaluate/outputs/Windows/windows_metrics.csv'
]

# Save all print into a file
import sys
sys.stdout = open('evaluate/outputs/outperforms.txt', 'w')

LLMLLMLogAnalyzer70B = 'LLMLogAnalyzer(Llama-3-70B)'

LLMLogAnalyzers = ['LLMLogAnalyzer(Llama-3-70B)', 'LLMLogAnalyzer(Llama-3-8B)']

# Initialize dictionaries to store average scores
cosine_similarity_avg = {}
rouge_1_f1_avg = {}

# Iterate over each CSV file
for csv_file in csv_files:
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract model names from column headers
    models = set()
    for col in df.columns:
        if col != 'Category':  # Exclude the 'Category' column
            model_name = col.split(' Cosine')[0].split(' ROUGE')[0]
            models.add(model_name)
            
    # Calculate average scores for each model
    for model in models:
        cosine_cols = [col for col in df.columns if model in col and 'Cosine' in col]
        rouge_cols = [col for col in df.columns if model in col and 'ROUGE' in col]
        
        # Initialize model scores if not already initialized
        if model not in cosine_similarity_avg:
            cosine_similarity_avg[model] = []
        if model not in rouge_1_f1_avg:
            rouge_1_f1_avg[model] = []
        
        # Append scores from current CSV file
        cosine_similarity_avg[model].append(df[cosine_cols].mean().mean())
        rouge_1_f1_avg[model].append(df[rouge_cols].mean().mean())
        
        
# sort average scores
cosine_similarity_avg = dict(sorted(cosine_similarity_avg.items(), key=lambda x: x[1], reverse=True))
rouge_1_f1_avg = dict(sorted(rouge_1_f1_avg.items(), key=lambda x: x[1], reverse=True))

# Calculate overall average scores for each model
for model in cosine_similarity_avg:
    cosine_similarity_avg[model] = sum(cosine_similarity_avg[model]) / len(cosine_similarity_avg[model])
for model in rouge_1_f1_avg:
    rouge_1_f1_avg[model] = sum(rouge_1_f1_avg[model]) / len(rouge_1_f1_avg[model])

# Print the results
print("Average Cosine Similarity Score:")
for model, score in cosine_similarity_avg.items():
    print(f"{model}: {score:.4f}")

print("\nAverage ROUGE-1 F1 Score:")
for model, score in rouge_1_f1_avg.items():
    print(f"{model}: {score:.4f}")

# Calculate percentage difference between LLMLogAnalyzer (Llama-3-70B) and other models
print("\nPercentage Difference for Cosine Similarity: ")
cosine_similarity_diff = []
models_sorted = sorted(cosine_similarity_avg.items(), key=lambda x: x[1], reverse=True)
for model, score in models_sorted:
    if model != LLMLLMLogAnalyzer70B:
        diff = ((cosine_similarity_avg[LLMLLMLogAnalyzer70B] - score) / cosine_similarity_avg[LLMLLMLogAnalyzer70B]) * 100
        cosine_similarity_diff.append(diff)
        print(f"{LLMLLMLogAnalyzer70B} outperforms {model} by {diff:.2f}% in Cosine Similarity")
       
# Calculate average percentage difference
avg_diff = sum(cosine_similarity_diff) / len(cosine_similarity_diff)
print(f"\nAverage Percentage Improvement in Cosine Similarity: {avg_diff:.2f}%")
        
print("\nPercentage Difference for ROUGE-1 F1 Score: ")
rouge_1_f1_diff = []
models_sorted = sorted(rouge_1_f1_avg.items(), key=lambda x: x[1], reverse=True)
for model, score in models_sorted:
    if model != LLMLLMLogAnalyzer70B:
        diff = ((rouge_1_f1_avg[LLMLLMLogAnalyzer70B] - score) / rouge_1_f1_avg[LLMLLMLogAnalyzer70B]) * 100
        rouge_1_f1_diff.append(diff)
        print(f"{LLMLLMLogAnalyzer70B} outperforms {model} by {diff:.2f}% in ROUGE-1 F1 Score")

# Calculate average percentage difference
avg_diff = sum(rouge_1_f1_diff) / len(rouge_1_f1_diff)
print(f"\nAverage Percentage Improvement in ROUGE-1 F1 Score: {avg_diff:.2f}%") 


print("\n", "-*-"*20)

# Calculate percentage difference between LLMLogAnalyzer (Llama-3-70B) and LLMLogAnalyzer (Llama-3-8B)
print("\nPercentage Difference between LLMLogAnalyzer (Llama-3-70B) and LLMLogAnalyzer (Llama-3-8B): ")
diff = ((cosine_similarity_avg[LLMLLMLogAnalyzer70B] - cosine_similarity_avg['LLMLogAnalyzer(Llama-3-8B)']) / cosine_similarity_avg[LLMLLMLogAnalyzer70B]) * 100
print(f"{LLMLLMLogAnalyzer70B} outperforms LLMLogAnalyzer(Llama-3-8B) by {diff:.2f}% in Cosine Similarity")

diff = ((rouge_1_f1_avg[LLMLLMLogAnalyzer70B] - rouge_1_f1_avg['LLMLogAnalyzer(Llama-3-8B)']) / rouge_1_f1_avg[LLMLLMLogAnalyzer70B]) * 100
print(f"{LLMLLMLogAnalyzer70B} outperforms LLMLogAnalyzer(Llama-3-8B) by {diff:.2f}% in ROUGE-1 F1 Score")

print("\n", "-*-"*20)

# Calculate percentage LLMLogAnalyzer(Llama-3-70B) on each tasks
print("\nPercentage LLMLogAnalyzer(Llama-3-70B) on each tasks: ")
tasks = ['Summarization', 'Pattern Extraction', 'Anomaly Detection', 'Root Cause Analysis', 'Predictive Failure Analysis', 'Log Understanding and Interpretation', 'Log Filtering and Search']

# Initialize dictionaries to store scores
cosine_similarity_scores = {task: [] for task in tasks}
rouge_1_f1_scores = {task: [] for task in tasks}

# Iterate over each CSV file
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # Iterate over each task
    for task in tasks:
        task_row = df[df['Category'] == task]
        cosine_similarity = task_row['LLMLogAnalyzer(Llama-3-70B) Cosine Similarity'].values[0]
        rouge_1_f1 = task_row['LLMLogAnalyzer(Llama-3-70B) ROUGE-1 F1'].values[0]
        
        #print(f"CSV: {csv_file}, Task: {task}")
        #print(f"LLMLogAnalyzer(Llama-3-70B) Cosine Similarity: {cosine_similarity}")
        #rint(f"LLMLogAnalyzer(Llama-3-70B) ROUGE-1 F1: {rouge_1_f1}")
        
        cosine_similarity_scores[task].append(cosine_similarity)
        rouge_1_f1_scores[task].append(rouge_1_f1)

# Calculate averages
task_avg_cosine_similarity = {}
task_avg_rouge_1_f1 = {}

for task in tasks:
    avg_cosine_similarity = sum(cosine_similarity_scores[task]) / len(cosine_similarity_scores[task])
    avg_rouge_1_f1 = sum(rouge_1_f1_scores[task]) / len(rouge_1_f1_scores[task])
    
    task_avg_cosine_similarity[task] = avg_cosine_similarity
    task_avg_rouge_1_f1[task] = avg_rouge_1_f1

# Sort tasks by average score in descending order
sorted_cosine_tasks = sorted(task_avg_cosine_similarity.items(), key=lambda x: x[1], reverse=True)
sorted_rouge_tasks = sorted(task_avg_rouge_1_f1.items(), key=lambda x: x[1], reverse=True)

# Print sorted tasks
print("\nTasks by Average Cosine Similarity (High to Low):")
for task, avg_score in sorted_cosine_tasks:
    print(f"{task}: {avg_score:.4f}")

print("\nTasks by Average ROUGE-1 F1 Score (High to Low):")
for task, avg_score in sorted_rouge_tasks:
    print(f"{task}: {avg_score:.4f}")
    