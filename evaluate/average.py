import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of CSV files
csv_files = [
    'evaluate/outputs/Apache/apache_metrics.csv',
    'evaluate/outputs/Linux/linux_metrics.csv',
    'evaluate/outputs/Mac/mac_metrics.csv',
    'evaluate/outputs/Windows/windows_metrics.csv'
]

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

LLMLLMLogAnalyzer70B = 'LLMLogAnalyzer(Llama-3-70B)'

# Save all print into a file
import sys
sys.stdout = open('evaluate/outputs/average.txt', 'w')


########
# Step 1: Calculate the average value for the target column and row
########

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)
    
# Set display options to show all columns
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping

numeric_dfs = [df.select_dtypes(include='number') for df in dfs]

# Sum the df adn divide by the number of dfs
average_metrics_df = sum(numeric_dfs) / len(numeric_dfs)
# Round the result to 2 decimal places
average_metrics_df = average_metrics_df.round(2)

# Add the first column back (assuming 'category' is the first column in original dfs)
category_column = dfs[0]['Category']

# Add the category column back to the average_metrics DataFrame
average_metrics_df.insert(0, 'Category', category_column)

# Export to CSV
average_metrics_df.to_csv('./evaluate/outputs/average_metrics.csv', index=False)

########
# Step 2: Calculate the percentage difference between LLMLogAnalyzer (Llama-3-70B) and other models
########

# Calculate percentage difference between LLMLogAnalyzer (Llama-3-70B) and other models for 7 tasks
print("\nPercentage Difference for Cosine Similarity and ROUGE-1 F1 Score:")

for i, category in enumerate(categories):
    print(f"\nCategory: *****{category}*****")
    # Initialize lists to store percentage differences
    cosine_similarity_diff = []
    for column in average_metrics_df.columns:
        if "Cosine Similarity" in column and LLMLLMLogAnalyzer70B not in column:
            base = average_metrics_df[LLMLLMLogAnalyzer70B + " Cosine Similarity"]
            diff = ((base - average_metrics_df[column]) / base) * 100
            cosine_similarity_diff.append(diff)
            print(f"{LLMLLMLogAnalyzer70B} outperforms {column} by {diff.values[i]:.2f}% in Cosine Similarity")

    print("\n")
    # Initialize lists to store percentage differences
    rouge_1_f1_diff = []
    for column in average_metrics_df.columns:
        if "ROUGE-1 F1" in column and LLMLLMLogAnalyzer70B not in column:
            base = average_metrics_df[LLMLLMLogAnalyzer70B + " ROUGE-1 F1"]
            diff = ((base - average_metrics_df[column]) / base) * 100
            rouge_1_f1_diff.append(diff)
            print(f"{LLMLLMLogAnalyzer70B} outperforms {column} by {diff.values[i]:.2f}% in ROUGE-1 F1")

    # Calculate average percentage difference in Cosine Similarity
    avg_diff_cs = round(sum(cosine_similarity_diff) / len(cosine_similarity_diff), 2)
    print(f"\nAverage Percentage Improvement for \"{category}\" in Cosine Similarity: {avg_diff_cs[i]:.2f}%")
    
    # Calculate average percentage difference in ROUGE-1 F1
    avg_diff_f1 = round(sum(rouge_1_f1_diff) / len(rouge_1_f1_diff), 2)
    print(f"Average Percentage Improvement for \"{category}\" in ROUGE-1 F1: {avg_diff_f1[i]:.2f}%")
    
    print("--*--" * 20)

    
# Print the total average percentage difference
print(f"\nTotal Average Percentage Improvement in Cosine Similarity: {avg_diff_cs.mean():.2f}%")
print(f"Total Average Percentage Improvement in ROUGE-1 F1: {avg_diff_f1.mean():.2f}%")

# Order the categories by the average percentage improvement in both Cosine Similarity and ROUGE-1 F1 with categories names
avg_diff_cs = avg_diff_cs.rename(index=dict(zip(range(len(categories)), categories)))
avg_diff_f1 = avg_diff_f1.rename(index=dict(zip(range(len(categories)), categories)))

# print the result
print("\nAverage Percentage Improvement in Cosine Similarity by Category:")
print(avg_diff_cs)
print("\nAverage Percentage Improvement in ROUGE-1 F1 by Category:")
print(avg_diff_f1)

# cobmited avg_diff_cs and avg_diff_f1 and get the average of the two values
avg_diff = (avg_diff_cs + avg_diff_f1) / 2
print("\nAverage of the two values with order of values:")
print(avg_diff.sort_values(ascending=False))

print(f"The top 3 categories with the highest average percentage improvement in Cosine Similarity and ROUGE-1 F1 are:")
print(f"\t{avg_diff.sort_values(ascending=False).index[:3].values}")

########
# Step 3: Plot the heatmaps for Cosine Similarity and ROUGE-1 F1 scores side by side
########

# Create two pivoted DataFrames for heatmap visualization
cosine_columns = [col for col in df.columns if 'Cosine Similarity' in col]
rouge_columns = [col for col in df.columns if 'ROUGE-1 F1' in col]

# Pivot the data for Cosine Similarity and ROUGE-1 F1
cosine_df = df[['Category'] + cosine_columns].set_index('Category')
rouge_df = df[['Category'] + rouge_columns].set_index('Category')

# Clean up the column names to remove parentheses and metric names
cosine_df.columns = [col.replace('Cosine Similarity', '') for col in cosine_df.columns]
rouge_df.columns = [col.replace('ROUGE-1 F1', '') for col in rouge_df.columns]

cosine_df = cosine_df.T
rouge_df = rouge_df.T

# Specify the desired order of the rows (LLM models)
desired_row_order = [
    'ChatGPT(GPT-4o)', 
    'ChatPDF', 
    'LLMLogAnalyzer(Llama-3-70B)', 
    'LLMLogAnalyzer(Llama-3-8B)', 
    'NotebookLM'
]

# Reindex the DataFrames by the sorted 'Category' values (alphabetical order)
cosine_df = cosine_df.reindex(sorted(cosine_df.index))
rouge_df = rouge_df.reindex(sorted(rouge_df.index))

# Plotting the heatmaps for Cosine Similarity and ROUGE-1 F1 side by side
fig, axs = plt.subplots(1, 2, figsize=(24, 8))

# Set display options to show all columns
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping

# Cosine Similarity heatmap
sns.heatmap(cosine_df, annot=True, ax=axs[0], cmap='YlOrRd', cbar=True, vmin=0, vmax=1, linewidth=.5, fmt=".2f")
axs[0].set_title('Cosine Similarity Heatmap', fontsize=14, pad=20)
axs[0].set_xlabel('Tasks', fontsize=12)
axs[0].set_ylabel('LLM Model', fontsize=12)
axs[0].tick_params(axis='x', rotation=45, labelsize=10)

# ROUGE-1 F1 heatmap
sns.heatmap(rouge_df, annot=True, ax=axs[1], cmap='YlOrRd', cbar=True, vmin=0, vmax=1, linewidth=.5, fmt=".2f")
axs[1].set_title('ROUGE-1 F1 Heatmap', fontsize=14, pad=20)
axs[1].set_xlabel('Tasks', fontsize=12)
axs[1].set_ylabel('LLM Model', fontsize=12)
axs[1].tick_params(axis='x', rotation=45, labelsize=10)


plt.tight_layout()
plt.savefig(f'./evaluate/outputs/average_heatmaps.png')

