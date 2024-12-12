# Script Requirements
# Requires Python version >= 3.9
# Dependencies:
#   pandas, seaborn, matplotlib, numpy, scipy, openai, scikit-learn, requests, ipykernel

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai

# Function: Analyze Data
def analyze_data(df):
    print("Analyzing the dataset...")
    
    summary_stats = df.describe()  # Summary statistics
    missing_values = df.isnull().sum()  # Missing value counts
    numeric_df = df.select_dtypes(include=[np.number])  # Numeric columns
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()  # Correlation matrix

    print("Data analysis completed.")
    return summary_stats, missing_values, corr_matrix

# Function: Detect Outliers
def detect_outliers(df):
    print("Detecting outliers...")
    
    df_numeric = df.select_dtypes(include=[np.number])  # Numeric columns only
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1  # Interquartile range

    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()

    print("Outlier detection completed.")
    return outliers

# Function: Visualize Data
def visualize_data(corr_matrix, outliers, df, output_dir):
    print("Generating visualizations...")

    # Correlation Heatmap
    if not corr_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(heatmap_file)
        plt.close()
    else:
        heatmap_file = None

    # Outlier Bar Plot
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.savefig(outliers_file)
        plt.close()
    else:
        outliers_file = None

    # Distribution Plot for First Numeric Column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {first_numeric_column}')
        dist_plot_file = os.path.join(output_dir, f'distribution_{first_numeric_column}.png')
        plt.savefig(dist_plot_file)
        plt.close()
    else:
        dist_plot_file = None

    print("Visualizations generated.")
    return heatmap_file, outliers_file, dist_plot_file

# Function: Generate README.md
def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    print("Creating README.md file...")

    readme_file = os.path.join(output_dir, 'README.md')
    try:
        with open(readme_file, 'w') as f:
            f.write("# Automated Data Analysis Report\n\n")
            f.write("## Introduction\n")
            f.write("This automated analysis provides a comprehensive view of the dataset, including summary statistics, visualizations, and key insights.\n\n")

            # Summary Statistics
            f.write("## Summary Statistics\n")
            f.write(summary_stats.to_markdown() + "\n\n")

            # Missing Values
            f.write("## Missing Values\n")
            f.write(missing_values.to_markdown() + "\n\n")

            # Outliers
            f.write("## Outliers Detection\n")
            if outliers.sum() > 0:
                f.write(outliers.to_markdown() + "\n\n")
            else:
                f.write("No significant outliers detected.\n\n")

            # Correlation Matrix
            f.write("## Correlation Matrix\n")
            if corr_matrix.empty:
                f.write("No numeric data available for correlation analysis.\n\n")
            else:
                f.write("![Correlation Matrix](correlation_matrix.png)\n\n")

            # Visualizations
            f.write("## Data Visualizations\n")
            if heatmap_file:
                f.write("- Correlation Matrix: ![Correlation Matrix](correlation_matrix.png)\n")
            if outliers_file:
                f.write("- Outliers Detection: ![Outliers](outliers.png)\n")
            if dist_plot_file:
                f.write(f"- Distribution: ![Distribution Plot](distribution_{first_numeric_column}.png)\n")

            f.write("## Conclusion\n")
            f.write("This analysis provides valuable insights into the dataset. Use the findings to guide further exploration and decision-making.\n")

        print(f"README.md created at {readme_file}")
        return readme_file

    except Exception as e:
        print(f"Error writing README.md: {e}")
        return None

# Function: Generate Story using LLM
def generate_story_with_llm(prompt, context):
    print("Generating story using LLM...")
    try:
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('AIPROXY_TOKEN', '')}"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context + "\n\n" + prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM generation error: {e}")
        return "Story generation failed."

# Main Function
def main(csv_file):
    print("Starting analysis pipeline...")

    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)

    output_dir = os.path.splitext(csv_file)[0]
    os.makedirs(output_dir, exist_ok=True)

    heatmap_file, outliers_file, dist_plot_file = visualize_data(corr_matrix, outliers, df, output_dir)

    story_context = f"Summary Statistics:\n{summary_stats}\n\nMissing Values:\n{missing_values}\n\nOutliers:\n{outliers}\n"
    story = generate_story_with_llm("Generate a creative story from the data analysis.", story_context)

    readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)
    if readme_file:
        with open(readme_file, 'a') as f:
            f.write("\n## Story\n")
            f.write(story)

    print(f"Analysis complete! Files saved in {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])

