# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai  # Make sure you install this library: pip install openai




# Function to analyze the data
def analyze_data(df):
    """
    Perform basic data analysis including summary statistics,
    missing values check, and correlation matrix computation.

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        tuple: Summary statistics, missing values, and correlation matrix.
    """
    summary_stats = df.describe()
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    return summary_stats, missing_values, corr_matrix

# Function to detect outliers using the IQR method
def detect_outliers(df):
    """
    Detect outliers in the dataset using the IQR method.

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.Series: Number of outliers in each column.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    return outliers

# Function to generate visualizations
def visualize_data(corr_matrix, outliers, df, output_dir):
    """
    Generate and save visualizations including correlation heatmap, 
    outliers plot, and distribution plot for the first numeric column.

    Parameters:
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Outliers count.
        df (pd.DataFrame): The dataset.
        output_dir (str): Directory to save the plots.

    Returns:
        tuple: Paths to the saved visualization files.
    """
    heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(heatmap_file)
    plt.close()

    outliers_file = None
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.savefig(outliers_file)
        plt.close()

    dist_plot_file = None
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_columns[0]], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {numeric_columns[0]}')
        dist_plot_file = os.path.join(output_dir, f'distribution_{numeric_columns[0]}.png')
        plt.savefig(dist_plot_file)
        plt.close()

    return heatmap_file, outliers_file, dist_plot_file

# Function to create README.md file
def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    """
    Generate a README.md file with analysis results and placeholders for visualizations.

    Parameters:
        summary_stats (pd.DataFrame): Summary statistics.
        missing_values (pd.Series): Missing values count.
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Outliers count.
        output_dir (str): Directory to save the README.md file.

    Returns:
        str: Path to the README.md file.
    """
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Introduction\n")
        f.write("This report provides a summary of the dataset, including key statistics, visualizations, and insights.\n\n")

        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown() + "\n\n")

        f.write("## Missing Values\n")
        f.write(missing_values.to_markdown() + "\n\n")

        f.write("## Correlation Matrix\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n\n")

        f.write("## Outliers Detection\n")
        if not outliers.empty:
            f.write(outliers.to_markdown() + "\n\n")
            f.write("![Outliers](outliers.png)\n\n")
        else:
            f.write("No significant outliers detected.\n\n")

        f.write("## Distribution of Data\n")
        f.write("![Distribution](distribution_.png)\n\n")

        f.write("## Conclusion\n")
        f.write("This analysis highlights key patterns and insights in the dataset.\n")

    return readme_path

# Function to generate a story using OpenAI API
def generate_story(prompt, context):
    """
    Generate a narrative based on the analysis using OpenAI API.

    Parameters:
        prompt (str): The prompt for the story.
        context (str): Context of the analysis.

    Returns:
        str: Generated story.
    """
    try:
        token = os.environ["AIPROXY_TOKEN"]
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{context}\n{prompt}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return "Error: Unable to generate story."
    except Exception as e:
        return f"Error: {e}"

# Main function
def main(csv_file):
    """
    Perform automated analysis, generate visualizations, and save a report.

    Parameters:
        csv_file (str): Path to the dataset.
    """
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error reading the dataset: {e}")
        return

    # Analyze the data
    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)

    # Set the output directory to the current working directory
    output_dir = os.getcwd()

    # Generate visualizations and save them in the current working directory
    heatmap_file, outliers_file, dist_plot_file = visualize_data(corr_matrix, outliers, df, output_dir)

    # Create a story context and generate a narrative
    story_context = f"Summary Statistics:\n{summary_stats}\n\nMissing Values:\n{missing_values}\n\nCorrelation Matrix:\n{corr_matrix}\n\nOutliers:\n{outliers}"
    story = generate_story("Create a narrative based on the data analysis.", story_context)

    # Create the README file in the current working directory
    readme_path = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)
    with open(readme_path, 'a') as f:
        f.write("## Data Story\n")
        f.write(story)

    print(f"Analysis complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_path>")
    else:
        main(sys.argv[1])
