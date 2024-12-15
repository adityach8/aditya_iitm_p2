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
#   "ipykernel",  # Required for running in Jupyter environments
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
import openai  # Ensure this library is installed: pip install openai

# ---------------------------- Helper Functions ---------------------------- #

def analyze_data(df):
    """
    Perform a comprehensive analysis of the dataset, including:
    - Summary statistics for numerical columns
    - Missing value counts
    - Correlation matrix for numeric data
    
    Args:
        df (pd.DataFrame): The dataset to analyze.

    Returns:
        tuple: Summary statistics, missing value counts, correlation matrix.
    """
    print("Analyzing the dataset...")
    summary_stats = df.describe()  # Numerical summary
    missing_values = df.isnull().sum()  # Count of missing values
    numeric_df = df.select_dtypes(include=[np.number])  # Numeric-only columns
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    print("Data analysis complete.")
    return summary_stats, missing_values, corr_matrix


def detect_outliers(df):
    """
    Detect outliers in numerical columns using the IQR method.

    Args:
        df (pd.DataFrame): The dataset to analyze.

    Returns:
        pd.Series: Count of outliers for each numeric column.
    """
    print("Detecting outliers...")
    df_numeric = df.select_dtypes(include=[np.number])  # Numeric columns only
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    print("Outlier detection complete.")
    return outliers


def visualize_data(corr_matrix, outliers, df, output_dir):
    """
    Generate visualizations for the analysis, including:
    - Correlation heatmap
    - Outlier counts
    - Distribution plot of the first numeric column

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Outlier counts per column.
        df (pd.DataFrame): The dataset.
        output_dir (str): Directory to save visualizations.

    Returns:
        tuple: File paths of saved visualizations (heatmap, outliers, distribution).
    """
    print("Generating visualizations...")

    # Correlation heatmap
    heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
    if not corr_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.savefig(heatmap_file)
        plt.close()
    else:
        heatmap_file = None

    # Outliers bar chart
    outliers_file = None
    if not outliers.empty and outliers.sum() > 0:
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        plt.savefig(outliers_file)
        plt.close()

    # Distribution plot for the first numeric column
    dist_plot_file = None
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        dist_plot_file = os.path.join(output_dir, 'distribution.png')
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_columns[0]], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {numeric_columns[0]}')
        plt.savefig(dist_plot_file)
        plt.close()

    print("Visualizations generated.")
    return heatmap_file, outliers_file, dist_plot_file


def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    """
    Generate a README.md file summarizing the analysis.

    Args:
        summary_stats (pd.DataFrame): Summary statistics.
        missing_values (pd.Series): Missing value counts.
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Outlier counts.
        output_dir (str): Directory to save the README.

    Returns:
        str: Path to the generated README.md file.
    """
    print("Creating README file...")
    readme_file = os.path.join(output_dir, 'README.md')
    try:
        with open(readme_file, 'w') as f:
            f.write("# Automated Data Analysis Report\n\n")

            # Introduction
            f.write("## Introduction\n")
            f.write("This report provides an automated analysis of the dataset, including statistical summaries, visualizations, and insights.\n\n")

            # Summary statistics
            f.write("## Summary Statistics\n")
            f.write(summary_stats.to_markdown() + "\n\n")

            # Missing values
            f.write("## Missing Values\n")
            f.write(missing_values.to_markdown() + "\n\n")

            # Correlation matrix visualization
            f.write("## Correlation Matrix\n")
            if corr_matrix.empty:
                f.write("No numeric columns available for correlation analysis.\n\n")
            else:
                f.write("![Correlation Matrix](correlation_matrix.png)\n\n")

            # Outliers visualization
            f.write("## Outliers\n")
            if outliers.sum() > 0:
                f.write("![Outliers](outliers.png)\n\n")
            else:
                f.write("No outliers detected in the dataset.\n\n")

            # Distribution plot
            f.write("## Distribution\n")
            f.write("![Distribution](distribution.png)\n\n")

            # Conclusion
            f.write("## Conclusion\n")
            f.write("This automated report summarizes the key patterns and insights from the dataset, helping to guide further analysis.\n")

        print(f"README file created: {readme_file}")
        return readme_file
    except Exception as e:
        print(f"Error writing to README.md: {e}")
        return None


def question_llm(prompt, context):
    """
    Generate a narrative story using an LLM API through a custom proxy.

    Args:
        prompt (str): Specific prompt for the LLM.
        context (str): Context to guide the LLM.

    Returns:
        str: Generated narrative.
    """
    print("Generating story using LLM...")
    try:
        token = os.environ.get("AIPROXY_TOKEN", "")
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{context}\n\n{prompt}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return "Failed to generate story."
    except Exception as e:
        return f"Error: {e}"


# ---------------------------- Main Execution ---------------------------- #

def main(csv_file):
    """
    Main function to integrate data analysis, visualization, and documentation.

    Args:
        csv_file (str): Path to the dataset CSV file.
    """
    print("Starting the analysis...")

    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)

    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    heatmap_file, outliers_file, dist_plot_file = visualize_data(corr_matrix, outliers, df, output_dir)

    story = question_llm(
        "Generate a creative and engaging story from the data analysis.",
        context=f"Dataset Analysis:\nSummary Statistics:\n{summary_stats}\n\nMissing Values:\n{missing_values}\n\nCorrelation Matrix:\n{corr_matrix}\n\nOutliers:\n{outliers}"
    )

    readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)
    if readme_file:
        with open(readme_file, 'a') as f:
            f.write("\n## Story\n")
            f.write(story)

        print(f"Analysis complete! Results saved in '{output_dir}' directory.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
