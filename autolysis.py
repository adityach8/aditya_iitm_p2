import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np
from dateutil.parser import parse
import openai
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Set OpenAI API Key from Environment
openai.api_key = os.getenv("AIPROXY_TOKEN")
if not openai.api_key:
    raise ValueError("Environment variable 'AIPROXY_TOKEN' is not set.")

# API Endpoint Configuration
OPENAI_API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {openai.api_key}",
    "Content-Type": "application/json"
}

class DatasetRequest(BaseModel):
    file_path: str


@app.post("/analyze")
def analyze_dataset_api(request: DatasetRequest):
    """
    API Endpoint to analyze the dataset and generate a report.
    """
    try:
        data, summary = analyze_dataset(request.file_path)
        visualizations = visualize_data(data, summary)
        narrative = narrate_with_llm(summary)
        generate_readme(narrative, visualizations, summary)
        return {"message": "Analysis complete. Results saved in README.md."}
    except Exception as e:
        return {"error": str(e)}


# Step 1: Analyze Dataset
def analyze_dataset(file_path):
    """
    Analyze the dataset for descriptive statistics, correlations, and missing values.
    """
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        raise ValueError(f"Error loading the file: {e}")

    numeric_data = data.select_dtypes(include=['number'])
    desc_stats = data.describe(include='all').transpose()

    summary = {
        "num_rows": len(data),
        "num_columns": len(data.columns),
        "columns": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "desc_stats_summary": desc_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']] if not desc_stats.empty else None,
        "corr_matrix_summary": numeric_data.corr().describe() if not numeric_data.empty else None,
    }

    return data, summary


# Step 2: Enhanced Visualizations
def visualize_data(data, summary, output_dir="./"):
    """
    Create and save visualizations based on the dataset.
    """
    visualizations = []

    numeric_columns = data.select_dtypes(include=['number']).columns
    if numeric_columns.any():
        for col in numeric_columns[:2]:  # Limit to top 2 visualizations
            plot_file = os.path.join(output_dir, f"{col}_distribution.png")
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col].dropna(), kde=True, color='skyblue', alpha=0.7)
            plt.title(f"Distribution of {col}", fontsize=16)
            plt.xlabel(col, fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(plot_file, dpi=100)
            visualizations.append(plot_file)
            plt.close()

    if summary.get("corr_matrix_summary") is not None:
        heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
        plt.figure(figsize=(12, 8))
        sns.heatmap(summary["corr_matrix_summary"], annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap", fontsize=16)
        plt.savefig(heatmap_file, dpi=100)
        visualizations.append(heatmap_file)
        plt.close()

    return visualizations


# Step 3: Generate Narrative with LLM
def narrate_with_llm(summary):
    """
    Generate a narrative report using an LLM based on the dataset summary.
    """
    sanitized_summary = sanitize_summary(summary)

    prompt = (
        f"Analyze this dataset summary and write a clear, concise narrative:\n\n"
        f"Dataset:\n- Rows: {sanitized_summary['num_rows']}, Columns: {sanitized_summary['num_columns']}\n"
        f"- Missing Values: {sanitized_summary['missing_values']}\n"
        f"- Column Types: {list(sanitized_summary['columns'].keys())}\n"
        f"- Key Stats: {sanitized_summary['desc_stats_summary']}\n"
        f"- Correlation Summary: {sanitized_summary['corr_matrix_summary']}\n\n"
        f"Focus on trends, patterns, and actionable insights."
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a concise and insightful data analyst."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        return validate_output(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        return f"Error generating narrative: {e}"


# Step 4: Generate README.md
def generate_readme(narrative, visualizations, summary):
    """
    Generate a README.md report based on the narrative, visualizations, and summary.
    """
    with open("README.md", "w") as file:
        file.write("# Automated Data Analysis Report\n\n")
        file.write("## Dataset Description\n")
        file.write(f"- **Rows:** {summary['num_rows']}\n")
        file.write(f"- **Columns:** {summary['num_columns']}\n")
        file.write(f"- **Missing Values:** {summary['missing_values']}\n\n")
        file.write("## Narrative Analysis\n")
        file.write(narrative + "\n\n")
        file.write("## Visualizations\n")
        for vis in visualizations:
            file.write(f"![{os.path.basename(vis)}]({vis})\n")


# Sanitization and Validation Utilities
def sanitize_summary(summary):
    """Ensure the summary is safe for further processing."""
    sanitized = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            sanitized[key] = {k: str(v).replace('\n', ' ') for k, v in value.items()}
        elif isinstance(value, pd.DataFrame):
            sanitized[key] = value.head(5).to_markdown()
        else:
            sanitized[key] = str(value).replace('\n', ' ')
    return sanitized


def validate_output(output):
    """Check for forbidden patterns in the LLM output."""
    forbidden_patterns = ["<script>", "exec(", "os.system(", "import ", "```python", "DROP TABLE"]
    if any(pattern in output for pattern in forbidden_patterns):
        raise ValueError("Invalid content detected in LLM output.")
    return output
