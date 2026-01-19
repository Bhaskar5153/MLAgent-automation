# ml_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
import nbformat
from nbformat import NotebookNode
from nbformat import v4 as nbf

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.file_loader import load_file
from google import genai


class MLAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.notebook_name = f"ml_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"

    def analyze_file(self, df: pd.DataFrame, notebook_name: str = None) -> NotebookNode:
        """Generate a structured Jupyter notebook from a DataFrame."""
        schema = {col: str(df[col].dtype) for col in df.columns}
        data_sample = df.head(10).to_dict(orient="records")

        prompt = f"""
            You are a machine learning expert. Given the following dataset schema and sample data, generate a Jupyter notebook.

            Schema:
            {schema}

            Sample data:
            {data_sample}

            Instructions:
            - Create architecture diagram using https://app.diagrams.net/ and download it as .svg file.
            - Return the notebook as plain text.
            - Use [MARKDOWN] to indicate markdown cells.
            - Use [CODE] to indicate Python code cells.
            - Do not use triple backticks or any other formatting.
            - Each section should start with either [MARKDOWN] or [CODE].
            - Include inline comments in code cells.
            - Add logging to see the flow of data and any issues that arise.
            - Include error handling to manage exceptions and provide informative messages.
            - Create a folder named ml_logs to store log files.
            - The notebook should include: 
            1. Introduction
            2. Data Loading (load the dataset from data directory)
            3. EDA
            4. Preprocessing and make sure outliers are handled if they exist.
            5. Visual representation of EDA. build various plots that explains data well and perform great analysis. use plotly library to build various plots. Explain the each plot in markdown cell below the code cell.
            6. Visual representation of correlation, covariance and explain the plots very clearly.
            7. Feature selection based on EDA
            8. Separate the selected features for training, explain why selected features are taken. 
            9. Modeling. if it is regression or classification or clustering, use appropriate models and explain why the model is selected.
            - Ensure to get best results using appropriate techniques based on the task.
            10. Evaluation metrics that is suitable for the tasks.
            11. Explain Local minima vs Global minima and Visual representation of gradients decent using the dataset.
            12. Explain Risuduals and how to visualize it. Explain the comparison and the metrics to suggest how to improve them.
            13. Explain overfitting or underfitting if it exists. explain how to fix it.
            14. Create example dataset with features used for modeling and make predictions on it
            15. Hyperparameter tuning on sample or small dataset
            16. Visual representation of the results, explain the comparision between predicted and true data.
            17. Final model selection based on best result.
            18. Ensure to save the final model using pickle library and create a folder named artifacts to store the model. Note: The artifacts folder should be created in the root directory.
            19. Insights
            20. Conclusion
            - The sample data is provided for context to understand the stucture and types of data.
            - Ensure to load the full dataset from the CSV file given in the file path.
            - All the instructions from 1 to 18 should followed on the full dataset.
            - If the dataset contains missing values, handle them without fail before preprocessing.
            - Include feature engineering steps if applicable.
            - Use appropriate machine learning models based on the dataset.
            - Include model evaluation metrics and visualizations.
            - You should explain the visualizations and the results in markdown cells.
            - If the dataset is imbalanced, include techniques to handle it.
            - Suggest hyperparameters for models used.
            """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        raw_text = response.text.strip()
        return self._parse_notebook_text(raw_text)

    def _parse_notebook_text(self, raw_text: str) -> NotebookNode:
        nb = nbf.new_notebook()
        cells = []
        current_cell = []
        cell_type = None

        for line in raw_text.splitlines():
            if line.strip() == "[MARKDOWN]":
                if current_cell:
                    if cell_type == "code":
                        cells.append(nbf.new_code_cell("\n".join(current_cell)))
                    elif cell_type == "markdown":
                        cells.append(nbf.new_markdown_cell("\n".join(current_cell)))
                    current_cell = []
                cell_type = "markdown"

            elif line.strip() == "[CODE]":
                if current_cell:
                    if cell_type == "code":
                        cells.append(nbf.new_code_cell("\n".join(current_cell)))
                    elif cell_type == "markdown":
                        cells.append(nbf.new_markdown_cell("\n".join(current_cell)))
                    current_cell = []
                cell_type = "code"

            else:
                current_cell.append(line)

        # Add final cell
        if current_cell:
            if cell_type == "code":
                cells.append(nbf.new_code_cell("\n".join(current_cell)))
            elif cell_type == "markdown":
                cells.append(nbf.new_markdown_cell("\n".join(current_cell)))

        nb['cells'] = cells
        return nb
    
    

if __name__ == "__main__":
    file_path = r"C:\Users\Priya Bhaskar\OneDrive\Documents\ml_agent_repo_1\MLAgent-automation\data\tweet_emotions.csv"
    df = load_file(file_path)
    agent = MLAgent()
    notebook = agent.analyze_file(df, notebook_name=agent.notebook_name)

    # Save notebook
    notebooks_dir = 'notebooks'
    os.makedirs(notebooks_dir, exist_ok=True)
    notebook_path = os.path.join(notebooks_dir, agent.notebook_name)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    print(f"Notebook saved to: {notebook_path}")

