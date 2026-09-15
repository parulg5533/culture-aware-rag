# Culture-Aware RAG Evaluation Pipeline

This project implements an evaluation pipeline comparing a baseline Large Language Model (LLM) against a Culture-Aware Retrieval-Augmented Generation (RAG) approach, specifically focusing on Indian languages. The evaluation utilizes the `shanearora/CaLMQA` dataset.

## Project Structure

The pipeline consists of several Python scripts that should be executed sequentially:

1. **`01_setup_dataset.py`**: Downloads and prepares the `shanearora/CaLMQA` dataset from Hugging Face. It filters the dataset to focus on a subset of Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Urdu, Gujarati, Malayalam, Kannada, Odia, Punjabi, Assamese) and saves the output to a CSV file.
2. **`02_run_baseline.py`**: Runs a baseline zero-shot evaluation on a sample of the dataset using Google's Gemini 3.5 Flash model. It saves the generated answers to a CSV file.
3. **`03_culture_aware_retriever.py`**: Implements the retrieval component for the RAG pipeline, retrieving culturally relevant context for the given questions.
4. **`04_culture_aware_generator.py`**: Implements the generation component for the RAG pipeline, producing answers augmented with the retrieved cultural context.
5. **`05_evaluate_rag.py`**: Evaluates and compares the performance of the baseline approach versus the Culture-Aware RAG approach.
6. **`06_dashboard.py`**: A Streamlit web application that visualizes the evaluation results, allowing for detailed side-by-side comparison of the baseline and RAG responses.

## Setup

1. Create a virtual environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment variables:
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the scripts in order:

```bash
# 1. Setup the dataset
python 01_setup_dataset.py

# 2. Generate baseline responses
python 02_run_baseline.py

# 3. Retrieve context
python 03_culture_aware_retriever.py

# 4. Generate RAG responses
python 04_culture_aware_generator.py

# 5. Evaluate the results
python 05_evaluate_rag.py
```

### Dashboard

To view the evaluation results interactively, launch the Streamlit dashboard:

```bash
streamlit run 06_dashboard.py
```

## Requirements
See `requirements.txt` for the full list of dependencies, which include:
- `datasets`
- `pandas`
- `google-generativeai`
- `tqdm`
- `python-dotenv`
- `duckduckgo-search`
- `streamlit` (implied by `06_dashboard.py`)
