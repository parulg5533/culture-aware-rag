import os
import sys
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import time

def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("Please set your GEMINI_API_KEY in the .env file")
    
    genai.configure(api_key=api_key)
    # Using gemini-3.5-flash as it is fast and cost-effective for large baselines
    model = genai.GenerativeModel('gemini-3.5-flash')
    return model

def run_baseline():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        model = setup_gemini()
    except Exception as e:
        print(f"Setup Error: {e}")
        return

    input_file = os.path.join(os.path.dirname(__file__), "calmqa_indian_subset.csv")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Please run 01_setup_dataset.py first.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # We will test on a smaller sample first (e.g., first 10 rows)
    # to avoid burning through API quota during testing
    sample_size = 10
    print(f"Running baseline evaluation on the first {sample_size} questions as a test batch...")
    df_sample = df.head(sample_size).copy()
    
    results = []
    
    # Simple zero-shot prompt for baseline
    system_instruction = "You are a helpful AI assistant. Answer the following question accurately in the requested language."
    
    for index, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Generating Answers"):
        question = row['question']
        try:
            # Generate content
            prompt = f"{system_instruction}\n\nQuestion: {question}\nAnswer:"
            response = model.generate_content(prompt)
            generated_answer = response.text.strip()
            
            # Simple rate limiting protection
            time.sleep(1)
            
        except Exception as e:
            print(f"Error generating answer for index {index}: {e}")
            generated_answer = f"ERROR: {e}"
            
        results.append(generated_answer)
        
    df_sample['baseline_generated_answer'] = results
    
    output_file = os.path.join(os.path.dirname(__file__), "baseline_results_sample.csv")
    df_sample.to_csv(output_file, index=False)
    print(f"\nBaseline results saved to {output_file}")
    
    print("\nSample Output:")
    for idx, row in df_sample.head(2).iterrows():
        print("-" * 50)
        print(f"Q ({row.get('language_name', 'Hindi')}): {row['question']}")
        print(f"Baseline Answer: {row['baseline_generated_answer'][:200]}...")

if __name__ == "__main__":
    run_baseline()
