import os
import sys
import pandas as pd
import time
import json
from tqdm import tqdm
import importlib

# Import functions from previous script
retriever_module = importlib.import_module("03_culture_aware_retriever")
setup_gemini = retriever_module.setup_gemini

def evaluate_answers(model, question, reference, answer_a, answer_b):
    """Uses LLM-as-a-judge to evaluate baseline vs RAG."""
    prompt = f"""
You are an expert evaluator of AI systems, with a deep understanding of Indian culture and local nuances.
Your task is to evaluate two different AI-generated answers to a question. 

Question: "{question}"
Reference (Ground Truth) Answer: "{reference}"

Answer A (Baseline): "{answer_a}"
Answer B (Culture-Aware RAG): "{answer_b}"

Evaluate both answers on a scale of 1 to 5 based on:
1. Accuracy compared to the reference answer.
2. Cultural awareness and nuance.
3. Clarity and helpfulness.

Respond strictly with a valid JSON object using this schema:
{{
    "score_A": <integer from 1 to 5>,
    "score_B": <integer from 1 to 5>,
    "winner": "<'A', 'B', or 'Tie'>",
    "reasoning": "<1-2 sentences explaining why>"
}}
"""
    try:
        # Generate JSON response directly
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Error evaluating: {e}")
        return {"score_A": 0, "score_B": 0, "winner": "Error", "reasoning": str(e)}

def run_evaluation():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        model = setup_gemini()
    except Exception as e:
        print(f"Setup Error: {e}")
        return

    input_file = os.path.join(os.path.dirname(__file__), "rag_results_sample.csv")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Please run 04_culture_aware_generator.py first.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    scores_baseline = []
    scores_rag = []
    winners = []
    reasonings = []
    
    print(f"Running LLM-as-a-Judge Evaluation on {len(df)} questions...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row['question']
        reference = row['answer']
        baseline = row['baseline_generated_answer']
        rag = row['rag_generated_answer']
        
        eval_result = evaluate_answers(model, question, reference, baseline, rag)
        
        scores_baseline.append(eval_result.get('score_A', 0))
        scores_rag.append(eval_result.get('score_B', 0))
        winners.append(eval_result.get('winner', 'Error'))
        reasonings.append(eval_result.get('reasoning', ''))
        
        time.sleep(2)  # Rate limiting
        
    df['eval_score_baseline'] = scores_baseline
    df['eval_score_rag'] = scores_rag
    df['eval_winner'] = winners
    df['eval_reasoning'] = reasonings
    
    output_file = os.path.join(os.path.dirname(__file__), "evaluation_results_sample.csv")
    df.to_csv(output_file, index=False)
    
    # Calculate statistics
    avg_baseline = sum(scores_baseline) / len(scores_baseline) if scores_baseline else 0
    avg_rag = sum(scores_rag) / len(scores_rag) if scores_rag else 0
    
    wins_rag = winners.count('B')
    wins_baseline = winners.count('A')
    ties = winners.count('Tie')
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS (LLM-as-a-Judge)")
    print("="*50)
    print(f"Average Baseline Score: {avg_baseline:.2f} / 5.0")
    print(f"Average RAG Score:      {avg_rag:.2f} / 5.0")
    print("-" * 50)
    print(f"RAG Wins:      {wins_rag}")
    print(f"Baseline Wins: {wins_baseline}")
    print(f"Ties:          {ties}")
    print("="*50)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    run_evaluation()
