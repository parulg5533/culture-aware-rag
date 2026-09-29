import os
import sys
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json
import random

def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    # Using a fast model for large-scale evaluation
    return genai.GenerativeModel('gemini-3.5-flash-lite')

COMPREHENSIVE_EVAL_PROMPT = """
You are an expert AI researcher conducting a rigorous, blinded evaluation of a Retrieval-Augmented Generation (RAG) system for Indian cultural QA.
You will evaluate both the QUALITY of the generated answers (blinded) and the QUALITY of the retrieval mechanism itself.

INPUTS:
Question: {question}
Ground Truth Reference: {reference}
Retrieved Context (for RAG): {context}

Answer A: {answer_a}
Answer B: {answer_b}

TASK:
Return a JSON object strictly adhering to this schema:
{{
  "quality_evaluation": {{
    "score_A": int (1-5),
    "score_B": int (1-5),
    "winner": str ("A", "B", or "Tie"),
    "reasoning": str (short explanation)
  }},
  "retrieval_evaluation": {{
    "entity_match": bool (Did the context retrieve information about the correct cultural entity?),
    "cultural_relevance": int (0=None, 1=Partial, 2=Highly relevant),
    "evidence_sufficiency": int (0=No useful evidence, 1=Partially useful, 2=Sufficient to answer)
  }},
  "generation_evaluation": {{
    "is_rag_answer_grounded": bool (Did the RAG answer strictly follow the retrieved context?),
    "rag_hallucinated": bool (Did the RAG answer invent ungrounded facts?)
  }},
  "failure_modes": {{
    "retrieval_failures": list of strings (Choose from: "Entity ambiguity", "Wrong cultural context", "Wrong region", "Multilingual retrieval failure", "Conflicting sources", "Insufficient evidence", "None"),
    "generation_failures": list of strings (Choose from: "Hallucinated / unsupported answer", "Correct evidence but bad generation", "None")
  }}
}}

NOTE: To evaluate RAG's generation/failures, you must determine which Answer (A or B) corresponds to the RAG system by comparing them to the 'Retrieved Context'. The RAG answer is the one that heavily utilizes the provided context.
"""

def main():
    print("Initializing Phase 2: Comprehensive Failure Analysis & Evaluation...")
    model = setup_gemini()
    
    # We load the output from Phase 3 (Generator) which has both Baseline and RAG answers
    input_file = "rag_results_sample.csv" 
    output_file = "phase2_exploratory_analysis.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the generator pipeline first.")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} questions for rigorous analysis.")
    
    results = []
    
    print("Running blinded, multi-label evaluation...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        # BLINDING THE JUDGE: Randomly assign Baseline and RAG to A or B
        is_rag_a = random.choice([True, False])
        
        if is_rag_a:
            answer_a = row['rag_generated_answer']
            answer_b = row['baseline_generated_answer']
        else:
            answer_a = row['baseline_generated_answer']
            answer_b = row['rag_generated_answer']
            
        prompt = COMPREHENSIVE_EVAL_PROMPT.format(
            question=row['question'],
            reference=row['answer'],
            context=row['rag_retrieved_context'],
            answer_a=answer_a,
            answer_b=answer_b
        )
        
        try:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            eval_data = json.loads(response.text)
            
            # De-blind the scores
            if is_rag_a:
                rag_score = eval_data["quality_evaluation"]["score_A"]
                baseline_score = eval_data["quality_evaluation"]["score_B"]
                if eval_data["quality_evaluation"]["winner"] == "A": winner = "RAG"
                elif eval_data["quality_evaluation"]["winner"] == "B": winner = "Baseline"
                else: winner = "Tie"
            else:
                rag_score = eval_data["quality_evaluation"]["score_B"]
                baseline_score = eval_data["quality_evaluation"]["score_A"]
                if eval_data["quality_evaluation"]["winner"] == "B": winner = "RAG"
                elif eval_data["quality_evaluation"]["winner"] == "A": winner = "Baseline"
                else: winner = "Tie"
                
            record = {
                "question": row['question'],
                "rag_winner": True if winner == "RAG" else False,
                "baseline_score": baseline_score,
                "rag_score": rag_score,
                
                # Retrieval Metrics
                "entity_match": eval_data["retrieval_evaluation"]["entity_match"],
                "cultural_relevance": eval_data["retrieval_evaluation"]["cultural_relevance"],
                "evidence_sufficiency": eval_data["retrieval_evaluation"]["evidence_sufficiency"],
                
                # Generation Metrics
                "rag_grounded": eval_data["generation_evaluation"]["is_rag_answer_grounded"],
                "rag_hallucinated": eval_data["generation_evaluation"]["rag_hallucinated"],
                
                # Hierarchical Failures (Stored as comma-separated strings for CSV compatibility)
                "retrieval_failures": ", ".join(eval_data["failure_modes"]["retrieval_failures"]),
                "generation_failures": ", ".join(eval_data["failure_modes"]["generation_failures"]),
            }
            results.append(record)
            
        except Exception as e:
            q_safe = str(row['question']).encode('ascii', 'ignore').decode()
            print(f"\nAPI Error on question '{q_safe}': {e}")
            
        time.sleep(3) # Rate limit protection
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nComprehensive Analysis complete. Saved to {output_file}")
    
    # Print Quick Stats
    if not results_df.empty:
        print("\n=== QUICK STATS ===")
        print(f"RAG Win Rate: {results_df['rag_winner'].mean()*100:.1f}%")
        print(f"Entity Match Rate: {results_df['entity_match'].mean()*100:.1f}%")
        print(f"Evidence Sufficiency (0-2): {results_df['evidence_sufficiency'].mean():.2f}")
        print(f"RAG Groundedness Rate: {results_df['rag_grounded'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()
