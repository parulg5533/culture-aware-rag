import os
import sys
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from tqdm import tqdm

def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    # Using a fast model for large-scale dataset generation
    return genai.GenerativeModel('gemini-3.5-flash-lite')

def analyze_and_formulate(model, question):
    prompt = f"""
    You are an expert in cultural linguistics and query formulation. 
    Analyze the following Hindi question and provide both query characteristics and search strategies.
    
    Question: "{question}"
    
    Respond ONLY with a valid JSON object matching this exact schema:
    {{
        "domain": "One of: [Food, Bollywood/Entertainment, Religion/Mythology, History, Ayurveda/Medicine, Geography, Other]",
        "entity_type": "One of: [Person, Location, Concept, Tradition, Object, Other]",
        "is_ambiguous": boolean (True if the core entity has multiple meanings, e.g., Arjuna the tree vs person),
        
        "query_llm_native": "A highly specific 3-5 word search query in Hindi",
        "query_english": "The direct English translation of the most effective search terms (3-5 words)",
        "query_hybrid": "A mix of the core Hindi entity name and English keywords (e.g., 'अर्जुन छाल medicinal uses')"
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Error in analyze_and_formulate: {e}")
        return None

def retrieve_context(query, region="in-en"):
    if not query:
        return "", 0
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region=region, max_results=3, backend='auto')
            res_list = list(results) if results else []
            
            context_parts = []
            for r in res_list:
                context_parts.append(f"{r.get('title', '')}: {r.get('body', '')}")
            
            return " | ".join(context_parts), len(res_list)
    except Exception as e:
        return "", 0

def evaluate_retrieval(model, question, context):
    if not context.strip():
        return {"entity_match": 0, "cultural_relevance": 0, "evidence_sufficiency": 0}
        
    prompt = f"""
    You are a retrieval evaluator. Evaluate if the retrieved context is useful for answering the question.
    
    Question: "{question}"
    Retrieved Context: "{context}"
    
    Respond ONLY with a valid JSON object matching this schema:
    {{
        "entity_match": int (1 if the context is about the correct entity/sense meant in the question, 0 otherwise),
        "cultural_relevance": int (0 = irrelevant, 1 = partially relevant, 2 = highly culturally relevant),
        "evidence_sufficiency": int (0 = no useful evidence to answer the question, 1 = partial evidence, 2 = directly sufficient evidence)
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception:
        return {"entity_match": 0, "cultural_relevance": 0, "evidence_sufficiency": 0}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Initializing Phase 2: Retrieval Strategy Matrix Generation...")
    
    model = setup_gemini()
    input_file = "calmqa_indian_subset.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    df = pd.read_csv(input_file)
    
    # For the overnight run, we sample 500 questions. 
    # For testing, you can change this to a smaller number.
    SAMPLE_SIZE = min(500, len(df))
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Running matrix generation on {SAMPLE_SIZE} questions...")
    
    matrix_logs = []
    
    for idx, row in tqdm(df_sample.iterrows(), total=SAMPLE_SIZE):
        question = row['question']
        
        record = {
            "question": question,
            "language": "Hindi",
            "region": "in-en",
        }
        
        # 1. Analyze and Formulate
        analysis = analyze_and_formulate(model, question)
        if not analysis:
            time.sleep(2)
            continue
            
        record["domain"] = analysis.get("domain")
        record["entity_type"] = analysis.get("entity_type")
        record["is_ambiguous"] = analysis.get("is_ambiguous")
        
        queries = {
            "Native": question, # Strategy A
            "LLM_Native": analysis.get("query_llm_native", ""), # Strategy B
            "English": analysis.get("query_english", ""), # Strategy C
            "Hybrid": analysis.get("query_hybrid", "") # Strategy D
        }
        
        # 2 & 3. Retrieve and Evaluate for each strategy
        for strategy_name, query in queries.items():
            record[f"query_{strategy_name}"] = query
            
            context, num_results = retrieve_context(query, region="in-en")
            record[f"results_{strategy_name}"] = num_results
            
            eval_metrics = evaluate_retrieval(model, question, context)
            record[f"entity_match_{strategy_name}"] = eval_metrics.get("entity_match", 0)
            record[f"relevance_{strategy_name}"] = eval_metrics.get("cultural_relevance", 0)
            record[f"evidence_{strategy_name}"] = eval_metrics.get("evidence_sufficiency", 0)
            
            time.sleep(1) # DuckDuckGo rate limiting
            
        matrix_logs.append(record)
        time.sleep(2) # Gemini rate limiting
        
        # Save checkpoints every 10 questions in case of failure
        if len(matrix_logs) % 10 == 0:
            pd.DataFrame(matrix_logs).to_csv("retrieval_strategy_matrix.csv", index=False, encoding='utf-8')
            
    # Final save
    pd.DataFrame(matrix_logs).to_csv("retrieval_strategy_matrix.csv", index=False, encoding='utf-8')
    print("\nMatrix Generation Complete! Saved to retrieval_strategy_matrix.csv")

if __name__ == "__main__":
    main()
