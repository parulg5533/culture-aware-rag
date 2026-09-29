import os
import sys
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
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

def analyze_and_formulate(model, question, language_name):
    prompt = f"""
    You are an expert in cultural linguistics and query formulation. 
    Analyze the following {language_name} question and provide both query characteristics and search strategies.
    
    Question: "{question}"
    
    Respond ONLY with a valid JSON object matching this exact schema:
    {{
        "domain": "One of: [Food, Bollywood/Entertainment, Religion/Mythology, History, Ayurveda/Medicine, Geography, Other]",
        "entity_type": "One of: [Person, Location, Concept, Tradition, Object, Other]",
        "is_ambiguous": boolean (True if the core entity has multiple meanings, e.g., Arjuna the tree vs person),
        
        "query_llm_native": "A highly specific 3-5 word search query in {language_name}",
        "query_english": "The direct English semantic translation (3-5 words)",
        "query_hybrid": "A mix of the core {language_name} entity name and English keywords (e.g., 'अर्जुन छाल medicinal uses')"
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
    result_data = {
        "status": "UNKNOWN",
        "error_type": "None",
        "num_results": 0,
        "latency_ms": 0,
        "context": ""
    }
    if not query:
        return result_data
        
    start_time = time.time()
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region=region, max_results=3, backend='auto')
            res_list = list(results) if results else []
            
            result_data["num_results"] = len(res_list)
            if len(res_list) > 0:
                result_data["status"] = "SUCCESS_WITH_RESULTS"
                context_parts = []
                for r in res_list:
                    # Capture title, URL, and snippet for manual validation
                    context_parts.append(f"[{r.get('title', '')}]({r.get('href', '')}): {r.get('body', '')}")
                result_data["context"] = " | ".join(context_parts)
            else:
                result_data["status"] = "SUCCESS_ZERO_RESULTS"
                
    except DuckDuckGoSearchException as e:
        err_str = str(e).lower()
        if "ratelimit" in err_str or "202" in err_str or "429" in err_str:
            result_data["status"] = "RATE_LIMIT"
        elif "timeout" in err_str:
            result_data["status"] = "TIMEOUT"
        else:
            result_data["status"] = "OTHER_EXCEPTION"
        result_data["error_type"] = str(e)
    except Exception as e:
        result_data["status"] = "OTHER_EXCEPTION"
        result_data["error_type"] = str(e)
        
    result_data["latency_ms"] = int((time.time() - start_time) * 1000)
    return result_data

def evaluate_retrieval(model, question, reference, context):
    if not context.strip():
        return {"entity_match": 0, "cultural_relevance": 0, "evidence_sufficiency": 0}
        
    prompt = f"""
    You are a blinded retrieval evaluator. Evaluate if the retrieved context contains evidence sufficient to support the reference answer.
    
    Question: "{question}"
    Reference Answer: "{reference}"
    Retrieved Context: "{context}"
    
    Respond ONLY with a valid JSON object matching this schema:
    {{
        "entity_match": int (1 if the context is about the exact cultural entity/sense meant in the question, 0 otherwise),
        "cultural_relevance": int (0 = irrelevant, 1 = partially relevant, 2 = highly culturally relevant),
        "evidence_sufficiency": int (0 = no useful evidence to support the reference answer, 1 = partial evidence, 2 = directly sufficient evidence)
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
    output_file = "retrieval_strategy_matrix.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    if 'language_name' not in df.columns:
        raise ValueError("CRITICAL ERROR: 'language_name' column missing from dataset. Cannot perform multilingual analysis.")
        
    # 500 questions exploratory cohort (deterministic seed, stratified by language)
    SAMPLE_SIZE = min(500, len(df))
    
    # Proportional stratified sampling
    df_sample = df.groupby('language_name', group_keys=False).apply(
        lambda x: x.sample(frac=SAMPLE_SIZE/len(df), random_state=42)
    )
    # Adjust to exact SAMPLE_SIZE if rounding caused a mismatch
    if len(df_sample) < SAMPLE_SIZE:
        needed = SAMPLE_SIZE - len(df_sample)
        df_sample = pd.concat([df_sample, df.drop(df_sample.index).sample(n=needed, random_state=42)])
    elif len(df_sample) > SAMPLE_SIZE:
        df_sample = df_sample.sample(n=SAMPLE_SIZE, random_state=42)
    
    # Resume checkpoint logic
    existing_questions = set()
    matrix_logs = []
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_questions = set(existing_df['question'].tolist())
            matrix_logs = existing_df.to_dict('records')
            print(f"Found existing matrix. Resuming from {len(existing_questions)} processed questions.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            
    print(f"Running matrix generation on {SAMPLE_SIZE} questions...")
    
    for idx, row in tqdm(df_sample.iterrows(), total=SAMPLE_SIZE):
        question = row['question']
        reference = row['answer']
        lang = row['language_name']
        
        if question in existing_questions:
            continue
            
        record = {
            "question": question,
            "language": lang,
            "region": "in-en",
        }
        
        # 1. Analyze and Formulate
        analysis = analyze_and_formulate(model, question, lang)
        if not analysis:
            time.sleep(2)
            continue
            
        record["domain"] = analysis.get("domain")
        record["entity_type"] = analysis.get("entity_type")
        record["is_ambiguous"] = analysis.get("is_ambiguous")
        
        queries = {
            "Native": question,
            "LLM_Native": analysis.get("query_llm_native", ""),
            "English": analysis.get("query_english", ""),
            "Hybrid": analysis.get("query_hybrid", "")
        }
        
        # 2 & 3. Retrieve and Evaluate for each strategy
        for strategy_name, query in queries.items():
            record[f"query_{strategy_name}"] = query
            
            res_data = retrieve_context(query, region="in-en")
            record[f"status_{strategy_name}"] = res_data["status"]
            record[f"error_{strategy_name}"] = res_data["error_type"]
            record[f"latency_{strategy_name}"] = res_data["latency_ms"]
            record[f"results_{strategy_name}"] = res_data["num_results"]
            
            context = res_data["context"]
            record[f"context_{strategy_name}"] = context
            
            # Only evaluate if we successfully got results, otherwise trivial zeros
            if res_data["status"] == "SUCCESS_WITH_RESULTS":
                eval_metrics = evaluate_retrieval(model, question, reference, context)
            else:
                eval_metrics = {"entity_match": 0, "cultural_relevance": 0, "evidence_sufficiency": 0}
                
            record[f"entity_match_context_{strategy_name}"] = eval_metrics.get("entity_match", 0)
            record[f"cultural_relevance_context_{strategy_name}"] = eval_metrics.get("cultural_relevance", 0)
            record[f"evidence_sufficiency_context_{strategy_name}"] = eval_metrics.get("evidence_sufficiency", 0)
            
            time.sleep(1) # DuckDuckGo rate limiting
            
        matrix_logs.append(record)
        existing_questions.add(question)
        time.sleep(2) # Gemini rate limiting
        
        # Save checkpoints
        if len(matrix_logs) % 5 == 0:
            pd.DataFrame(matrix_logs).to_csv(output_file, index=False, encoding='utf-8')
            
    # Final save
    pd.DataFrame(matrix_logs).to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nMatrix Generation Complete! Saved {len(matrix_logs)} records to {output_file}")

if __name__ == "__main__":
    main()
