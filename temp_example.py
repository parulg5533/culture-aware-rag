import pandas as pd
df = pd.read_csv('evaluation_results_sample.csv')
rag_wins = df[(df['eval_winner'] == 'B') & (df['rag_retrieved_context'].notna())]
if not rag_wins.empty:
    ex = rag_wins.iloc[0]
else:
    ex = df[df['eval_winner'] == 'B'].iloc[0] # Fallback
with open('example.md', 'w', encoding='utf-8') as f:
    f.write(f"Q: {ex['question']}\n\nBaseline:\n{ex['baseline_generated_answer']}\n\nRAG Context:\n{ex['rag_retrieved_context']}\n\nRAG Answer:\n{ex['rag_generated_answer']}\n\nReasoning:\n{ex['eval_reasoning']}")
