import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Culture-Aware RAG Evaluation", layout="wide")

st.title("🇮🇳 Culture-Aware RAG vs Baseline")
st.markdown("Comparing standard LLM responses against our Culture-Aware Retrieval-Augmented Generation pipeline.")

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "evaluation_results_sample.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data()

if df is not None:
    # Top Level Metrics
    st.header("Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    avg_baseline = df['eval_score_baseline'].mean()
    avg_rag = df['eval_score_rag'].mean()
    rag_wins = len(df[df['eval_winner'] == 'B'])
    baseline_wins = len(df[df['eval_winner'] == 'A'])
    
    col1.metric("Avg RAG Score", f"{avg_rag:.2f} / 5", f"{(avg_rag - avg_baseline):.2f}")
    col2.metric("Avg Baseline Score", f"{avg_baseline:.2f} / 5")
    col3.metric("RAG Wins", rag_wins)
    col4.metric("Baseline Wins", baseline_wins)
    
    st.divider()
    
    # Detailed Analysis
    st.header("Detailed Matchups")
    
    # Let user select a question
    question_list = df['question'].tolist()
    selected_q = st.selectbox("Select a question to inspect:", question_list)
    
    row = df[df['question'] == selected_q].iloc[0]
    
    st.subheader(f"Q: {row['question']}")
    with st.expander("Show Ground Truth Reference Answer"):
        st.info(row['answer'])
        
    st.markdown("### Evaluation Reasoning")
    winner_text = "RAG" if row['eval_winner'] == 'B' else ("Baseline" if row['eval_winner'] == 'A' else "Tie")
    if winner_text == "RAG":
        st.success(f"**Winner: {winner_text}**\n\n{row['eval_reasoning']}")
    elif winner_text == "Baseline":
        st.error(f"**Winner: {winner_text}**\n\n{row['eval_reasoning']}")
    else:
        st.warning(f"**Winner: {winner_text}**\n\n{row['eval_reasoning']}")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown(f"### 🤖 Baseline (Score: {row['eval_score_baseline']})")
        st.write(row['baseline_generated_answer'])
        
    with colB:
        st.markdown(f"### 🇮🇳 RAG Pipeline (Score: {row['eval_score_rag']})")
        with st.expander("View Retrieved Context"):
            st.caption(row['rag_retrieved_context'])
        st.write(row['rag_generated_answer'])
        
    st.divider()
    st.subheader("Raw Evaluation Data")
    st.dataframe(df)

else:
    st.warning("No evaluation data found. Please run the evaluation script first.")
