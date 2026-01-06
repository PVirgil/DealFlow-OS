# streamlit_app.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
import logging

# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Helper

def call_llm(prompt: str, model: str = "mixtral-8x7b-32768") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a venture capital analyst, due diligence officer, and LP memo writer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Modules

def generate_deal_memo(df: pd.DataFrame) -> str:
    cols = ", ".join(df.columns)
    prompt = (
        f"Given the following startup data with columns: {cols}, generate an investor-style deal memo."
        " Include Problem, Solution, Market Size, Traction, Product, Team, Risks, and Recommendation."
    )
    return call_llm(prompt)

def evaluate_term_sheet(text: str) -> str:
    prompt = f"Evaluate and summarize this startup term sheet: {text}"
    return call_llm(prompt)

def investor_qna(question: str, context: str) -> str:
    prompt = f"Based on this startup data: {context}, answer: {question} like a VC partner."
    return call_llm(prompt)

def generate_scorecard(df: pd.DataFrame) -> str:
    prompt = (
        "Rate this startup using a scorecard from 1 to 10 for: Team, Market, Product, Traction, Competitive Advantage, Risk."
        f" Use the following data: {df.head(3).to_dict()}"
    )
    return call_llm(prompt)

# Streamlit UI

def main():
    st.set_page_config("DealFlow OS", page_icon="📁", layout="wide")
    st.title("📁 DealFlow OS — The AI Diligence Analyst")
    st.write("Upload startup data, get investor memos, term sheet analysis, scorecards, and partner-style Q&A.")

    uploaded_file = st.file_uploader("Upload startup dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Startup data loaded!")
    else:
        df = pd.DataFrame()

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Deal Memo", "📊 Scorecard", "🧾 Term Sheet", "💬 Investor Q&A"])

    with tab1:
        st.subheader("📝 Generate AI Deal Memo")
        if st.button("Generate Memo"):
            if df.empty:
                st.error("Please upload startup data.")
            else:
                memo = generate_deal_memo(df)
                st.text_area("Deal Memo", value=memo, height=400)

    with tab2:
        st.subheader("📊 AI Scorecard")
        if st.button("Generate Scorecard"):
            if df.empty:
                st.error("Please upload startup data.")
            else:
                score = generate_scorecard(df)
                st.text_area("Scorecard", value=score, height=300)

    with tab3:
        st.subheader("🧾 Term Sheet Evaluator")
        term = st.text_area("Paste term sheet text")
        if st.button("Evaluate Term Sheet"):
            if not term:
                st.error("Paste the term sheet text.")
            else:
                out = evaluate_term_sheet(term)
                st.text_area("Summary", value=out, height=300)

    with tab4:
        st.subheader("💬 Partner-Style Q&A")
        context = st.text_area("Paste or describe the startup context")
        question = st.text_input("Ask a funding-related question")
        if st.button("Ask Investor AI"):
            if not context or not question:
                st.error("Provide both context and question.")
            else:
                a = investor_qna(question, context)
                st.markdown(f"**Investor AI:** {a}")

if __name__ == "__main__":
    main()
