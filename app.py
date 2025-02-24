import os
import re
import json
import inspect
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Any, Optional, Dict, Callable, TypeVar
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.caches import InMemoryCache

load_dotenv()
set_llm_cache(InMemoryCache())

class DataAnalyzerAssistant:
    def __init__(self, model):
        self.model = model
        self.setup_prompts()    

    def setup_prompts(self):
        self.analysis_prompt = PromptTemplate.from_template("""
            You are a data analysis expert. Analyze the dataset and user's question to:
            1. Understand the data structure
            2. Determine relevant columns for visualization
            3. Recommend appropriate chart types
            
            Dataset sample (first 3 rows):
            {sample_data}
            
            Columns: {columns}
            User question: {question}
            
            Provide response in this format:
            {{
                "chart_type": "bar|line|scatter|pie|histogram|none",
                "x_column": "column_name",
                "y_column": "column_name|none",
                "color_column": "column_name|none",
                "reason": "brief explanation"
            }}
            """)
            
        self.summary_prompt = PromptTemplate.from_template("""
            Generate a natural language summary of the data analysis based on:
            - User question: {question}
            - Dataset shape: {shape}
            - Selected columns: {columns}
            - Chart type: {chart_type}
            
            Include key insights and patterns. Response:
            """)

    def suggest_visualization(self, df, question):
        chain = (
            RunnablePassthrough.assign(
                sample_data=lambda _: df.head(3).to_dict(),
                columns=lambda _: list(df.columns),
                shape=lambda _: df.shape
            )
            | self.analysis_prompt
            | self.model
            | StrOutputParser()
        )
        result_str = chain.invoke({"question": question})
        if not result_str:
            raise ValueError("Received empty response from the model.")
        try:
            # Extract JSON content from within code fences
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in the model's response.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {result_str}") from e

    def generate_insight(self, df, question, chart_info):
        chain = (
            self.summary_prompt
            | self.model
            | StrOutputParser()
        )
        return chain.invoke({
            "question": question,
            "shape": df.shape,
            "columns": list(df.columns),
            "chart_type": chart_info.get('chart_type', 'none')
        })

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "📊 Welcome! Upload a CSV file and ask me anything about the data!"}
        ]

def plot_data(df, chart_info):
    try:
        chart_type = chart_info.get('chart_type', 'none')
        x = chart_info.get('x_column')
        y = chart_info.get('y_column')
        color = chart_info.get('color_column')

        if chart_type == 'none':
            return None

        if color == 'none':
            color = None  # No color grouping

        if chart_type == 'bar':
            fig = px.bar(df, x=x, y=y, color=color, barmode='group')
        elif chart_type == 'line':
            fig = px.line(df, x=x, y=y, color=color)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x, y=y, color=color)
        elif chart_type == 'pie':
            fig = px.pie(df, names=x, values=y)
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x, color=color)
        else:
            return None

        fig.update_layout(height=500)
        return fig

    except Exception as e:
        st.error(f"Plotting error: {str(e)}")
        return None


def process_question(question, df, assistant):
    # Extract year from the question (e.g., "2016" from "show me the data for 2016")
    year_match = re.search(r'\b(20\d{2})\b', question)
    year = int(year_match.group(1)) if year_match else None
    
    # Filter the dataframe for the requested year (if provided)
    if year:
        df = df[df['date'].dt.year == year]

    # Analyze data and suggest visualization
    chart_info = assistant.suggest_visualization(df, question)
    
    # Generate natural language summary
    summary = assistant.generate_insight(df, question, chart_info)
    
    # Create visualization
    fig = plot_data(df, chart_info)
    
    return {
        "summary": summary,
        "chart_info": chart_info,
        "figure": fig
    }


def convert_million_format(value):
    if isinstance(value, str):
        match = re.match(r"([\d.]+)([MmKk]?)", value)
        if match:
            num = float(match.group(1))
            suffix = match.group(2).lower()
            if suffix == 'm':
                return num * 1000000
            elif suffix == 'k':
                return num * 1000
            else:
                return num
    return value  # Return original value if no match    

def main():
    st.set_page_config(page_title="SCAP on KAI", page_icon="")
    initialize_session_state()

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    assistant = DataAnalyzerAssistant(model)

    st.title("SCAP on KAI")

    # File upload section
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Rename the timestamp column to "date"
            df.rename(columns={"Time-null": "date"}, inplace=True)
            
            # Convert 'date' column to datetime objects with the correct format
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce')
            
            # Check for conversion issues
            if df['date'].isnull().any():
                st.warning("Some dates could not be parsed. Check your date format.")
            
            # Convert million/thousand format columns (if needed)
            for col in df.columns:
                try:
                    df[col] = df[col].apply(convert_million_format)
                except (TypeError, AttributeError):
                    pass

            st.session_state.df = df
            st.success(f"Uploaded {uploaded_file.name} with {len(df)} rows")
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV: {e}")
            return
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return
        
    # Display chat history
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            if isinstance(content, dict):
                st.markdown(content.get("summary", ""))
                if content.get("figure"):
                    # Use index-based key for unique identification
                    st.plotly_chart(content["figure"], key=f"chart_{index}")
            else:
                st.write(content)

    # Chat input
    if "df" not in st.session_state:
        st.info("Please upload a CSV file to begin analysis")
        return

    df = st.session_state.df
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = process_question(prompt, df, assistant)
            
            st.markdown(response.get("summary", ""))
            if response.get("figure"):
                # Generate unique key based on message count
                chart_key = f"chart_{len(st.session_state.messages)}"
                st.plotly_chart(response["figure"], key=chart_key)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })


if __name__ == "__main__":
    main()
