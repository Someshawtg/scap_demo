import streamlit as st
from PIL import Image
import os

# Check if the image file exists
if os.path.exists("awtg-new-logo.png"):
    image = Image.open("awtg-new-logo.png")
    
    # Create two columns with custom width ratios
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display the smaller image in the first column
        st.image(image, width=200)
        
    with col2:
        # Display the caption in the second column on one line
        st.markdown(
            "<h2 style='text-align: left; white-space: nowrap;'>Smart Analytics Dashboard</h2>",
            unsafe_allow_html=True
        )
else:
    st.error("Logo file not found!")



# Now import other modules and continue with the rest of your code...

import os
import re
import json
import logging
import pandas as pd
import numpy as np
import shutil
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "telecom-demo"

# Pinecone initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric="euclidean", 
                   spec=ServerlessSpec(cloud="aws", region="us-west-2"))

# Cache directory for embeddings
CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class AdvancedDataAnalyzer:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze telecom data and user query to:
1. Identify relevant metrics from: {metrics_list}
2. Determine appropriate visualization type (line, bar, heatmap, scatter, histogram, box)
3. Select axis columns based on query intent

Respond STRICTLY with valid JSON format:
{{
    "key_insights": ["list of technical insights"],
    "visualization": {{
        "type": "best_chart_type",
        "parameters": {{
            "x": "column_name", 
            "y": "column_name",
            "z": "column_name (if heatmap)",
            "color": "column_name (optional)",
            "title": "auto-generated title"
        }},
        "reasoning": "chart choice explanation"
    }}
}}"""),
            ("human", """Dataset sample (first 3 rows):
{sample}

User query: {query}

Current date: {current_date}""")
        ])

class EnhancedDataProcessor:
    def __init__(self):
        self.column_patterns = [
            (r'(rrc|radio resource control).*estab.*(att|attempt)', 'RRC.EstabAtt'),
            (r'(drb|data radio bearer).*estab.*(att|attempt)', 'DRB.EstabAtt'),
            (r'(active|connected).*(ue|users).*(dl|downlink)', 'Active.UEs.DL'),
            (r'(dl|downlink).*(throughput|thpt|tput)', 'DL.Throughput'),
            (r'(element|node|site|cell|enb|gnb)', 'Network.Element'),
            (r'(time|timestamp|date|hour)', 'MeasurementTime'),
            (r'(qos|quality of service).*level', 'QoS.Level'),
            (r'(latency|delay)', 'Latency'),
            (r'(throughput|thpt|tput)', 'Throughput'),
            (r'(utilization|usage)', 'Utilization')
        ]

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean and standardize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        column_mapping = {}
        for col in df.columns:
            for pattern, target in self.column_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    column_mapping[col] = target
                    logger.info(f"Mapping column '{col}' to '{target}'")
                    break
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert temporal columns and extract features
        temporal_cols = [c for c in df.columns if c in ['measurementtime', 'date', 'timestamp']]
        if temporal_cols:
            primary_time_col = temporal_cols[0]
            df[primary_time_col] = pd.to_datetime(df[primary_time_col], errors='coerce')
            df['hour'] = df[primary_time_col].dt.hour
            df['dayofweek'] = df[primary_time_col].dt.day_name()
            df['week'] = df[primary_time_col].dt.isocalendar().week
            logger.info(f"Processed time column '{primary_time_col}' with new features: hour, dayofweek, week")
            
        # Fill missing values
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in df.columns:
            if df[col].isnull().any():
                fill_val = df[col].median() if col in numeric_cols else 'Unknown'
                df[col].fillna(fill_val, inplace=True)
                logger.info(f"Filling missing values in '{col}' with '{fill_val}'")
        return df

class InteractiveVisualizer:
    def __init__(self):
        self.chart_registry = {
            'line': px.line,
            'linechart': px.line,
            'bar': px.bar,
            'barchart': px.bar,
            'heatmap': px.density_heatmap,
            'scatter': px.scatter,
            'histogram': px.histogram,
            'box': px.box,
            'boxplot': px.box
        }

    def create_visualization(self, df: pd.DataFrame, chart_spec: dict) -> go.Figure:
        try:
            chart_type = chart_spec['chart_type'].lower().replace('_', '')
            params = chart_spec.get('parameters', {})
            logger.info(f"Creating {chart_type} chart with parameters: {params}")
            
            # Normalize chart type names
            chart_type = self._normalize_chart_type(chart_type)
            
            # Validate parameters
            validated_params = self._validate_parameters(df, chart_type, params)
            
            fig = self.chart_registry[chart_type](df, **validated_params)
            fig.update_layout(
                height=400,
                width = 1000,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )
            logger.info(f"Chart created successfully: {chart_type} with {validated_params}")
            return fig
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return self._error_figure(str(e))

    def _normalize_chart_type(self, chart_type: str) -> str:
        type_map = {
            'linechart': 'line',
            'barchart': 'bar',
            'boxplot': 'box'
        }
        normalized = type_map.get(chart_type, chart_type)
        logger.info(f"Normalized chart type: {chart_type} to {normalized}")
        return normalized

    def _validate_parameters(self, df, chart_type, params):
        requirements = {
            'line': ['x', 'y'],
            'bar': ['x', 'y'],
            'scatter': ['x', 'y'],
            'heatmap': ['x', 'y', 'z'],
            'histogram': ['x'],
            'box': ['y']
        }
        
        validated = {}
        for param in requirements.get(chart_type, []):
            if param in params and params[param] in df.columns:
                validated[param] = params[param]
            else:
                error_msg = f"Missing required parameter: {param}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        # Add optional parameters
        for param in ['color', 'facet_col', 'title']:
            if param in params:
                validated[param] = params[param]
        logger.info(f"Validated parameters for {chart_type}: {validated}")
        return validated

    def _error_figure(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization Error: {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

class TelecomAnalysisPipeline:
    def __init__(self):
        self.analyzer = AdvancedDataAnalyzer()
        self.processor = EnhancedDataProcessor()
        self.visualizer = InteractiveVisualizer()

    def process_query(self, df: pd.DataFrame, query: str) -> dict:
        logger.info(f"Received user query: {query}")
        if self._is_conversational_query(query):
            logger.info("Query identified as conversational; skipping full analysis.")
            return {'analysis': {'conversational': True}, 'visualizations': [], 'insights': []}
        
        # Process data and run through AI analysis
        processed_df = self.processor.process_dataframe(df.copy())
        logger.info("Data processing complete. Beginning AI analysis.")
        analysis_result = self._get_ai_analysis(processed_df, query)
        logger.info(f"AI analysis result: {analysis_result}")
        
        visualization = self._generate_visualization(processed_df, analysis_result, query)
        insights = self._generate_insights(processed_df, query)
        logger.info(f"Generated insights: {insights}")
        
        return {
            'processed_data': processed_df,
            'analysis': analysis_result,
            'visualizations': [visualization] if visualization else [],
            'insights': insights
        }

    def _generate_visualization(self, df, analysis, query):
        try:
            vis_config = analysis.get('visualization', {})
            chart_type = vis_config.get('type', 'line').lower().replace('_', '')
            params = vis_config.get('parameters', {})
            logger.info(f"AI suggested chart type: {chart_type} with parameters: {params}")
            
            # Auto-detect parameters if missing
            default_params = {
                'title': f"{query} Analysis",
                'x': self._detect_axis(df, 'x', params.get('x')),
                'y': self._detect_axis(df, 'y', params.get('y')),
                'z': params.get('z', self._detect_secondary_metric(df))
            }
            logger.info(f"Default parameters before merge: {default_params}")
            
            # Merge AI suggestions with defaults
            final_params = {**default_params, **params}
            logger.info(f"Final parameters after merging AI suggestions: {final_params}")
            
            # Clean parameters: keep only those columns that exist in df
            final_params = {k: v for k, v in final_params.items() if v in df.columns}
            logger.info(f"Cleaned final parameters: {final_params}")
            
            # Log the AI's reasoning behind the chart selection
            reasoning = vis_config.get('reasoning', 'No reasoning provided')
            logger.info(f"Visualization reasoning from AI: {reasoning}")
            
            return self.visualizer.create_visualization(
                df,
                {'chart_type': chart_type, 'parameters': final_params}
            )
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            return None

    def _detect_axis(self, df, axis_type, suggested):
        if suggested and suggested in df.columns:
            logger.info(f"Using suggested {axis_type}-axis: {suggested}")
            return suggested
        if axis_type == 'x':
            detected = self._detect_time_column(df) or 'index'
            logger.info(f"Auto-detected {axis_type}-axis: {detected}")
            return detected
        detected = self._detect_primary_metric(df) or df.columns[-1]
        logger.info(f"Auto-detected {axis_type}-axis: {detected}")
        return detected

    def _detect_time_column(self, df):
        time_cols = ['measurementtime', 'timestamp', 'date']
        # Create a mapping: lower-case -> actual column name
        col_mapping = {col.lower(): col for col in df.columns}
        for t in time_cols:
            if t in col_mapping:
                logger.info(f"Detected time column: {col_mapping[t]}")
                return col_mapping[t]
        logger.info("Detected time column: None")
        return None

    def _detect_primary_metric(self, df):
        metrics = ['dl.throughput', 'rrc.estabatt', 'drb.estabatt', 'latency']
        for m in metrics:
            # Do a case-insensitive check
            for col in df.columns:
                if col.lower() == m:
                    logger.info(f"Detected primary metric for y-axis: {col}")
                    return col
        logger.info("Detected primary metric for y-axis: None")
        return None

    def _detect_secondary_metric(self, df):
        primary = self._detect_primary_metric(df)
        for col in df.columns:
            if col != primary:
                logger.info(f"Detected secondary metric: {col}")
                return col
        return None

    def _get_ai_analysis(self, df, query):
        try:
            chain = self.analyzer.analysis_prompt | self.analyzer.model | StrOutputParser()
            response = chain.invoke({
                "sample": df.head(3).to_string(),
                "query": query,
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "metrics_list": ", ".join(df.columns.tolist())
            })
            logger.info(f"Raw AI response: {response}")
            # Clean JSON response
            json_str = re.sub(r'(?i)^[^{]*', '', response)  # Remove non-JSON prefixes
            json_str = re.sub(r'[^}]*$', '', json_str)      # Remove non-JSON suffixes
            
            analysis = json.loads(json_str)
            logger.info(f"Parsed AI analysis: {analysis}")
            return analysis
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed. Raw response: {response}")
            return {"error": "Analysis failed", "raw_response": response}
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"error": str(e)}

    def _generate_insights(self, df: pd.DataFrame, query: str) -> list:
        insights = []
        time_col = self._detect_time_column(df)
        
        if time_col:
            time_range = f"from {df[time_col].min()} to {df[time_col].max()}"
            insights.append(f"Data covers {time_range}")
            logger.info(f"Insight: Data covers {time_range}")
            
        if 'RRC.EstabAtt' in df.columns:
            rrc_mean = df['RRC.EstabAtt'].mean()
            rrc_max = df['RRC.EstabAtt'].max()
            rrc_min = df['RRC.EstabAtt'].min()
            insight = f"RRC Attempts: Avg {rrc_mean:.1f} (Max {rrc_max}, Min {rrc_min})"
            insights.append(insight)
            logger.info(f"Insight: {insight}")
            
        if 'DL.Throughput' in df.columns:
            dl_mean = df['DL.Throughput'].mean()
            insight = f"Downlink Throughput: {dl_mean:.2f} Mbps average"
            insights.append(insight)
            logger.info(f"Insight: {insight}")
            
        return insights

    def _is_conversational_query(self, query: str) -> bool:
        # Use regex with word boundaries to avoid matching substrings like "hi" in "histogram"
        greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "how are you", "what's up"}
        query_lower = query.lower().strip()
        for greeting in greetings:
            if re.search(r'\b' + re.escape(greeting) + r'\b', query_lower):
                logger.info(f"Found conversational greeting '{greeting}' in query.")
                return True
        return False

class TelecomAnalyticsUI:
    def __init__(self):
        self.pipeline = TelecomAnalysisPipeline()
        self._init_session_state()

    def _init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "processed_df" not in st.session_state:
            st.session_state.processed_df = None
        if "current_query" not in st.session_state:
            st.session_state.current_query = None

    def render_interface(self):
        
        #st.title("📈 AWTG Smart Analytics Dashboard")
        self._display_messages()
        
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload Network Data", type=["csv", "xlsx", "parquet"])
            if uploaded_file:
                self._handle_file_upload(uploaded_file)

        if prompt := st.chat_input("Ask about network performance..."):
            self._process_query(prompt)
            st.rerun()

    def _handle_file_upload(self, file):
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            elif file.name.endswith('.parquet'):
                df = pd.read_parquet(file)
            else:
                raise ValueError("Unsupported file format")
                
            st.session_state.processed_df = self.pipeline.processor.process_dataframe(df)
            st.success(f"Successfully loaded {len(df)} records")
            logger.info(f"File '{file.name}' uploaded successfully with {len(df)} records")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            logger.error(f"Error loading file: {str(e)}")

    def _process_query(self, query):
        try:
            st.session_state.current_query = query
            logger.info(f"User submitted query: {query}")
            if st.session_state.processed_df is None:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ Please upload a dataset first",
                    "visualizations": []
                })
                logger.info("No dataset found. Prompting user to upload a dataset.")
                return

            result = self.pipeline.process_query(st.session_state.processed_df, query)
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "visualizations": []
            })
            st.session_state.messages.append(self._format_response(result))
            logger.info("Completed processing the query and generated analysis and visualization.")
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            logger.error(f"Processing error: {str(e)}")

    def _format_response(self, result):
        if result.get('analysis', {}).get('conversational'):
            logger.info("Formatting conversational response.")
            return {
                "role": "assistant",
                "content": self._format_conversational_response(),
                "visualizations": []
            }
            
        content = ["## Analysis Results"]
        
        if result.get('analysis', {}).get('error'):
            content.extend([
                "⚠️ **Analysis Limitations**",
                "- Could not fully interpret the query",
                "- Insufficient data for complete analysis",
                f"Raw AI response: ```{result['analysis'].get('raw_response', '')}```"
            ])
        else:
            if result.get('insights'):
                content.append("### Key Metrics")
                content.extend(f"- {insight}" for insight in result['insights'])
                
            if 'visualization' in result.get('analysis', {}):
                content.append("\n### Chart Selection Reasoning")
                content.append(result['analysis']['visualization']['reasoning'])
                logger.info(f"Chart Selection Reasoning: {result['analysis']['visualization']['reasoning']}")

        if result.get('visualizations'):
            content.append("\n### Generated Visualizations")
        else:
            content.append("\n🚫 No visualizations generated - try:")
            content.append("- Specifying time ranges")
            content.append("- Comparing specific metrics")
            content.append("- Checking data column names")

        return {
            "role": "assistant",
            "content": "\n".join(content),
            "visualizations": result.get("visualizations", [])
        }

    def _format_conversational_response(self):
        greeting = "Good morning" if 5 <= datetime.now().hour < 12 else "Good afternoon" if 12 <= datetime.now().hour < 17 else "Good evening"
        content = [
            f"🌟 **{greeting}! Welcome to AWTG Analytics**",
            "How can I assist you today?",
            "**Try these sample queries:**",
            "- 'Show histogram of RRC attempt frequencies'",
            "- 'Show relationship between active UEs and throughput",
            "- 'Show throughput trends from last week'",
            "- 'Can you generate a histogram showing the distribution of DL UE Throughput Mbps?",
            "- 'Heatmap of network activity by hour and cell'",
            "**Quick Start Guide:**",
            "1. Upload your network data file",
            "2. Ask questions about specific metrics",
            "3. Explore interactive visualizations",
            "Need help? Just ask!"
        ]
        return "\n".join(content)

    def _display_messages(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("visualizations", []):
                    self._render_visualizations(msg["visualizations"])
                elif msg["role"] == "assistant":
                    st.info("No graphical representation available")

    def _render_visualizations(self, figures):
        cols = st.columns(2)
        for idx, fig in enumerate(figures):
            with cols[idx % 2]:
                if fig and isinstance(fig, go.Figure):
                    st.plotly_chart(fig, use_container_width=True, 
                                      key=f"chart_{hash(fig.to_json())}_{time.time_ns()}")

if __name__ == "__main__":
    if os.path.exists("embedding_cache"):
        shutil.rmtree("embedding_cache")
    ui = TelecomAnalyticsUI()
    ui.render_interface()
