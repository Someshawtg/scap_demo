import os
import re
import json
import uuid
import time
import shutil
import hashlib
import pickle
import logging
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle Timestamp objects
class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        return super().default(o)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "telecom-demo"

# Pinecone initialization
logger.info("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes()
existing_index_names = [index["name"] for index in existing_indexes]
if PINECONE_INDEX_NAME not in existing_index_names:
    logger.info(f"Index '{PINECONE_INDEX_NAME}' not found. Creating new index.")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
else:
    logger.info(f"Index '{PINECONE_INDEX_NAME}' already exists.")

# Cache directory for embeddings
CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
logger.debug(f"Cache directory set to: {CACHE_DIR}")

def get_cache_key(texts):
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode())
    key = os.path.join(CACHE_DIR, f"{hasher.hexdigest()}.pkl")
    logger.debug(f"Generated cache key: {key}")
    return key

def load_cached_embeddings(cache_key):
    if os.path.exists(cache_key):
        logger.debug(f"Loading cached embeddings from: {cache_key}")
        with open(cache_key, "rb") as f:
            return pickle.load(f)
    logger.debug("No cached embeddings found.")
    return None

def save_embeddings_to_cache(cache_key, embeddings):
    logger.debug(f"Saving embeddings to cache: {cache_key}")
    with open(cache_key, "wb") as f:
        pickle.dump(embeddings, f)

class AdvancedDataAnalyzer:
    def __init__(self):
        logger.info("Initializing AdvancedDataAnalyzer...")
        self.model = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, api_key=OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        self.vector_store = None
        self._setup_prompts()

    def _setup_prompts(self):
        logger.info("Setting up analysis prompts...")
        self.analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze telecom data and user query to:
1. Identify relevant metrics from: {metrics_list}
2. Determine appropriate visualization types
3. Select axis columns based on query intent

Respond with JSON format:
{{
    "key_insights": ["list of technical insights"],
    "visualization": {{
        "type": "best_chart_type",
        "parameters": {{
            "x": "column_name", 
            "y": "column_name",
            "z": "column_name (if heatmap)",
            "title": "auto-generated title based on query"
        }},
        "reasoning": "explanation of chart choice"
    }}
}}"""),
    ("human", """Dataset sample (first 3 rows):
{sample}

User query: {query}

Current date: {current_date}""")
])

        logger.debug("Analysis prompt setup complete.")

    def start_background_embedding(self, df):
        logger.info("Starting background embedding generation...")
        texts = df.astype(str).values.flatten().tolist()
        cache_key = get_cache_key(texts)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._generate_embeddings, texts, cache_key)
            st.session_state.embedding_future = future

    def _generate_embeddings(self, texts, cache_key):
        try:
            cached_embeddings = load_cached_embeddings(cache_key)
            if cached_embeddings:
                logger.info("Cached embeddings found; using them.")
                return cached_embeddings
            logger.info("No cached embeddings; generating new embeddings.")
            batches = [texts[i:i+200] for i in range(0, len(texts), 200)]
            embeddings = []
            for idx, batch in enumerate(batches):
                logger.debug(f"Embedding batch {idx+1}/{len(batches)} with {len(batch)} texts.")
                embeddings.extend(self.embeddings.embed_documents(batch))
            save_embeddings_to_cache(cache_key, embeddings)
            self.vector_store = PineconeVectorStore.from_texts(
                texts,
                index_name=PINECONE_INDEX_NAME,
                embedding=self.embeddings
            )
            logger.info("Embedding generation complete.")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

# Cell 1: EnhancedDataProcessor
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
        """Dynamically process dataframe with intelligent column detection"""
        # Standardize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Dynamic column mapping
        column_mapping = {}
        for col in df.columns:
            for pattern, target in self.column_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    if col not in column_mapping:  # First match wins
                        column_mapping[col] = target
                        break
                        
        df.rename(columns=column_mapping, inplace=True)
        
        # Auto-detect temporal columns
        temporal_cols = [c for c in df.columns if c in ['measurementtime', 'date', 'timestamp']]
        if temporal_cols:
            primary_time_col = temporal_cols[0]
            df[primary_time_col] = pd.to_datetime(df[primary_time_col], errors='coerce')
            df['hour'] = df[primary_time_col].dt.hour
            df['dayofweek'] = df[primary_time_col].dt.day_name()
            df['week'] = df[primary_time_col].dt.isocalendar().week
            
        # Auto-detect numeric columns
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Auto-handle missing values
        for col in df.columns:
            if df[col].isnull().any():
                if col in self.numeric_cols:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
                    
        return df

# Cell 3: InteractiveVisualizer
class InteractiveVisualizer:
    def __init__(self):
        self.chart_registry = {
            'line': self._create_line_chart,
            'bar': self._create_bar_chart,
            'heatmap': self._create_heatmap,
            'scatter': self._create_scatter,
            'histogram': self._create_histogram,
            'box': self._create_box_plot
        }

    def create_visualization(self, df: pd.DataFrame, chart_spec: dict) -> go.Figure:
        try:
            chart_type = chart_spec['chart_type']
            params = chart_spec.get('parameters', {})
            
            # Validate required parameters
            required_params = {
                'line': ['x', 'y'],
                'bar': ['x', 'y'],
                'scatter': ['x', 'y'],
                'heatmap': ['x', 'y', 'z'],
                'histogram': ['x'],
                'box': ['y']
            }.get(chart_type, [])
            
            for param in required_params:
                if param not in params or params[param] not in df.columns:
                    raise ValueError(f"Missing required parameter: {param}")
            
            # Create visualization
            fig = self.chart_registry[chart_type](df, params)
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            return fig
            
        except Exception as e:
            return self._create_error_visualization(str(e))

    def _is_temporal(self, df, col):
        return pd.api.types.is_datetime64_any_dtype(df[col]) if col in df.columns else False

    def _create_line_chart(self, df, params):
        return px.line(df, x=params['x'], y=params['y'], 
                      title=params.get('title', 'Temporal Analysis'))

    def _create_bar_chart(self, df, params):
        return px.bar(df, x=params['x'], y=params['y'],
                     title=params.get('title', 'Comparative Analysis'))

    def _create_heatmap(self, df, params):
        return px.density_heatmap(df, x=params['x'], y=params['y'], z=params['z'],
                                 title=params.get('title', 'Correlation Analysis'))

    def _create_scatter(self, df, params):
        return px.scatter(df, x=params['x'], y=params['y'],
                         title=params.get('title', 'Scatter Analysis'))

    def _create_histogram(self, df, params):
        return px.histogram(df, x=params['x'],
                          title=params.get('title', 'Distribution Analysis'))

    def _create_box_plot(self, df, params):
        return px.box(df, y=params['y'],
                     title=params.get('title', 'Statistical Distribution'))

    def _create_error_visualization(self, error_msg):
        fig = go.Figure()
        fig.add_annotation(text=f"Visualization Error: {error_msg}",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

# Cell 2: TelecomAnalysisPipeline
class TelecomAnalysisPipeline:
    def __init__(self):
        self.analyzer = AdvancedDataAnalyzer()
        self.processor = EnhancedDataProcessor() 
        self.chart_requirements = {
            'line': {'x_type': 'temporal', 'y_type': 'numeric'},
            'bar': {'x_type': 'categorical', 'y_type': 'numeric'},
            'heatmap': {'x_type': 'categorical', 'y_type': 'categorical', 'z_type': 'numeric'},
            'scatter': {'x_type': 'numeric', 'y_type': 'numeric'},
            'histogram': {'x_type': 'numeric'},
            'box': {'y_type': 'numeric'}
        }


    def _create_error_response(self, error: Exception) -> dict:
        """Create standardized error response"""
        return {
            'processed_data': None,
            'analysis': {
                'error': True,
                'message': str(error),
                'visualization': {'types': [], 'parameters': {}}
            },
            'visualizations': [],
            'insights': []
        }

    def _generate_insights(self, df: pd.DataFrame, query: str) -> list:
        """Generate query-aware insights"""
        insights = []
        
        # Example insight generation
        if 'RRC.EstabAtt' in df.columns:
            insights.append(
                f"RRC Attempts: Avg {df['RRC.EstabAtt'].mean():.1f} "
                f"(Max {df['RRC.EstabAtt'].max()}, Min {df['RRC.EstabAtt'].min()})"
            )
            
        if 'DL.Throughput' in df.columns:
            insights.append(
                f"Downlink Throughput: {df['DL.Throughput'].mean():.2f} Mbps average"
            )
            
        return insights

    def _auto_detect_columns(self, df):
        """Dynamically detect column types and metrics"""
        column_types = {}
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = 'temporal'
            elif pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = 'numeric'
            else:
                column_types[col] = 'categorical'
                
        return {
            'temporal': [c for c, t in column_types.items() if t == 'temporal'],
            'numeric': [c for c, t in column_types.items() if t == 'numeric'],
            'categorical': [c for c, t in column_types.items() if t == 'categorical']
        }

    def _generate_dynamic_specs(self, df):
        """Generate visualization suggestions based on data characteristics"""
        column_types = self._auto_detect_columns(df)
        specs = []
        
        # Line/Time Series Charts
        if column_types['temporal'] and column_types['numeric']:
            specs.append({
                'chart_type': 'line',
                'parameters': {
                    'x': column_types['temporal'][0],
                    'y': column_types['numeric'][0],
                    'title': f"{column_types['numeric'][0]} Over Time"
                }
            })
            
        # Bar Charts
        if column_types['categorical'] and column_types['numeric']:
            specs.append({
                'chart_type': 'bar',
                'parameters': {
                    'x': column_types['categorical'][0],
                    'y': column_types['numeric'][0],
                    'title': f"{column_types['numeric'][0]} Distribution"
                }
            })
            
        # Heatmaps
        if len(column_types['categorical']) >= 2 and column_types['numeric']:
            specs.append({
                'chart_type': 'heatmap',
                'parameters': {
                    'x': column_types['categorical'][0],
                    'y': column_types['categorical'][1],
                    'z': column_types['numeric'][0],
                    'title': "Activity Correlation"
                }
            })
            
        # Scatter Plots
        if len(column_types['numeric']) >= 2:
            specs.append({
                'chart_type': 'scatter',
                'parameters': {
                    'x': column_types['numeric'][0],
                    'y': column_types['numeric'][1],
                    'title': "Metric Correlation"
                }
            })
            
        return specs

    def process_query(self, df: pd.DataFrame, query: str) -> dict:
        # First check for conversational queries
        if self._is_conversational_query(query):
            return self._handle_conversational_query(query)
            
        # Proceed with technical analysis for other queries
        processed_df = self.processor.process_dataframe(df.copy())
        analysis_result = self._get_ai_analysis(processed_df, query)
        visualization = self._generate_visualization(processed_df, analysis_result, query)
        
        return {
            'processed_data': processed_df,
            'analysis': analysis_result,
            'visualizations': [visualization] if visualization else [],
            'insights': self._generate_insights(processed_df, query)
        }

    def _is_conversational_query(self, query: str) -> bool:
        query = query.lower().strip()
        greetings = {"hello", "hi", "hey", "good morning", "good afternoon", 
                   "how are you", "what's up"}
        return any(greeting in query for greeting in greetings)

    def _handle_conversational_query(self, query: str) -> dict:
        return {
            'processed_data': None,
            'analysis': {'conversational': True},
            'visualizations': [],
            'insights': []
        }
    

    def _generate_visualization(self, df, analysis, query):
        """Enhanced visualization generator with query context"""
        try:
            chart_type = analysis.get('visualization', {}).get('type', 'line')
            params = analysis.get('visualization', {}).get('parameters', {})
            
            # Ensure required parameters exist
            base_params = {
                'title': f"{query} Analysis" if query else "Network Analysis",
                'x': 'MeasurementTime',
                'y': 'DL.Throughput'
            }
            
            return InteractiveVisualizer().create_visualization(
                df,
                {'chart_type': chart_type, 'parameters': {**base_params, **params}}
            )
        except KeyError as e:
            logger.error(f"Missing visualization parameter: {str(e)}")
            return None


    def _get_ai_analysis(self, df, query):
        sample_data = df.head(3).to_string()
        metrics = ", ".join(df.columns.tolist())
        
        chain = self.analyzer.analysis_prompt | self.analyzer.model | StrOutputParser()
        response = chain.invoke({
            "sample": sample_data,
            "query": query,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "metrics_list": metrics
        })
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse AI response")
            return {"error": "Analysis failed"}

    def _generate_ai_visualization(self, df, analysis, query):  # Add query parameter
        try:
            if "visualization" not in analysis:
                return None
                
            chart_type = analysis["visualization"]["type"]
            params = analysis["visualization"]["parameters"]
            
            # Validate columns exist in dataframe
            valid_params = {}
            for param in ['x', 'y', 'z']:
                if param in params and params[param] in df.columns:
                    valid_params[param] = params[param]
            
            # Add title from analysis or use query-based title
            valid_params['title'] = params.get(
                'title', 
                f"{query} Analysis" if query else "Network Performance Analysis"
            )
            
            return InteractiveVisualizer().create_visualization(
                df, 
                {'chart_type': chart_type, 'parameters': valid_params}
            )
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            return None

    def _generate_data_driven_insights(self, df):
        insights = []
        if 'RRC.EstabAtt' in df.columns:
            insights.append(f"RRC Attempts: Max {df['RRC.EstabAtt'].max()}, Min {df['RRC.EstabAtt'].min()}")
        if 'DL.Throughput' in df.columns:
            insights.append(f"Throughput Avg: {df['DL.Throughput'].mean():.2f} Mbps")
        return insights


# Cell 4: TelecomAnalyticsUI
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
        st.set_page_config(page_title="Telecom Analytics", layout="wide")
        st.title("📈 Dynamic Network Analytics Dashboard")
        
        # Immediate message display
        self._display_messages()
        
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload Network Data", 
                                           type=["csv", "xlsx", "parquet"])
            if uploaded_file:
                self._handle_file_upload(uploaded_file)

        self._handle_user_input()

    def _display_messages(self):
        # Display all messages except current processing
        for msg in st.session_state.messages:
            self._render_message(msg)
        
        # Show current processing state
        if st.session_state.current_query:
            with st.chat_message("user"):
                st.markdown(st.session_state.current_query)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing network patterns..."):
                    st.empty()

    def _handle_user_input(self):
        if prompt := st.chat_input("Ask about network performance..."):
            self._process_query(prompt)
            st.rerun()

    def _process_query(self, query):
        try:
            # Store current query separately
            st.session_state.current_query = query
            
            if st.session_state.processed_df is None:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ Please upload a dataset first",
                    "visualizations": []
                })
                return

            # Process query
            result = self.pipeline.process_query(
                st.session_state.processed_df,
                query
            )
            
            # Atomic update of messages
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "visualizations": []
            })
            st.session_state.messages.append(
                self._format_response(result)
            )

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        finally:
            st.session_state.current_query = None

    def _format_response(self, result):
        # Handle conversational responses first
        if result.get('analysis', {}).get('conversational'):
            return self._format_conversational_response()
        
        content = []
        visualizations = result.get("visualizations", [])

        # Error handling
        if result.get('analysis', {}).get('error'):
            content.append("⚠️ **Analysis Limitations**")
            content.append("Some metrics could not be analyzed due to:")
            content.append("- Missing required data columns")
            content.append("- Insufficient temporal coverage")
        else:
            # Only show technical headers when there's actual technical content
            if result.get('insights') or visualizations:
                content.append("## Network Analysis Report")
            
            if result.get('insights'):
                content.append("### Key Findings")
                content.extend(f"- {insight}" for insight in result['insights'])
            elif not visualizations:  # Only show "no patterns" if no visualizations either
                content.append("🔍 No significant patterns detected in current data scope")

        # Visualization status messaging
        if visualizations:
            content.append("\n### Data Visualizations")
        elif not result.get('analysis', {}).get('error'):  # Don't show suggestions if error
            content.append("\n🚫 No visualizations generated - try:")
            content.append("- Specifying time ranges")
            content.append("- Comparing specific metrics")
            content.append("- Checking data column names")

        return {
            "role": "assistant",
            "content": "\n".join(content),
            "visualizations": visualizations
        }

    def _format_conversational_response(self):
        # Add time-based greeting
        hour = datetime.now().hour
        greeting = (
            "Good morning" if 5 <= hour < 12 else
            "Good afternoon" if 12 <= hour < 17 else
            "Good evening"
        )
        
        content = [
            f"🌟 **{greeting}! Welcome to Network Analytics**",
            "",
            "How can I assist you with your telecom data today?",
            "",
            "**Try these sample queries:**",
            "- 'Compare latency between 5G and 4G nodes'",
            "- 'Show throughput trends from last week'",
            "- 'Visualize QoS level distribution by region'",
            "",
            "**Quick Start Guide:**",
            "1. Upload your network data file",
            "2. Ask questions about specific metrics",
            "3. Explore interactive visualizations",
            "",
            "Need help? Just ask!"
        ]
        
        return {
            "role": "assistant",
            "content": "\n".join(content),
            "visualizations": [],
            "conversational": True  # Add explicit marker
        }

    def _render_message(self, msg):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["visualizations"]:
                self._render_visualizations(msg["visualizations"])
            elif msg["role"] == "assistant":
                st.info("No graphical representation available")

    def _render_visualizations(self, figures):
        cols = st.columns(2)
        for idx, fig in enumerate(figures):
            with cols[idx % 2]:
                try:
                    if fig and isinstance(fig, go.Figure):
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"chart_{hash(fig.to_json())}_{time.time_ns()}"
                        )
                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")

    def _handle_file_upload(self, file):
        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            st.session_state.processed_df = self.pipeline.process_query(df, "")['processed_data']
            st.success(f"Successfully loaded {len(df)} records")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

if __name__ == "__main__":
    # Clear cache safely
    if os.path.exists("embedding_cache"):
        try:
            shutil.rmtree("embedding_cache")
            logger.info("Cleared embedding cache")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
    
    # Initialize fresh instance
    ui = TelecomAnalyticsUI()
    ui.render_interface()
