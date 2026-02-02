import streamlit as st
import re
import json
import os
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import andrews_curves
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# --- AUTOCOMPLETE ---
from streamlit_searchbox import st_searchbox

# -----------------------------------------------------------------------------
# 1. CORE LOGIC & DATA LOADING
# -----------------------------------------------------------------------------

# Try importing main logic
try:
    from main import get_cached_vectors
except ImportError:
    # Fallback mock for standalone testing
    def get_cached_vectors():
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        data = {
            'EDL_Company_nm': ['Apple', 'Microsoft', 'Tesla', 'Amazon', 'Google', 'Meta', 'Nvidia', 'Berkshire', 'Visa', 'JPM'],
            'AUM': [1000000, 2000000, 500000, 1500000, 1800000, 900000, 2200000, 800000, 600000, 1200000],
            'Region': ['US', 'US', 'US', 'EU', 'US', 'US', 'US', 'US', 'EU', 'US'],
            'Portfolio_Group_nm': ['Tech', 'Tech', 'Auto', 'Retail', 'Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Finance'],
            'Account_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        }
        df = pd.DataFrame(data)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(df['EDL_Company_nm'])
        return vectors, df, vectorizer

# -----------------------------------------------------------------------------
# 2. MODEL CONFIGURATION (LOCAL FOLDER IS: ./model)
# -----------------------------------------------------------------------------
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")  # <-- IMPORTANT: model (singular)

# Try importing Transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    torch = None

# Try importing Ollama
try:
    import ollama
    from ollama import ResponseError
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ResponseError = None

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Cache HF objects
hf_bundle = None  # will hold (tokenizer, model)


@st.cache_resource
def load_local_hf_model(model_path: str):
    """Load a local HF chat model (TinyLlama etc.) from ./model folder."""
    if not HF_AVAILABLE:
        return (None, None)

    if not os.path.exists(model_path):
        return (None, None)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # Llama-family models sometimes have no pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch and torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        model.eval()
        return (tokenizer, model)
    except Exception as e:
        print(f"Failed to load local model: {e}")
        return (None, None)


def extract_last_json_object(text: str):
    """Best-effort: pull the last {...} block."""
    if not text:
        return None
    # strip code fences
    cleaned = re.sub(r"^```json?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    # find JSON objects (non-greedy)
    matches = re.findall(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if matches:
        return matches[-1]
    # fallback: maybe it's already pure JSON
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    return None


def ask_llm(query: str, df_info: str, column_names: list) -> dict:
    """Use Local HF Model (TinyLlama) or Ollama to understand the query."""
    system_prompt = f"""You are a data analyst. Analyze this query about a dataset.
Dataset columns: {column_names}
Dataset info: {df_info}
User query: "{query}"

Respond with ONLY valid JSON:
{{
    "intent": "<intent type>", "n": <number>,
    "sort_column": "<col>", "sort_order": "desc|asc",
    "filter_column": "<col>", "filter_operator": "equals|contains|greater_than|less_than",
    "filter_value": "<val>", "chart_type": "bar|line|pie|scatter|histogram|box|violin|heatmap|correlation|none",
    "answer": "<explanation>"
}}"""

    # -------------------------
    # Local HF Strategy (TinyLlama)
    # -------------------------
    global hf_bundle
    if HF_AVAILABLE and os.path.exists(LOCAL_MODEL_PATH):
        if hf_bundle is None:
            with st.spinner("🚀 Loading local AI model..."):
                hf_bundle = load_local_hf_model(LOCAL_MODEL_PATH)

    if hf_bundle and hf_bundle[0] is not None and hf_bundle[1] is not None:
        tokenizer, model = hf_bundle
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            # KEY: use the model's chat template (TinyLlama chat works best with this)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.95,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            json_blob = extract_last_json_object(generated)
            if not json_blob:
                return {"_error": "hf_parse_failure", "details": "No JSON found in HF output."}

            return json.loads(json_blob)
        except Exception as e:
            return {"_error": "hf_failure", "details": str(e)}

    # -------------------------
    # Ollama Strategy
    # -------------------------
    if OLLAMA_AVAILABLE:
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": system_prompt}]
            )
            result_text = response["message"]["content"].strip()
            json_blob = extract_last_json_object(result_text) or result_text
            return json.loads(json_blob)
        except Exception as e:
            return {"_error": "ollama_failure", "details": str(e)}

    return {"_error": "no_llm_available", "details": "No local HF model found and Ollama not available."}


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_search_suggestions(search_term: str):
    all_suggestions = [
        "top 30 companies by aum violin correlation pie bar",
        "first 3 companies by aum",
        "top 10 companies by aum",
        "bottom 10 by aum",
        "pie chart",
        "bar chart",
        "violin chart",
        "correlation chart",
        "heatmap",
        "scatter plot",
    ]
    if not search_term:
        return []
    results = [s for s in all_suggestions if search_term.lower() in s.lower()]
    if search_term.strip() and search_term not in results:
        results.insert(0, search_term)
    return results


def get_company_column(dataframe, categorical_columns):
    """Best-effort detection of the company *name* column."""
    if dataframe is None or dataframe.empty:
        return None
    cols = list(dataframe.columns)

    def is_id_like(col: str) -> bool:
        cl = col.lower()
        return ('key' in cl or cl.endswith('_id') or cl.endswith('id') or cl.endswith('_key') or cl in {'id', 'key'})

    preferred_patterns = [r'^edl_company_nm$', r'^edl_company_name$', r'^company_nm$', r'^company_name$']
    for c in cols:
        if c in categorical_columns and not is_id_like(c):
            cl = c.lower()
            if any(re.match(p, cl) for p in preferred_patterns):
                return c
    for c in cols:
        if c in categorical_columns and not is_id_like(c):
            cl = c.lower()
            if 'company' in cl and ('name' in cl or cl.endswith('_nm') or '_nm_' in cl):
                return c
    for c in cols:
        if c in categorical_columns and not is_id_like(c):
            if 'company' in c.lower():
                return c
    return None


def reorder_columns(dataframe):
    if dataframe is None or dataframe.empty:
        return dataframe
    cols = list(dataframe.columns)
    company_col = None
    for col in cols:
        if 'edl_company_nm' in col.lower():
            company_col = col
            break
    if company_col and company_col in cols:
        cols.remove(company_col)
    final_cols = ([company_col] if company_col else []) + cols
    valid_cols = [c for c in final_cols if c in dataframe.columns]
    return dataframe[valid_cols]


def detect_chart_types(query_lower: str):
    detected = []
    chart_keywords = {
        'line': ['line chart', 'line graph', 'line plot'],
        'bar': ['bar chart', 'bar graph', 'barchart', ' bar '],
        'pie': ['pie chart', 'pie graph', ' pie '],
        'scatter': ['scatter plot', 'scatter chart'],
        'histogram': ['histogram', 'hist'],
        'heatmap': ['heatmap'],
        'correlation': ['correlation', 'corr', 'correlation chart'],
        'box': ['box plot', 'boxplot'],
        'violin': ['violin', 'violin chart'],
        'area': ['area chart'],
        'hexbin': ['hexbin']
    }

    for chart_type, keywords in chart_keywords.items():
        for keyword in keywords:
            if keyword in query_lower and chart_type not in detected:
                detected.append(chart_type)

    if not detected and any(word in query_lower for word in ['chart', 'graph', 'plot', 'visualize']):
        return ['bar']

    return detected


def infer_intent_from_text(query_lower: str) -> str:
    if not query_lower:
        return "general"
    top_words = ["top", "highest", "largest", "biggest", "first", "rank", "best"]
    bottom_words = ["bottom", "lowest", "smallest", "last", "worst"]
    if any(w in query_lower for w in top_words):
        return "top"
    if any(w in query_lower for w in bottom_words):
        return "bottom"
    if "show all" in query_lower or "all records" in query_lower:
        return "show_all"
    return "general"


def wants_company_aggregation(query_lower: str) -> bool:
    """If user says 'companies ... by aum' treat it as a grouped ranking."""
    if not query_lower:
        return False
    return (
        ("company" in query_lower or "companies" in query_lower) and
        ("aum" in query_lower) and
        (" by " in query_lower or "rank" in query_lower or "top" in query_lower or "first" in query_lower)
    )


# --------------------------
# NEW: chart normalization + dedupe
# --------------------------
def normalize_chart_type(x):
    """Return normalized chart type(s): str or list[str]."""
    if not x:
        return None
    if isinstance(x, list):
        out = []
        for i in x:
            n = normalize_chart_type(i)
            if isinstance(n, list):
                out.extend(n)
            elif n:
                out.append(n)
        return out

    s = str(x).strip().lower()
    aliases = {
        "bar chart": "bar",
        "bargraph": "bar",
        "barchart": "bar",
        "pie chart": "pie",
        "piegraph": "pie",
        "line chart": "line",
        "scatter plot": "scatter",
        "box plot": "box",
        "corr": "correlation",
        "correlation chart": "correlation",
        "corr chart": "correlation",
        "violin chart": "violin",
    }
    return aliases.get(s, s)


def dedupe_preserve_order(items):
    seen = set()
    out = []
    for it in items:
        if not it:
            continue
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def create_advanced_chart(result_df, chart_type, query_lower, numeric_cols, categorical_cols, aum_col=None):
    if result_df is None or result_df.empty:
        return None

    group_col = None
    company_col = get_company_column(result_df, result_df.columns.tolist())
    wants_company_grouping = any(kw in (query_lower or '') for kw in ['company', 'companies', 'edl_company_nm'])

    if wants_company_grouping and company_col:
        group_col = company_col
    if not group_col:
        for gc in result_df.columns:
            if 'portfolio' in gc.lower() and 'group' in gc.lower():
                group_col = gc
                break
    if not group_col:
        for gc in result_df.columns:
            if gc in categorical_cols:
                group_col = gc
                break

    val_col = aum_col if aum_col and aum_col in result_df.columns else (numeric_cols[0] if numeric_cols else None)
    available_numeric = [c for c in numeric_cols if c in result_df.columns]
    available_categorical = [c for c in categorical_cols if c in result_df.columns]

    try:
        if chart_type == 'line':
            if group_col and val_col:
                chart_df = result_df.groupby(group_col)[val_col].sum().reset_index().sort_values(val_col, ascending=False).head(20)
                return px.line(chart_df, x=group_col, y=val_col, markers=True, title=f"Line: {val_col} by {group_col}")

        elif chart_type == 'bar':
            if group_col and val_col:
                chart_df = result_df.groupby(group_col)[val_col].sum().reset_index().sort_values(val_col, ascending=False).head(20)
                return px.bar(chart_df, x=group_col, y=val_col, title=f"Bar: {val_col} by {group_col}")

        elif chart_type == 'pie':
            if group_col and val_col:
                chart_df = result_df.groupby(group_col)[val_col].sum().reset_index().sort_values(val_col, ascending=False).head(10)
                return px.pie(chart_df, values=val_col, names=group_col, title=f"Pie: {val_col} by {group_col}")

        elif chart_type == 'scatter':
            if len(available_numeric) >= 2:
                x, y = available_numeric[0], available_numeric[1]
                c = available_categorical[0] if available_categorical else None
                return px.scatter(result_df.head(500), x=x, y=y, color=c, title=f"Scatter: {x} vs {y}")
            elif group_col and val_col:
                return px.scatter(result_df.head(500), x=group_col, y=val_col, title=f"Scatter: {val_col} by {group_col}")

        elif chart_type == 'histogram':
            if val_col:
                return px.histogram(result_df, x=val_col, nbins=30, title=f"Histogram: {val_col}")

        elif chart_type == 'area':
            if group_col and val_col:
                chart_df = result_df.groupby(group_col)[val_col].sum().reset_index().sort_values(val_col, ascending=False).head(20)
                return px.area(chart_df, x=group_col, y=val_col, title=f"Area: {val_col} by {group_col}")

        elif chart_type == 'box':
            if val_col:
                c = available_categorical[0] if available_categorical else None
                return px.box(result_df.head(1000), y=val_col, x=c, color=c, title=f"Box Plot: {val_col}")

        elif chart_type == 'violin':
            if val_col:
                c = available_categorical[0] if available_categorical else None
                return px.violin(result_df.head(1000), y=val_col, x=c, color=c, box=True, title=f"Violin Plot: {val_col}")

        elif chart_type == 'heatmap':
            if len(available_categorical) >= 2 and val_col:
                pivot = result_df.pivot_table(
                    values=val_col,
                    index=available_categorical[0],
                    columns=available_categorical[1],
                    aggfunc='sum'
                ).fillna(0).iloc[:15, :15]
                return px.imshow(pivot, title=f"Heatmap: {val_col}", aspect='auto')
            elif len(available_numeric) >= 2:
                corr = result_df[available_numeric[:10]].corr()
                return px.imshow(corr, title="Correlation Matrix", text_auto='.2f', zmin=-1, zmax=1)

        elif chart_type == 'correlation':
            if len(available_numeric) >= 2:
                corr = result_df[available_numeric[:10]].corr()
                return px.imshow(corr, title="Correlation Matrix", text_auto='.2f', zmin=-1, zmax=1)

        else:
            if group_col and val_col:
                chart_df = result_df.groupby(group_col)[val_col].sum().reset_index().sort_values(val_col, ascending=False).head(20)
                return px.bar(chart_df, x=group_col, y=val_col, title=f"{val_col} by {group_col}")

    except Exception as e:
        st.warning(f"Chart error: {e}")
        return None

    return None


# -----------------------------------------------------------------------------
# 4. PAGE SETUP & STYLING
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Institutional AUM", page_icon="📊", layout="wide")

st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #f8fafc; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
        .stButton>button { border-radius: 0.5rem; border: 1px solid #e2e8f0; background-color: #ffffff; color: #1e293b; font-weight: 500; transition: all 0.2s; }
        .stButton>button:hover { border-color: #3b82f6; color: #3b82f6; background-color: #eff6ff; }
        .stButton>button[kind="primary"] { background-color: #3b82f6; color: white; border: none; }
        .stButton>button[kind="primary"]:hover { background-color: #2563eb; }
    </style>
""", unsafe_allow_html=True)

# Session State
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'last_result_df' not in st.session_state:
    st.session_state.last_result_df = None
if 'last_message' not in st.session_state:
    st.session_state.last_message = None
if 'generated_charts' not in st.session_state:
    st.session_state.generated_charts = []
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

# Load Data
with st.spinner("Loading data..."):
    vectors, df, vectorizer = get_cached_vectors()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# -----------------------------------------------------------------------------
# 5. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="mb-6">
            <h2 class="text-xl font-bold text-slate-800 flex items-center gap-2">
                <span>⚙️</span> Control Panel
            </h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="bg-white p-4 rounded-lg border border-slate-200 shadow-sm mb-6">', unsafe_allow_html=True)
    st.markdown('<h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">System Status</h3>',
                unsafe_allow_html=True)

    local_model_ok = HF_AVAILABLE and os.path.exists(LOCAL_MODEL_PATH)
    if local_model_ok:
        st.markdown(
            """<div class="flex items-center gap-2 p-2 bg-green-50 text-green-700 rounded-md border border-green-100 mb-2">
                <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                <span class="text-sm font-medium">Local HF Model Found (./model)</span>
            </div>""",
            unsafe_allow_html=True
        )
    elif OLLAMA_AVAILABLE:
        st.markdown(
            f"""<div class="flex items-center gap-2 p-2 bg-green-50 text-green-700 rounded-md border border-green-100 mb-2">
                <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                <span class="text-sm font-medium">Ollama Active ({OLLAMA_MODEL})</span>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div class="flex items-center gap-2 p-2 bg-blue-50 text-blue-700 rounded-md border border-blue-100">
                <span class="text-sm font-medium">Semantic Search Mode (No LLM)</span>
            </div>""",
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="bg-white p-4 rounded-lg border border-slate-200 shadow-sm mb-6">
        <div class="grid grid-cols-2 gap-4">
            <div class="text-center">
                <div class="text-2xl font-bold text-slate-700">{len(df)}</div>
                <div class="text-xs text-slate-400">Rows</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold text-slate-700">{len(df.columns)}</div>
                <div class="text-xs text-slate-400">Cols</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 px-1">📚 Reference</h3>',
                unsafe_allow_html=True)
    with st.expander("📋 Available Columns"):
        st.markdown("\n".join([f"- {col}" for col in df.columns]))

    with st.expander("📈 Chart Types"):
        st.markdown("""
        - Line, Bar, Pie
        - Scatter, Histogram
        - Area, Box, Violin
        - Heatmap, Correlation
        """)

    st.markdown('<div class="mt-6"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 px-1">⚡ Quick Actions</h3>',
                unsafe_allow_html=True)
    q1 = st.button("📅 AUM (Apr 2025)")
    q2 = st.button("🌲 Timber Template")
    q3 = st.button("🔒 Private Markets")
    q4 = st.button("💵 Currency: USD")

# -----------------------------------------------------------------------------
# 6. MAIN CONTENT
# -----------------------------------------------------------------------------

st.markdown("""
    <div class="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 mb-8">
        <div class="flex items-start gap-6">
            <div class="bg-blue-600 text-white p-4 rounded-xl text-3xl shadow-lg shadow-blue-200">📊</div>
            <div>
                <h1 class="text-3xl font-bold text-slate-800 m-0 leading-tight">Institutional AUM Data Explorer</h1>
                <p class="text-slate-500 mt-2 text-lg">Use natural language to query, analyze, and visualize your institutional data assets.</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

col_main, col_help = st.columns([3, 1])
with col_main:
    st.markdown('<div class="bg-white p-1 rounded-xl shadow-sm border border-slate-200">', unsafe_allow_html=True)
    selected_value = st_searchbox(
        get_search_suggestions,
        key="main_searchbox",
        label="",
        placeholder="🔍 Ask a question (e.g., 'Top 30 companies by AUM violin correlation pie bar')...",
        clear_on_submit=False
    )
    st.markdown('</div>', unsafe_allow_html=True)

    query = selected_value
    if q1:
        query = "Show me all AUM records for April 30, 2025"
    if q2:
        query = "What entries come from the MIM Timber & Agriculture Template source?"
    if q3:
        query = "Which records are marked as private markets?"
    if q4:
        query = "Show all accounts where the original currency is USD"

    col_spacer, col_btn = st.columns([4, 1])
    with col_btn:
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        submit_button = st.button("Run Analysis ➤", type="primary", use_container_width=True)

    if selected_value:
        submit_button = True

with col_help:
    st.markdown("""
        <div class="bg-blue-50 p-4 rounded-xl border border-blue-100 h-full">
            <h4 class="text-sm font-bold text-blue-800 mb-2">💡 Tips</h4>
            <ul class="text-xs text-blue-700 space-y-1 list-disc pl-4">
                <li>Try "Top 30 companies by AUM violin correlation pie bar"</li>
                <li>Ask for "pie chart, bar chart"</li>
                <li>Add "correlation chart"</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. PROCESSING
# -----------------------------------------------------------------------------
st.markdown("<div class='mb-8'></div>", unsafe_allow_html=True)

if query and submit_button:
    st.session_state.last_query = query
    st.session_state.generated_charts = []
    st.session_state.feedback_given = False
    query_lower = query.lower()

    aum_col = next((c for c in numeric_cols if 'aum' in c.lower()), numeric_cols[0] if numeric_cols else None)

    llm_response = None
    can_llm = (HF_AVAILABLE and os.path.exists(LOCAL_MODEL_PATH)) or OLLAMA_AVAILABLE
    if can_llm:
        df_info = f"Rows: {len(df)}, NumColsSample: {numeric_cols[:5]}, CatColsSample: {categorical_cols[:5]}"
        llm_response = ask_llm(query, df_info, list(df.columns))

    result_df = df.copy()
    intent_msg = "Keyword Search"

    if llm_response and not llm_response.get('_error'):
        intent = (llm_response.get('intent') or 'general').strip().lower()
        n = llm_response.get('n', 10)

        numbers = re.findall(r'\d+', query_lower)
        if numbers:
            try:
                n = int(numbers[0])
            except Exception:
                pass

        if intent in ["general", "unknown", "", None]:
            intent = infer_intent_from_text(query_lower)

        # Filter
        f_col = llm_response.get('filter_column')
        f_val = llm_response.get('filter_value')
        if f_col and f_val:
            matches = [c for c in df.columns if f_col.lower() in c.lower()]
            if matches:
                target_col = matches[0]
                clean_val = str(f_val)
                if f_col.lower() in clean_val.lower():
                    clean_val = re.sub(f_col, '', clean_val, flags=re.IGNORECASE).strip()
                clean_val = clean_val.strip("'").strip('"')
                if clean_val:
                    mask = result_df[target_col].astype(str).str.contains(clean_val, case=False, na=False)
                    if mask.any():
                        result_df = result_df[mask]

        # Sort col
        s_col = llm_response.get('sort_column')
        target_sort = None
        if s_col:
            matches = [c for c in df.columns if s_col.lower() in c.lower()]
            if matches:
                target_sort = matches[0]
        if not target_sort:
            target_sort = aum_col

        company_col = get_company_column(df, categorical_cols)
        do_company_rank = wants_company_aggregation(query_lower) and company_col and target_sort

        if do_company_rank:
            ranked = (
                result_df.groupby(company_col, dropna=False)[target_sort]
                .sum()
                .reset_index()
                .sort_values(target_sort, ascending=False)
                .head(n)
            )
            result_df = ranked
            intent_msg = f"AI Intent: Top {n} Companies by {target_sort} (grouped)"
        else:
            if intent == "top" and target_sort:
                result_df = result_df.nlargest(n, target_sort)
                intent_msg = f"AI Intent: Top {n} by {target_sort}"
            elif intent == "bottom" and target_sort:
                result_df = result_df.nsmallest(n, target_sort)
                intent_msg = f"AI Intent: Bottom {n} by {target_sort}"
            elif intent == "show_all":
                result_df = result_df.head(n)
                intent_msg = f"AI Intent: Showing first {n} rows"
            else:
                if any(w in query_lower for w in ["first", "top", "bottom", "last"]) or numbers:
                    result_df = result_df.head(n)
                    intent_msg = f"AI Intent: Limited to first {n} rows"
                else:
                    intent_msg = f"AI Intent: {intent.replace('_', ' ').title()}"

        # -------------------------
        # FIXED CHART COLLECTION (normalize + dedupe)
        # -------------------------
        ai_chart_raw = llm_response.get("chart_type")
        ai_norm = normalize_chart_type(ai_chart_raw)

        ai_list = []
        if isinstance(ai_norm, list):
            ai_list.extend(ai_norm)
        elif ai_norm:
            ai_list.append(ai_norm)

        ai_list = [c for c in ai_list if c and c != "none"]

        manual_list = [normalize_chart_type(c) for c in detect_chart_types(query_lower)]
        # normalize_chart_type returns str here because detect_chart_types returns str list
        manual_list = [c for c in manual_list if c and c != "none"]

        chart_types = dedupe_preserve_order(ai_list + manual_list)

    else:
        # Fallback logic (no LLM)
        numbers = re.findall(r'\d+', query_lower)
        n = int(numbers[0]) if numbers else 10

        if "companies" in query_lower and "aum" in query_lower and ("top" in query_lower or "first" in query_lower):
            company_col = get_company_column(df, categorical_cols)
            if company_col and aum_col:
                result_df = (
                    result_df.groupby(company_col, dropna=False)[aum_col]
                    .sum()
                    .reset_index()
                    .sort_values(aum_col, ascending=False)
                    .head(n)
                )
                intent_msg = f"Fallback: Top {n} Companies by {aum_col} (grouped)"
            else:
                result_df = result_df.head(n)
                intent_msg = f"Fallback: First {n} rows"
        elif 'top' in query_lower:
            result_df = result_df.nlargest(n, aum_col) if aum_col else result_df.head(n)
            intent_msg = f"Fallback: Top {n}"
        elif 'bottom' in query_lower:
            result_df = result_df.nsmallest(n, aum_col) if aum_col else result_df.tail(n)
            intent_msg = f"Fallback: Bottom {n}"
        else:
            q_vec = vectorizer.transform([query]).toarray()
            sims = cosine_similarity(q_vec, vectors).flatten()
            result_df = df.iloc[sims.argsort()[-n:][::-1]]
            intent_msg = f"Fallback: Semantic Search (Top {n})"

        # Charts in fallback (normalize + dedupe)
        manual_list = [normalize_chart_type(c) for c in detect_chart_types(query_lower)]
        manual_list = [c for c in manual_list if c and c != "none"]
        chart_types = dedupe_preserve_order(manual_list)

    # Generate charts ONCE per type
    for c_type in chart_types:
        fig = create_advanced_chart(result_df, c_type, query_lower, numeric_cols, categorical_cols, aum_col)
        if fig:
            st.session_state.generated_charts.append({"type": c_type, "fig": fig})

    st.session_state.last_result_df = reorder_columns(result_df)
    st.session_state.last_message = intent_msg

# -----------------------------------------------------------------------------
# 8. RESULTS DISPLAY
# -----------------------------------------------------------------------------
if st.session_state.last_query:
    st.markdown(f"""
        <div class="flex items-center justify-between mb-4">
            <h2 class="text-xl font-bold text-slate-800">
                Results for: <span class="text-blue-600">"{st.session_state.last_query}"</span>
            </h2>
            <div class="bg-blue-100 text-blue-800 text-xs font-bold px-2 py-1 rounded uppercase tracking-wide">
                {st.session_state.last_message or "Analysis"}
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.generated_charts:
        st.markdown("### 📈 Visual Analysis")
        for chart_obj in st.session_state.generated_charts:
            c_type_str = str(chart_obj.get("type", "chart"))
            st.markdown(f"**{c_type_str.title()} Chart**")
            st.plotly_chart(chart_obj["fig"], use_container_width=True)
            st.markdown("---")

    if st.session_state.last_result_df is not None:
        st.markdown(f"### 📋 Data Records ({len(st.session_state.last_result_df)})")
        st.dataframe(st.session_state.last_result_df, use_container_width=True)

    if not st.session_state.feedback_given:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([4, 2, 4])
        with c2:
            st.markdown("**Was this helpful?**")
            b1, b2 = st.columns(2)
            if b1.button("👍"):
                st.toast("Thanks!")
                st.session_state.feedback_given = True
            if b2.button("👎"):
                st.toast("Thanks!")
                st.session_state.feedback_given = True