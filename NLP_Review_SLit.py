# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import matplotlib.pyplot as plt
from googletrans import Translator
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

# Download NLTK resources if not present
nltk.download('vader_lexicon', quiet=True)

# ----------------------
# Dark Theme CSS
# ----------------------

# Main color variables (change these as you like)
APP_BG = "#222831"            # Background of main app
SIDEBAR_BG = "#00ADB5"        # Sidebar background
TEXT_COLOR = "#EEEEEE"        # Main text color
BUTTON_BG = "#00ADB5"         # Button background
BUTTON_TEXT = "#ffffff"       # Button text color
TABLE_HEADER_BG = "#1f2630"   # Table header background
TABLE_HEADER_TEXT = "#d7dbe0" # Table header text color
MARKDOWN_TEXT = "#00ADB5"     # Markdown/text color
HIGHLIGHT_BG = "#bbc5d7"      # Optional highlighted sections
HIGHLIGHT_TEXT = "#ffcc00"    # Optional highlight text color
FORM_LABEL_COLOR = "#1f2630"  # Form label color (selectbox, slider, input, etc.)

# CSS code using the variables
DARK_CSS = f"""
<style>
    /* Main app background and text */
    .stApp {{
        background-color: {APP_BG};
        color: {TEXT_COLOR};
    }}

    /* Sidebar */
    .stSidebar {{
        background-color: {SIDEBAR_BG};
    }}

    /* Buttons */
    .stButton>button {{
        background-color: {BUTTON_BG};
        color: {BUTTON_TEXT};
        font-weight: bold;
    }}

    /* Dataframe header */
    .dataframe thead th {{
        background-color: {TABLE_HEADER_BG};
        color: {TABLE_HEADER_TEXT};
    }}

    /* Markdown / normal text */
    .stMarkdown>div p {{
        color: {MARKDOWN_TEXT};
    }}

    /* Highlighted sections */
    .highlight {{
        background-color: {HIGHLIGHT_BG};
        color: {HIGHLIGHT_TEXT};
        padding: 5px;
        border-radius: 4px;
    }}

    /* Form labels (selectbox, multiselect, slider, text_input, text_area, number_input) */
    .stSelectbox label,
    .stCheckbox label,
    .stMultiselect label,
    .stSlider label,
    .stNumberInput label,
    .stTextInput label,
    .stTextArea label {{
        color: {FORM_LABEL_COLOR};
        font-weight: bold;
    }}

    /* Style Streamlit download buttons */
    div.stDownloadButton>button {{
        background-color: #00ADB5;
        color: #222831;
        font-weight: bold;
    }}
</style>
"""

# Apply dark CSS
st.markdown(DARK_CSS, unsafe_allow_html=True)

# Optional: set page config after CSS
st.set_page_config(page_title="NLP Sentiment Analysis Review", layout="wide")

##################### Sidebar: Logo  #####################

st.sidebar.image("myLogo.png", width='stretch')


# Initialize translator and sentiment analyzer
translator = Translator()
sia = SentimentIntensityAnalyzer()

# Streamlit page config
st.set_page_config(page_title="NLP Review Analyzer", layout="wide")


##################### Sidebar: file upload and options #####################

st.sidebar.header("Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])

# Small option to set a delay between translations to reduce throttling
sleep_between_calls = 0.05

# --- NEW OPTIONS ---
st.sidebar.header("Advanced options to speed up processing:")

# Option 1: Reviews are already in English
skip_translation = st.sidebar.checkbox(
    "Reviews are already in English (skip translation)", 
    value=False,
    help="Check this box if your reviews are already in English. This will skip the translation step and increase processing speed."
)

# Option 2: Disable autocorrect
disable_autocorrect = st.sidebar.checkbox(
    "Disable autocorrect",
    value=False,
    help="Check this box to skip TextBlob autocorrect and increase processing speed. Leave unchecked to apply autocorrect."
)


##################### Functions #####################

def review_cleaner(text):
     # Simple cleaning: handle NaN and strip spaces
    if pd.isna(text) or not str(text).strip():
        return ''
    return str(text).strip()

def trans_to_eng(text):
      # Translate text to English using googletrans.
      # Returns (translated_text, error_flag)
    if not text:
        return "", False
    if skip_translation:
        return text, False  # skip translation if user says it's already English
    try:
        translated = translator.translate(text, src='auto', dest='en')
        return translated.text, False
    except Exception:
        # If translation fails, return empty string and mark error True
        return "", True

def autocorrect_text(text):
     # Apply TextBlob.correct. If it fails, return original text
    if not text:
        return ""
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text

def analyze_sentiment(text):
     # Return compound score and sentiment label (Positive/Neutral/Negative)
    if not text:
        # treat empty text as Neutral with 0 compound
        return 0.0, "Neutral"
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return compound, label


##################### Main Page #####################

st.title("NLP Review Analyzer")
st.write("Upload a reviews file in the sidebar.")

if uploaded_file is None:
    st.info("Upload a CSV or Excel to begin.")
    st.stop()

# Read the uploaded file
try:
    if uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        df = pd.read_excel(uploaded_file)
    else:
        # assume CSV
        uploaded_file.seek(0)
        sample = uploaded_file.read(1024).decode()
        uploaded_file.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        df = pd.read_csv(uploaded_file, sep=dialect.delimiter, engine='python')
except Exception as e:
    st.error(f"Could not read uploaded file. Error: {e}")
    st.stop()

# Choose review column if multiple
cols = df.columns.tolist()
if len(cols) > 1:
    review_col = st.sidebar.selectbox("Select the review column", options=cols, index=0)
else:
    review_col = cols[0]

# Show first 50 rows on main page (as requested)
st.subheader("Initial dataset preview (first 50 rows)")
try:
    st.dataframe(df.head(50), use_container_width=True)
except Exception:
    # fallback if dataframe contains problematic types
    st.write(df.head(50))

# Button to start processing
start = st.button("Start processing")

if not start:
    st.info("Press 'Start processing' to run the combined NLP steps.")
    st.stop()


##################### Processing Loop #####################

st.subheader("Processing reviews")
# Prepare storage for processed columns
n = len(df)
cleaned_list = []
eng_list = []
translation_err_list = []
corrected_list = []
compound_list = []
sentiment_list = []

progress_bar = st.progress(0)
status_text = st.empty()

# Iterate rows once and do all steps inside
for i, raw in enumerate(df[review_col].astype(str).tolist()):
    # Clean Text
    cleaned = review_cleaner(raw)
    cleaned_list.append(cleaned)

    # Translate to English
    eng_text, trans_err = trans_to_eng(cleaned)
    eng_list.append(eng_text)
    translation_err_list.append(trans_err)

    # Autocorrect only if translation succeeded
    if trans_err or disable_autocorrect:
        corrected = eng_text  # skip autocorrect, keep translated text
    else:
        corrected = autocorrect_text(eng_text)
    corrected_list.append(corrected)

    # Sentiment on corrected text
    compound, label = analyze_sentiment(corrected)
    compound_list.append(compound)
    sentiment_list.append(label)

    # update progress and status
    progress = int(((i + 1) / n) * 100)
    progress_bar.progress((i + 1) / n)
    status_text.markdown(f'Processing row {i+1} of {n} — {progress}%</span>',unsafe_allow_html=True)
    if sleep_between_calls > 0:
        time.sleep(sleep_between_calls)

# Once done, clear status
status_text.markdown('Processing complete.</span>',unsafe_allow_html=True)
time.sleep(0.5)
progress_bar.empty()

# Build processed dataframe
df_proc = df.copy()
df_proc['cleaned_text'] = cleaned_list
df_proc['eng_text'] = eng_list
df_proc['translation_error'] = translation_err_list
df_proc['correct_text'] = corrected_list
df_proc['Compound'] = compound_list
df_proc['Sentiment'] = sentiment_list


##################### Display Result #####################

st.subheader("Processed results (first 200 rows)")
display_cols = ['cleaned_text', 'eng_text', 'correct_text', 'Compound', 'Sentiment', 'translation_error']
available_cols = [c for c in display_cols if c in df_proc.columns]
st.dataframe(df_proc[available_cols].head(200), use_container_width=True)

# Sentiment counts
st.subheader("Sentiment counts")
sentiment_counts = df_proc['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0).astype(int)
st.write(sentiment_counts)


##################### Bar Chart and Pie Chart #####################

st.subheader("Charts")
col1, col2 = st.columns(2)

# Bar chart
with col1:
    st.markdown("**Bar chart**")
    fig1, ax1 = plt.subplots(figsize=(5,3))
    ax1.bar(sentiment_counts.index, sentiment_counts.values)
    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("Count")
    ax1.set_title("Sentiment Counts")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig1)
    plt.close(fig1)

# Pie chart
with col2:
    st.markdown("**Pie chart**")
    fig2, ax2 = plt.subplots(figsize=(5,3))
    sizes = sentiment_counts.values.tolist()
    labels = sentiment_counts.index.tolist()
    if sum(sizes) == 0:
        ax2.text(0.5, 0.5, "No data to plot", ha='center', va='center')
    else:
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax2.axis('equal')
    st.pyplot(fig2)
    plt.close(fig2)