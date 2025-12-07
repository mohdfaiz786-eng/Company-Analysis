

import os
import io
import tempfile
from datetime import datetime
from dateutil import parser

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


#set the Enviroment Variable :-
os.environ["PATH"] += os.pathsep + r"D:\Training\spi\Python-with-Datascience\my-softpro-project\softpro-Analytics\ffmpeg\bin"

# Import with Expectional Handling.
try:
    import whisper  # OpenAI Whisper
except Exception:
    whisper = None

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import wave
except Exception:
    VoskModel = None
    KaldiRecognizer = None

from transformers import pipeline

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Softpro Sentiment & Sales Insights", layout="wide")
st.title("Softpro Sentiment & Sales Insights")
st.caption("Audio + CRM logs → Transcripts → Sentiment → Insights → Recommendations")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Settings")
asr_engine = st.sidebar.selectbox("ASR Engine (Audio → Text)", ["Whisper", "Vosk (offline)"])
if asr_engine == "Whisper":
    whisper_size = st.sidebar.selectbox("Whisper model size", ["tiny", "base", "small", "medium"], index=1)
else:
    vosk_model_dir = st.sidebar.text_input("Vosk model directory (unzipped)", value="")

# st.sidebar.markdown("---")
# use_pretrained = st.sidebar.checkbox("Force Pretrained Sentiment (skip training even if labels exist)", value=False)

st.sidebar.markdown("---")
st.sidebar.write("**Export**")
save_intermediate = st.sidebar.checkbox("Save processed CSV", value=True)

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    if whisper is None:
        raise RuntimeError("Whisper not installed. pip install openai-whisper and ensure ffmpeg is present.")
    return whisper.load_model(model_size)

@st.cache_resource(show_spinner=False)
def load_vosk(model_dir: str):
    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError("Valid Vosk model directory not provided.")
    if VoskModel is None:
        raise RuntimeError("Vosk not installed. pip install vosk")
    return VoskModel(model_dir)

# High Frequency pipelines to be loaded 
# open source distilbert-base-uncased-finetuned-sst-2-english
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
# sentiment Analysis best model 
@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    # Fast, widely used binary sentiment model
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource(show_spinner=False)
def train_sklearn_sentiment(texts: pd.Series, labels: pd.Series):
    # labels expected as strings: positive/neutral/negative (case-insensitive is handled)
    y = labels.astype(str).str.lower().replace({
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
        "n": "negative",
        "p": "positive"
    })
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    report = classification_report(y_test, y_pred, output_dict=False)
    return vectorizer, clf, report


def safe_parse_date(x):
    if pd.isna(x):
        return None
    try:
        return parser.parse(str(x), dayfirst=False, yearfirst=True)
    except Exception:
        return None


def transcribe_with_whisper(audio_bytes: bytes, model, filename: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
    try:
        result = model.transcribe(path)
        return result.get("text", "").strip()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def transcribe_with_vosk(audio_bytes: bytes, model, filename: str) -> str:
    # Vosk expects WAV PCM 16k mono. We'll try to coerce using wave if already wav; otherwise rely on ffmpeg via whisper isn't possible here.
    # For simplicity: if not WAV, we save and try to open. If not WAV PCM, we warn the user.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
    try:
        if not path.lower().endswith('.wav'):
            return "[Vosk] Please upload WAV PCM audio (16k mono) or use Whisper for auto-conversion."
        wf = wave.open(path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            return "[Vosk] WAV must be mono 16-bit PCM. Convert your file or use Whisper."
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        text_pieces = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = rec.Result()
                text_pieces.append(res)
        final = rec.FinalResult()
        text_pieces.append(final)
        # Combine naive
        return " ".join(text_pieces)
    except Exception as e:
        return f"[Vosk Error] {e}"
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# -----------------------------
# File Uploaders
# -----------------------------
st.subheader("1) Upload Data")
col1, col2 = st.columns([1,1])
with col1:
    csv_file = st.file_uploader("Upload CSV logs (remarks, student, year, tech stack, location, date, optional label)", type=["csv"]) 
with col2:
    audio_files = st.file_uploader("Upload call recordings (mp3/wav)", type=["mp3", "wav", "m4a", "aac"], accept_multiple_files=True)

# -----------------------------
# Load DataFrame + Column Mapping
# -----------------------------
if csv_file is not None:
    try:
        df_raw = pd.read_csv(csv_file)
    except Exception:
        df_raw = pd.read_csv(csv_file, encoding="latin-1")
    st.success(f"CSV loaded with shape {df_raw.shape}")
    with st.expander("Map columns (flexible)"):
        cols = ["<none>"] + list(df_raw.columns)
        map_student = st.selectbox("Student Name column", cols, index=cols.index("student_name") if "student_name" in df_raw.columns else 0)
        map_year = st.selectbox("Year column", cols, index=cols.index("year") if "year" in df_raw.columns else 0)
        map_stack = st.selectbox("Tech Stack column", cols, index=cols.index("tech_stack") if "tech_stack" in df_raw.columns else 0)
        map_loc = st.selectbox("Location column", cols, index=cols.index("location") if "location" in df_raw.columns else 0)
        map_remarks = st.selectbox("Remarks/Notes column", cols, index=cols.index("remarks") if "remarks" in df_raw.columns else 0)
        map_callid = st.selectbox("Call ID column (optional)", cols, index=cols.index("call_id") if "call_id" in df_raw.columns else 0)
        map_date = st.selectbox("Date column (optional)", cols, index=cols.index("date") if "date" in df_raw.columns else 0)
        map_label = st.selectbox("Sentiment label column (optional: positive/neutral/negative)", cols, index=cols.index("label") if "label" in df_raw.columns else 0)

    def pick(colname):
        return None if colname == "<none>" else df_raw[colname]

    df = pd.DataFrame({
        "call_id": pick(map_callid) if map_callid != "<none>" else pd.Series([None]*len(df_raw)),
        "student_name": pick(map_student) if map_student != "<none>" else pd.Series([None]*len(df_raw)),
        "year": pick(map_year) if map_year != "<none>" else pd.Series([None]*len(df_raw)),
        "tech_stack": pick(map_stack) if map_stack != "<none>" else pd.Series([None]*len(df_raw)),
        "location": pick(map_loc) if map_loc != "<none>" else pd.Series([None]*len(df_raw)),
        "remarks": pick(map_remarks) if map_remarks != "<none>" else pd.Series([""]*len(df_raw)),
        "date": pick(map_date) if map_date != "<none>" else pd.Series([None]*len(df_raw)),
        "label": pick(map_label) if map_label != "<none>" else pd.Series([None]*len(df_raw)),
    })
    if "date" in df.columns:
        df["date_parsed"] = df["date"].apply(safe_parse_date)
    else:
        df["date_parsed"] = None
else:
    df = None

# -----------------------------
# Transcribe Audio
# -----------------------------
transcripts = []
if audio_files:
    st.subheader("2) Transcribe Audio")
    if asr_engine == "Whisper":
        try:
            whisper_model = load_whisper(whisper_size)
        except Exception as e:
            st.error(str(e))
            whisper_model = None
    else:
        try:
            vosk_model = load_vosk(vosk_model_dir)
        except Exception as e:
            st.error(str(e))
            vosk_model = None

    prog = st.progress(0)
    for i, f in enumerate(audio_files):
        audio_bytes = f.read()
        if asr_engine == "Whisper" and whisper_model is not None:
            text = transcribe_with_whisper(audio_bytes, whisper_model, f.name)
        elif asr_engine == "Vosk (offline)" and 'vosk_model' in locals() and vosk_model is not None:
            text = transcribe_with_vosk(audio_bytes, vosk_model, f.name)
        else:
            text = "[ASR not available]"
        transcripts.append({
            "call_id": os.path.splitext(os.path.basename(f.name))[0],
            "transcript_text": text
        })
        prog.progress(int(((i+1)/len(audio_files))*100))
    st.success(f"Transcribed {len(transcripts)} file(s)")

if transcripts:
    df_tr = pd.DataFrame(transcripts)
else:
    df_tr = pd.DataFrame(columns=["call_id", "transcript_text"])  # empty

# -----------------------------
# Merge Transcripts with CSV
# -----------------------------
if df is not None:
    st.subheader("3) Merge Logs + Transcripts")
    # Try join on call_id if available, else outer merge on index
    if df["call_id"].notna().any() and not df_tr.empty:
        merged = pd.merge(df, df_tr, on="call_id", how="outer")
    else:
        # append transcripts as new rows if missing call_id
        merged = df.copy()
        if not df_tr.empty:
            extra = pd.DataFrame({
                "call_id": df_tr["call_id"],
                "student_name": None,
                "year": None,
                "tech_stack": None,
                "location": None,
                "remarks": "",
                "date": None,
                "label": None,
                "date_parsed": None,
                "transcript_text": df_tr["transcript_text"]
            })
            merged = pd.concat([merged, extra], ignore_index=True)

        merged["remarks"] = merged.get("remarks", pd.Series([""] * len(merged))).fillna("")
    if "transcript_text" not in merged.columns:
        merged["transcript_text"] = ""
    else:
        merged["transcript_text"] = merged["transcript_text"].fillna("")
        merged["combined_text"] = (merged["remarks"].astype(str) + " " + merged["transcript_text"].astype(str)).str.strip()

    st.dataframe(merged.head(50), use_container_width=True)
else:
    merged = None

# -----------------------------
# Sentiment: Train or Pretrained
# -----------------------------
if merged is not None and len(merged) > 0:
    st.subheader("4) Sentiment Analysis")
    can_train = ("label" in merged.columns) and merged["label"].notna().any() and not use_pretrained

    if can_train:
        st.write("Training custom TF-IDF + LogisticRegression on provided labels…")
        with st.spinner("Training model…"):
            try:
                vectorizer, clf, report = train_sklearn_sentiment(merged["combined_text"].fillna("") , merged["label"])
                st.text("Classification report (hold-out test):\n" + report)
                # Prediction Algorithms for the custom model
                
                #x data : train 
                #y data : test 
                # 
                Xall = vectorizer.transform(merged["combined_text"].fillna(""))
                merged['sentiment'] = clf.predict(Xall)
                merged['sentiment_score'] = np.nan
                model_used = 'custom_sklearn'
                
            except Exception as e:
                st.error(f"Training failed: {e}. Falling back to pretrained pipeline.")
                can_train = False
    #if training failes
    if not can_train:
        with st.spinner('Running pre-trained sentiment models'):
            nlp = load_hf_pipeline()
            preds = []
            scores = []
            for txt in merged["combined_text"].fillna(""):
                try:
                    # 1024,2098,4096
                    r = nlp(txt[:4096])[0]
                    label = r['label'].lower()
                    
                    # pasitive,negative or neutral
                    if label == 'positive':
                       preds.append('positive')
                    elif label == 'negative':
                        preds.append('negative')
                    else:
                        preds.append(label)
                    scores.append(float(r.get('score',np.nan)))
                except Exception as e:
                    print(f' Exception in cannot train {str(e)}')
                    st.error(f' Exception in cannot train {str(e)}')
                    preds.append('neutral')
                    scores.append(np.nan)
            merged['sentiment'] = preds
            merged['sentiment_score'] = scores
            model_used = 'hf_distilbert'
            
st.success(f'sentiment computed using {model_used}')

#-------------------------------------
## Analytics 
#------------------------------------

#locations,tech_stack
st.subheader('5) Analytics')
colA,colB,colC = st.columns(3)

with colA: #distribution of the pie-chart 
    fig = px.pie(merged,names='sentiment',title='sentiment distribution')
    st.plotly_chart(fig,use_container_width=True)
    
#location Analytics 
with colB:
    if 'location' in merged.columns:
        fig2 = px.bar(merged.fillna({'location','unknown'}),x='location',color='sentiment',title='sentiment by location')
        st.plotly_chart(fig2,use_container_width=True)
#tech stack Analytics 
if 'tech_stack' in merged.columns:
        fig3 = px.bar(merged.fillna({'tech_stack','unknown'}),x='tech_stack',color='sentiment',title='sentiment by tech stack')
        st.plotly_chart(fig3,use_container_width=True)
        


#Trends Analysis
#Negative Keywords 