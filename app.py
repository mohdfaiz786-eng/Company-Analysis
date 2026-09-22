"""
Softpro Sentiment & Sales Insights
Professional Streamlit Application
Version: 3.0.0 - Final Production Ready
"""

import os
import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import hashlib
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dateutil import parser

# Optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================
# Configuration
# ============================================

class Config:
    """Application configuration"""
    APP_NAME = os.getenv("APP_NAME", "Softpro Sentiment Insights")
    APP_VERSION = os.getenv("APP_VERSION", "3.0.0")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2000"))
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"D:\Training\spi\Python-with-Datascience\my-softpro-project\softpro-Analytics\ffmpeg\bin")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
        os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Page configuration
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .rec-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .rec-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .rec-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Session State Initialization
# ============================================

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'transcripts_cache' not in st.session_state:
    st.session_state.transcripts_cache = {}

# ============================================
# Utility Functions
# ============================================

@st.cache_data(ttl=3600)
def cache_audio_hash(audio_bytes: bytes) -> str:
    return hashlib.md5(audio_bytes).hexdigest()

def preprocess_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s\.\!\?\,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_parse_date(x):
    if pd.isna(x) or x is None:
        return None
    try:
        if isinstance(x, (datetime, pd.Timestamp)):
            return x
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
            try:
                return datetime.strptime(str(x), fmt)
            except ValueError:
                continue
        return parser.parse(str(x), fuzzy=True)
    except Exception:
        return None

def generate_wordcloud(text_series: pd.Series):
    try:
        text = ' '.join(text_series.dropna().astype(str))
        if not text.strip():
            return None
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            colormap='viridis', max_words=100,
            contour_width=1, contour_color='steelblue'
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.warning(f"Word cloud generation failed: {e}")
        return None

# ============================================
# Sentiment Analysis
# ============================================

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

def analyze_sentiment_vader(text: str) -> Tuple[str, float]:
    if not VADER_AVAILABLE or not text:
        return "neutral", 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "positive", compound
    elif compound <= -0.05:
        return "negative", compound
    return "neutral", compound

def analyze_sentiment_textblob(text: str) -> Tuple[str, float]:
    if not TEXTBLOB_AVAILABLE or not text:
        return "neutral", 0.0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive", polarity
    elif polarity < -0.1:
        return "negative", polarity
    return "neutral", polarity

# ============================================
# Audio Transcription
# ============================================

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

def transcribe_audio(audio_bytes: bytes, model_size: str = "base") -> Tuple[str, Dict]:
    if not WHISPER_AVAILABLE:
        return "Whisper not available", {"error": "Whisper not installed"}
    
    tmp_path = None
    try:
        model = whisper.load_model(model_size)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()
            tmp_path = tmp_file.name
        
        result = model.transcribe(tmp_path, language='en', task='transcribe', verbose=False)
        metadata = {
            "language": result.get('language', 'unknown'),
            "duration": result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0,
            "segments_count": len(result.get('segments', []))
        }
        return result['text'], metadata
    except Exception as e:
        return f"Transcription error: {str(e)}", {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

# ============================================
# Analytics
# ============================================

def create_dashboard(df: pd.DataFrame):
    if df.empty:
        st.warning("No data available")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Calls</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        positive = (df['sentiment'] == 'positive').mean() * 100 if 'sentiment' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #28a745;">{positive:.1f}%</div>
            <div class="metric-label">Positive Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        negative = (df['sentiment'] == 'negative').mean() * 100 if 'sentiment' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #dc3545;">{negative:.1f}%</div>
            <div class="metric-label">Negative Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_score = df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_score:.2f}</div>
            <div class="metric-label">Avg Sentiment Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment", "📍 Geographic", "💻 Tech Stack", "📈 Trends"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if 'sentiment' in df.columns:
                fig_pie = px.pie(
                    df, names='sentiment', title='Sentiment Distribution',
                    color='sentiment',
                    color_discrete_map={'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'},
                    hole=0.3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            if 'combined_text' in df.columns:
                st.subheader("Word Cloud")
                fig_wc = generate_wordcloud(df['combined_text'])
                if fig_wc:
                    st.pyplot(fig_wc)
    
    with tab2:
        if 'location' in df.columns and 'sentiment' in df.columns and df['location'].notna().any():
            location_data = df.groupby(['location', 'sentiment']).size().unstack(fill_value=0)
            fig_loc = px.bar(location_data, title="Sentiment by Location", barmode='stack')
            st.plotly_chart(fig_loc, use_container_width=True)
        else:
            st.info("No location data available")
    
    with tab3:
        if 'tech_stack' in df.columns and 'sentiment' in df.columns and df['tech_stack'].notna().any():
            tech_df = df[df['tech_stack'].notna() & (df['tech_stack'] != 'Unknown')]
            if len(tech_df) > 0:
                positive_ratio = tech_df.groupby('tech_stack').apply(
                    lambda x: (x['sentiment'] == 'positive').mean()
                ).sort_values(ascending=False).head(10)
                fig_tech = px.bar(
                    x=positive_ratio.values, y=positive_ratio.index,
                    orientation='h', title="Top Performing Tech Stacks",
                    labels={'x': 'Positive Ratio', 'y': 'Tech Stack'},
                    color=positive_ratio.values, color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_tech, use_container_width=True)
        else:
            st.info("No tech stack data available")
    
    with tab4:
        if 'date_parsed' in df.columns and df['date_parsed'].notna().any():
            temp_df = df.copy()
            temp_df['date'] = pd.to_datetime(temp_df['date_parsed'])
            temp_df['week'] = temp_df['date'].dt.to_period('W').astype(str)
            
            weekly_data = temp_df.groupby('week').agg({
                'sentiment_score': 'mean',
                'call_id': 'count'
            }).reset_index()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=weekly_data['week'], y=weekly_data['sentiment_score'],
                name='Avg Sentiment', line=dict(color='#667eea', width=3)
            ))
            fig_trend.add_trace(go.Bar(
                x=weekly_data['week'], y=weekly_data['call_id'],
                name='Call Volume', yaxis='y2', marker_color='#764ba2'
            ))
            fig_trend.update_layout(
                title="Weekly Trends", xaxis_title="Week",
                yaxis_title="Sentiment Score",
                yaxis2=dict(title="Call Volume", overlaying='y', side='right')
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No date data available")

def generate_recommendations(df: pd.DataFrame) -> List[Dict]:
    recommendations = []
    
    if df.empty or 'sentiment' not in df.columns:
        return recommendations
    
    negative_ratio = (df['sentiment'] == 'negative').mean()
    
    if negative_ratio > 0.4:
        recommendations.append({
            "priority": "HIGH",
            "title": "Critical: High Negative Sentiment",
            "description": f"{negative_ratio:.1%} of calls show negative sentiment",
            "action": "Schedule emergency team meeting. Review call scripts.",
            "impact": "Expected 20% improvement in 2 weeks"
        })
    elif negative_ratio > 0.25:
        recommendations.append({
            "priority": "MEDIUM",
            "title": "Moderate Negative Sentiment",
            "description": f"{negative_ratio:.1%} negative calls detected",
            "action": "Conduct focused training for underperforming counselors",
            "impact": "Reduce negative sentiment by 10% in 1 month"
        })
    
    if 'location' in df.columns:
        for location in df['location'].dropna().unique():
            loc_df = df[df['location'] == location]
            if len(loc_df) > 5:
                neg_rate = (loc_df['sentiment'] == 'negative').mean()
                if neg_rate > 0.35:
                    recommendations.append({
                        "priority": "MEDIUM",
                        "title": f"Location Alert: {location}",
                        "description": f"{neg_rate:.1%} negative rate with {len(loc_df)} calls",
                        "action": f"Launch location-specific satisfaction survey",
                        "impact": "Improve local sentiment by 15%"
                    })
    
    if 'combined_text' in df.columns:
        all_text = ' '.join(df['combined_text'].dropna().astype(str)).lower()
        issues = {
            'fee': ("Pricing Concerns", "Introduce EMI, scholarships, or early bird discounts"),
            'placement': ("Placement Anxiety", "Share success stories and placement statistics"),
            'timing': ("Schedule Conflicts", "Offer flexible batch timings and weekend classes"),
            'faculty': ("Faculty Quality", "Invest in instructor training and certification"),
            'support': ("Support Issues", "Enhance mentorship and doubt-solving sessions")
        }
        for keyword, (title, action) in issues.items():
            if keyword in all_text:
                recommendations.append({
                    "priority": "MEDIUM",
                    "title": title,
                    "description": f"Multiple mentions of '{keyword}' detected in calls",
                    "action": action,
                    "impact": "Address key pain points effectively"
                })
    
    positive_ratio = (df['sentiment'] == 'positive').mean()
    if positive_ratio > 0.6:
        recommendations.append({
            "priority": "LOW",
            "title": "Excellent Performance",
            "description": f"{positive_ratio:.1%} positive sentiment achieved",
            "action": "Document best practices, reward top performers",
            "impact": "Maintain and scale success"
        })
    
    return recommendations

# ============================================
# Main Application
# ============================================

def main():
    st.markdown(f"""
    <div class="main-header">
        <h1>🎯 {Config.APP_NAME}</h1>
        <p>Professional Sentiment Analysis & Sales Insights Platform | Version {Config.APP_VERSION}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        sentiment_model = st.selectbox(
            "Sentiment Analysis Model",
            ["VADER", "TextBlob", "Ensemble"]
        )
        st.markdown("---")
        st.markdown("## 🎵 Audio Settings")
        enable_cache = st.checkbox("Enable Caching", value=True)
        show_transcription = st.checkbox("Show Full Transcription", value=False)
        st.markdown("---")
        st.markdown("## 📤 Export")
        export_format = st.selectbox("Export Format", ["CSV", "HTML Report"])
        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.markdown(f"- Cache: {'Enabled' if enable_cache else 'Disabled'}")
        st.markdown(f"- Whisper: {'Available' if WHISPER_AVAILABLE else 'Not installed'}")
        st.markdown(f"- VADER: {'Available' if VADER_AVAILABLE else 'Not installed'}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 📁 Data Upload")
        csv_file = st.file_uploader("Upload CRM Data (CSV)", type=['csv'])
    
    with col2:
        st.markdown("## 🎙️ Audio Upload")
        audio_files = st.file_uploader(
            "Upload Call Recordings",
            type=['wav', 'mp3', 'm4a'],
            accept_multiple_files=True
        )
    
    # Initialize
    df = None
    transcripts = []
    
    # Process CSV
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.success(f"✅ Loaded {df.shape[0]} records")
            
            with st.expander("📊 Data Preview & Column Selection"):
                st.dataframe(df.head(), use_container_width=True)
                text_col = st.selectbox("Select Text/Remarks Column", df.columns)
                df['combined_text'] = df[text_col].fillna('').astype(str)
                
                if 'date' in df.columns:
                    df['date_parsed'] = df['date'].apply(safe_parse_date)
                
                if 'call_id' not in df.columns:
                    df['call_id'] = [f"call_{i}" for i in range(len(df))]
                else:
                    df['call_id'] = df['call_id'].astype(str)
                    
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            df = None
    
    # Process Audio
    if audio_files:
        st.markdown("## 🎤 Audio Transcription")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, audio_file in enumerate(audio_files):
            status_text.text(f"Processing: {audio_file.name}")
            audio_hash = cache_audio_hash(audio_file.getvalue())
            
            if enable_cache and audio_hash in st.session_state.transcripts_cache:
                transcript = st.session_state.transcripts_cache[audio_hash]
            else:
                transcript, _ = transcribe_audio(audio_file.getvalue(), Config.WHISPER_MODEL)
                if enable_cache:
                    st.session_state.transcripts_cache[audio_hash] = transcript
            
            transcripts.append({
                'call_id': Path(audio_file.name).stem,
                'transcript_text': transcript
            })
            progress_bar.progress((idx + 1) / len(audio_files))
        
        status_text.text("✅ All audio files processed!")
        
        if show_transcription:
            with st.expander("📝 Transcription Results"):
                for trans in transcripts:
                    st.text(trans['transcript_text'][:300] + "..." if len(trans['transcript_text']) > 300 else trans['transcript_text'])
                    st.markdown("---")
    
    # ============================================
    # MERGE SECTION - FIXED
    # ============================================
    
    if transcripts:
        transcripts_df = pd.DataFrame(transcripts)
        
        if df is not None:
            # Ensure call_id exists in both
            if 'call_id' not in df.columns:
                df['call_id'] = [f"call_{i}" for i in range(len(df))]
            df['call_id'] = df['call_id'].astype(str)
            transcripts_df['call_id'] = transcripts_df['call_id'].astype(str)
            
            # Merge
            df = df.merge(transcripts_df, on='call_id', how='outer')
            
            # ✅ Safe column handling
            if 'transcript_text' not in df.columns:
                df['transcript_text'] = ''
            df['transcript_text'] = df['transcript_text'].fillna('').astype(str)
            
            if 'combined_text' not in df.columns:
                df['combined_text'] = ''
            df['combined_text'] = df['combined_text'].fillna('').astype(str)
            
            # Safe concatenation
            df['combined_text'] = (df['combined_text'] + " " + df['transcript_text']).str.strip()
        else:
            df = transcripts_df.copy()
            df['combined_text'] = df['transcript_text'].fillna('').astype(str)
            if 'call_id' not in df.columns:
                df['call_id'] = [f"audio_{i}" for i in range(len(df))]
    
    elif df is not None:
        # Only CSV, no audio
        if 'combined_text' not in df.columns:
            df['combined_text'] = ''
        df['combined_text'] = df['combined_text'].fillna('').astype(str)
    
    # ============================================
    # SENTIMENT ANALYSIS
    # ============================================
    
    if df is not None and len(df) > 0:
        st.markdown("---")
        st.markdown("## 🔍 Sentiment Analysis")
        
        with st.spinner("Analyzing sentiments..."):
            sentiments = []
            scores = []
            
            for text in df['combined_text'].fillna(''):
                processed_text = preprocess_text(text)[:Config.MAX_TEXT_LENGTH]
                
                if sentiment_model == "VADER":
                    sent, score = analyze_sentiment_vader(processed_text)
                elif sentiment_model == "TextBlob":
                    sent, score = analyze_sentiment_textblob(processed_text)
                else:
                    vader_sent, vader_score = analyze_sentiment_vader(processed_text)
                    blob_sent, blob_score = analyze_sentiment_textblob(processed_text)
                    sent = vader_sent if abs(vader_score) > abs(blob_score) else blob_sent
                    score = (vader_score + blob_score) / 2
                
                sentiments.append(sent)
                scores.append(score)
            
            df['sentiment'] = sentiments
            df['sentiment_score'] = scores
        
        st.success(f"✅ Analysis complete for {len(df)} records")
        
        # Dashboard
        create_dashboard(df)
        
        # Recommendations
        st.markdown("---")
        st.markdown("## 💡 Recommendations")
        
        recommendations = generate_recommendations(df)
        
        if recommendations:
            for rec in recommendations:
                priority_class = {
                    "HIGH": "rec-high",
                    "MEDIUM": "rec-medium",
                    "LOW": "rec-low"
                }.get(rec['priority'], "rec-medium")
                
                st.markdown(f"""
                <div class="{priority_class}">
                    <strong>🎯 {rec['title']}</strong><br>
                    📊 {rec['description']}<br>
                    💡 <strong>Action:</strong> {rec['action']}<br>
                    📈 <strong>Expected Impact:</strong> {rec['impact']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No critical issues detected. Keep up the good work!")
        
        # Export
        st.markdown("---")
        st.markdown("## 📤 Export Results")
        
        if export_format == "CSV":
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                f"softpro_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Softpro Analysis Report</title></head>
            <body>
                <h1>Softpro Sentiment Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Calls: {len(df)}</p>
                <p>Positive: {(df['sentiment'] == 'positive').mean() * 100:.1f}%</p>
                <p>Negative: {(df['sentiment'] == 'negative').mean() * 100:.1f}%</p>
                {df.head(20).to_html()}
            </body>
            </html>
            """
            st.download_button(
                "📄 Download HTML Report",
                html_content,
                f"softpro_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html"
            )
        
        st.session_state.processed_data = df
        st.session_state.analysis_complete = True
        
    else:
        st.info("👈 Please upload CSV data and/or audio files to begin analysis")

if __name__ == "__main__":
    main()
