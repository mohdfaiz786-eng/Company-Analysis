# Softpro Sentiment & Sales Insights

Professional sentiment analysis platform for sales calls.

## Features

- 🎤 Audio transcription (Whisper)
- 🔍 Sentiment analysis (VADER/TextBlob)
- 📊 Interactive dashboard
- 💡 Actionable recommendations
- 🔐 API with authentication
- 📱 Responsive design

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run FastAPI backend
uvicorn backend.main:app --reload
