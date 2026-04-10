"""
FastAPI Backend for Softpro Sentiment Analysis
Professional REST API with authentication
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import hashlib
import uuid

# Import modules
from .config import settings
from .auth import verify_token
from .sentiment_analyzer import SentimentAnalyzer
from .models import SessionLocal, Call, AnalysisReport

# Initialize FastAPI
app = FastAPI(
    title="Softpro Sentiment Analysis API",
    description="Professional sentiment analysis for sales calls",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize analyzer
analyzer = SentimentAnalyzer()

# ============================================
# Pydantic Models
# ============================================

class TextAnalysisRequest(BaseModel):
    text: str
    call_id: Optional[str] = None

class TextAnalysisResponse(BaseModel):
    call_id: str
    sentiment: str
    confidence: float
    keywords: List[str]
    timestamp: datetime

class AudioAnalysisResponse(BaseModel):
    task_id: str
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime

# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now()
    )

@app.post("/api/v1/analyze/text", response_model=TextAnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    token: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Analyze sentiment from text
    """
    # Verify authentication
    verify_token(token.credentials)
    
    # Analyze sentiment
    result = analyzer.analyze(request.text)
    
    # Generate call ID if not provided
    call_id = request.call_id or str(uuid.uuid4())[:8]
    
    return TextAnalysisResponse(
        call_id=call_id,
        sentiment=result['sentiment'],
        confidence=result['confidence'],
        keywords=result['keywords'][:10],
        timestamp=datetime.now()
    )

@app.post("/api/v1/analyze/audio", response_model=AudioAnalysisResponse)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    token: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Upload and analyze audio file (asynchronous)
    """
    # Verify authentication
    verify_token(token.credentials)
    
    # Validate file
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(400, "Invalid audio format")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Add to background tasks
    background_tasks.add_task(
        process_audio_task,
        task_id=task_id,
        audio_bytes=await audio.read(),
        filename=audio.filename
    )
    
    return AudioAnalysisResponse(
        task_id=task_id,
        status="processing",
        message="Audio uploaded successfully. Analysis in progress."
    )

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# Background Tasks
# ============================================

async def process_audio_task(task_id: str, audio_bytes: bytes, filename: str):
    """Process audio in background"""
    try:
        # Save to temporary file
        temp_path = f"/tmp/{task_id}_{filename}"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Transcribe (implement with Whisper)
        # Analyze sentiment
        # Store results in database
        
        # Cleanup
        os.remove(temp_path)
        
    except Exception as e:
        print(f"Error processing audio {task_id}: {e}")
