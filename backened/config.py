"""
Configuration management
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings"""
    
    # API Settings
    API_KEY = os.getenv("API_KEY", "softpro_secret_key_2024")
    JWT_SECRET = os.getenv("JWT_SECRET", "your_jwt_secret_key")
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # File settings
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Model settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "vader")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./softpro.db")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Create directories
    os.makedirs(UPLOAD_DIR, exist_ok=True)

settings = Settings()
