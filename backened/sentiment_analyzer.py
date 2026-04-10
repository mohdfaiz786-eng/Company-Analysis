"""
Sentiment analysis module
"""

import re
from typing import Dict, List, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalyzer:
    """Professional sentiment analyzer"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def preprocess(self, text: str) -> str:
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def analyze_vader(self, text: str) -> Tuple[str, float]:
        """VADER sentiment analysis"""
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return "positive", compound
        elif compound <= -0.05:
            return "negative", compound
        else:
            return "neutral", compound
    
    def analyze_textblob(self, text: str) -> Tuple[str, float]:
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "positive", polarity
        elif polarity < -0.1:
            return "negative", polarity
        else:
            return "neutral", polarity
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(top_n)]
    
    def analyze(self, text: str) -> Dict:
        """Complete sentiment analysis"""
        processed = self.preprocess(text)
        
        if not processed:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'keywords': [],
                'recommendations': []
            }
        
        sentiment, confidence = self.analyze_vader(processed)
        keywords = self.extract_keywords(processed)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(processed, sentiment)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'keywords': keywords,
            'recommendations': recommendations
        }
    
    def generate_recommendations(self, text: str, sentiment: str) -> List[str]:
        """Generate recommendations based on text"""
        recommendations = []
        
        if sentiment == 'negative':
            if 'fee' in text or 'price' in text:
                recommendations.append("Consider offering EMI options or discounts")
            if 'placement' in text:
                recommendations.append("Share placement success stories and statistics")
            if 'timing' in text:
                recommendations.append("Offer flexible batch timings")
        
        return recommendations
