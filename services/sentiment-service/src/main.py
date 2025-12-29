"""
Sentiment Analysis Service
Analyzes news and social media sentiment for stocks
"""
import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4
from enum import Enum
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis Service",
    description="News and social media sentiment analysis",
    version="2.0.0"
)


class SentimentScore(str, Enum):
    VERY_BEARISH = "VERY_BEARISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    VERY_BULLISH = "VERY_BULLISH"


class NewsArticle(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str] = []
    sentiment_score: float = 0
    sentiment: SentimentScore = SentimentScore.NEUTRAL


class SymbolSentiment(BaseModel):
    symbol: str
    overall_score: float
    overall_sentiment: SentimentScore
    news_count: int
    trending: bool = False
    updated_at: datetime = Field(default_factory=datetime.utcnow)


BULLISH_WORDS = {"buy", "bullish", "upgrade", "beat", "surge", "rally", "growth", "profit", "strong", "positive"}
BEARISH_WORDS = {"sell", "bearish", "downgrade", "miss", "crash", "plunge", "loss", "weak", "negative", "warning"}


class SentimentAnalyzer:
    def __init__(self):
        self.cache: Dict[str, SymbolSentiment] = {}
        self.news_cache: Dict[str, List[NewsArticle]] = defaultdict(list)
    
    def analyze_text(self, text: str) -> tuple:
        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))
        bullish = len(words & BULLISH_WORDS)
        bearish = len(words & BEARISH_WORDS)
        total = bullish + bearish
        if total == 0:
            return 0.0, SentimentScore.NEUTRAL
        score = (bullish - bearish) / total
        if score > 0.3:
            return score, SentimentScore.BULLISH
        elif score < -0.3:
            return score, SentimentScore.BEARISH
        return score, SentimentScore.NEUTRAL
    
    async def get_sentiment(self, symbol: str) -> SymbolSentiment:
        symbol = symbol.upper()
        if symbol in self.cache:
            cached = self.cache[symbol]
            if (datetime.utcnow() - cached.updated_at).seconds < 300:
                return cached
        
        sentiment = SymbolSentiment(
            symbol=symbol,
            overall_score=0.0,
            overall_sentiment=SentimentScore.NEUTRAL,
            news_count=0
        )
        self.cache[symbol] = sentiment
        return sentiment


analyzer = SentimentAnalyzer()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    return await analyzer.get_sentiment(symbol.upper())


@app.post("/analyze")
async def analyze(text: str):
    score, sentiment = analyzer.analyze_text(text)
    return {"score": score, "sentiment": sentiment}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8014, reload=True)
