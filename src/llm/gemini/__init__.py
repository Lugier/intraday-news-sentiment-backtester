"""
Modul für die Integration mit der Google Gemini API.

Dieses Modul bietet Funktionen für die Sentimentanalyse von Finanznachrichten
mithilfe des Gemini Large Language Models von Google.
"""

from src.llm.gemini.gemini_helper import (
    analyze_sentiment, 
    analyze_sentiments_batch
)

__all__ = [
    'analyze_sentiment', 
    'analyze_sentiments_batch'
]
