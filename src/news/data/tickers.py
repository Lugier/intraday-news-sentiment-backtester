"""
Consolidated ticker data module.

This module provides functions to load ticker data from JSON files.
"""

import os
import json
from typing import List

# Get the current directory path
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(DATA_DIR, 'json')

def get_dow_tickers() -> List[str]:
    """
    Get list of Dow Jones Industrial Average tickers.
    
    Returns:
        List of ticker symbols
    """
    try:
        with open(os.path.join(JSON_DIR, 'dow_jones.json'), 'r') as f:
            data = json.load(f)
            return [item['ticker'] for item in data['tickers'] if 'ticker' in item]
    except Exception as e:
        print(f"Error loading Dow Jones tickers: {e}")
        return []

def get_mag7_tickers() -> List[str]:
    """
    Get list of Magnificent 7 tickers.
    
    Returns:
        List of ticker symbols
    """
    try:
        with open(os.path.join(JSON_DIR, 'mag_7.json'), 'r') as f:
            data = json.load(f)
            return [item['ticker'] for item in data['tickers'] if 'ticker' in item]
    except Exception as e:
        print(f"Error loading Magnificent 7 tickers: {e}")
        return []

def get_custom_tickers() -> List[str]:
    """
    Get list of custom tickers.
    
    Returns:
        List of ticker symbols
    """
    try:
        with open(os.path.join(JSON_DIR, 'custom_stocks.json'), 'r') as f:
            data = json.load(f)
            return [item['ticker'] for item in data['tickers'] if 'ticker' in item]
    except Exception as e:
        print(f"Error loading custom tickers: {e}")
        return []

def get_top5_tickers() -> List[str]:
    """
    Get list of top 5 tech tickers.
    
    Returns:
        List of ticker symbols
    """
    try:
        with open(os.path.join(JSON_DIR, 'top5_stocks.json'), 'r') as f:
            data = json.load(f)
            return [item['ticker'] for item in data['tickers'] if 'ticker' in item]
    except Exception as e:
        print(f"Error loading top 5 tickers: {e}")
        return []

if __name__ == "__main__":
    # Print all tickers when run directly
    print("Dow Jones Industrial Average Components:")
    dow_tickers = get_dow_tickers()
    for i, ticker in enumerate(dow_tickers, 1):
        print(f"{i}. {ticker}")
    print(f"\nTotal: {len(dow_tickers)} stocks")
    
    print("\nMagnificent 7 Components:")
    mag7_tickers = get_mag7_tickers()
    for i, ticker in enumerate(mag7_tickers, 1):
        print(f"{i}. {ticker}")
    print(f"\nTotal: {len(mag7_tickers)} stocks")

    print("\nCustom Stock List Components:")
    custom_tickers = get_custom_tickers()
    for i, ticker in enumerate(custom_tickers, 1):
        print(f"{i}. {ticker}")
    print(f"\nTotal: {len(custom_tickers)} stocks")

    print("\nTop 5 Tech Components:")
    top5_tickers = get_top5_tickers()
    for i, ticker in enumerate(top5_tickers, 1):
        print(f"{i}. {ticker}")
    print(f"\nTotal: {len(top5_tickers)} stocks") 