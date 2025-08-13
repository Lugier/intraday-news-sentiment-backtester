"""
Gemini API Helper

Simple interface for querying Google's Gemini API with text and images.
"""

import os
import base64
from pathlib import Path
import google.generativeai as genai
from PIL import Image
from typing import Optional, Union, Dict, Any, List
from io import BytesIO
import logging
import time
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm
import json
from google.oauth2 import service_account

from src.config import API_CONFIG

logger = logging.getLogger(__name__)

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 2000
MAX_PARALLEL_REQUESTS = 200  # Maximum number of parallel requests

# Override with config values if available
if "max_requests_per_minute" in API_CONFIG["gemini_api"]:
    MAX_REQUESTS_PER_MINUTE = API_CONFIG["gemini_api"]["max_requests_per_minute"]

if "max_parallel_requests" in API_CONFIG["gemini_api"]:
    MAX_PARALLEL_REQUESTS = API_CONFIG["gemini_api"]["max_parallel_requests"]

REQUEST_TIMESTAMPS = []

def setup_gemini_api():
    """
    Set up the Gemini API with API key from environment variables or service account
    
    Returns:
        genai.GenerativeModel: Configured Gemini model
    """
    # First, try to use API key
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully with API key")
    else:
        # Try with service account if API key not available
        service_account_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if service_account_file and os.path.exists(service_account_file):
            try:
                # Load service account credentials
                logger.info(f"Using service account from: {service_account_file}")
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_file,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                
                # Configure genai with the service account
                genai.configure(credentials=credentials)
                logger.info("Gemini API configured successfully with service account")
            except Exception as e:
                logger.error(f"Error configuring Gemini API with service account: {str(e)}")
                # Fall back to default API key
                default_api_key = API_CONFIG["gemini_api"].get("api_key", "AIzaSyALHILnxO9nhOsGspszPXKL1a2MQwGYyEM")
                genai.configure(api_key=default_api_key)
                logger.info("Gemini API configured with default API key as fallback")
        else:
            # Use default API key as last resort
            logger.warning("No API key or service account found. Using default API key.")
            default_api_key = API_CONFIG["gemini_api"].get("api_key", "AIzaSyALHILnxO9nhOsGspszPXKL1a2MQwGYyEM")
            genai.configure(api_key=default_api_key)
            logger.info("Gemini API configured with default API key")
    
    model = genai.GenerativeModel(API_CONFIG["gemini_api"]["model"])
    return model

def wait_for_rate_limit():
    """Ensure we don't exceed the rate limit of 2000 requests per minute."""
    global REQUEST_TIMESTAMPS
    current_time = datetime.now()
    
    # Remove timestamps older than 1 minute
    REQUEST_TIMESTAMPS = [ts for ts in REQUEST_TIMESTAMPS 
                         if current_time - ts < timedelta(minutes=1)]
    
    # If we've made 2000 requests in the last minute, wait
    if len(REQUEST_TIMESTAMPS) >= MAX_REQUESTS_PER_MINUTE:
        wait_time = (REQUEST_TIMESTAMPS[0] + timedelta(minutes=1) - current_time).total_seconds()
        if wait_time > 0:
            logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    
    # Add current request timestamp
    REQUEST_TIMESTAMPS.append(current_time)

def query_gemini(prompt: str, image_path: Optional[Union[str, Path]] = None, 
                max_retries: int = API_CONFIG["gemini_api"]["max_retries"], 
                retry_delay: int = API_CONFIG["gemini_api"]["retry_delay"]) -> str:
    """
    Send a query to the Gemini API with retry logic
    
    Args:
        prompt: The text prompt to send
        image_path: Optional path to an image to include with the prompt
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries
        
    Returns:
        str: The model's response
    """
    wait_for_rate_limit()
    
    # Set up Gemini model
    try:
        model = setup_gemini_api()
    except Exception as e:
        return f"Error setting up Gemini API: {str(e)}"
    
    # Prepare content for API
    content = [{"role": "user", "parts": [prompt]}]
    
    # Add image if provided
    if image_path:
        try:
            # Open and compress image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (max 1280px on longest side)
                max_size = 1280
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Convert to JPEG bytes
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=75, optimize=True)
                image_data = buffer.getvalue()
            
            # Add image to content
            content[0]["parts"].append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode()
                }
            })
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {str(e)}")
            # Continue without image if there's an error
    
    # Get response from API with retries
    for attempt in range(max_retries):
        try:
            response = model.generate_content(content)
            return response.text
        except Exception as e:
            logger.warning(f"Error querying Gemini API (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    return f"Error querying Gemini API after {max_retries} attempts"

def analyze_sentiment(title: str, description: str, ticker: str) -> dict:
    """
    Analyze the sentiment of a news article about a stock
    
    Args:
        title: The news article title
        description: The news article description
        ticker: The stock ticker symbol
        
    Returns:
        dict: Sentiment analysis result with 'sentiment' and 'explanation' keys
    """
    prompt = f"""You are a **Hyper-Focused Algorithmic Trading Analyst**. Your sole mission is to determine the **immediate causal impact** of a specific news item on the stock price of **{ticker}** within the **next 60 minutes**. Speed, accuracy, and decisiveness are paramount.

**TASK:** Classify the news below as **POSITIVE**, **NEGATIVE**, or **NEUTRAL** based *only* on its likely effect on **{ticker}**'s price in the **next 60 minutes**.

**NEWS CONTENT:**
*   **Title:** {title}
*   **Description:** {description}

**CRITICAL TRADING CONTEXT:**
*   **Time Horizon:** STRICTLY the next 60 minutes. Ignore long-term effects.
*   **Trading System:** An algorithm will automatically enter a trade 2 minutes after news publication based on your classification and exit within 60 minutes.
*   **Goal:** Predict the *direction* the news itself will push the price.

**ANALYSIS FRAMEWORK (Apply this rigorously):**
1.  **Direct Impact:** Does the news directly affect {ticker}'s revenue, costs, operations, or market share?
2.  **Magnitude & Surprise:** Is this significant, unexpected news, or minor/anticipated? High surprise = higher potential impact.
3.  **Sentiment Trigger:** How will *most* short-term traders likely interpret and react to this *specific* news *right now*? (e.g., FOMO buying, panic selling).
4.  **Quantifiable Data:** Does the news contain specific numbers (e.g., earnings figures, user growth, settlement amounts) that beat/miss expectations?
5.  **Catalyst Check:** Is this a known catalyst type (e.g., FDA decision, M&A, major contract win/loss)?

**SENTIMENT CLASSIFICATION RULES:**
*   **POSITIVE:** High confidence the news *itself* will cause the stock price to **increase significantly** relative to its expected movement within 60 minutes.
*   **NEGATIVE:** High confidence the news *itself* will cause the stock price to **decrease significantly** relative to its expected movement within 60 minutes.
*   **NEUTRAL:** Low confidence the news *itself* will cause a significant price deviation within 60 minutes. This includes:
    *   News already priced in or widely expected.
    *   Minor updates with no immediate financial implications.
    *   Ambiguous or conflicting information where the net 60-minute impact is unclear.
    *   News irrelevant to {ticker}'s core business or value.

**OUTPUT FORMAT (Strictly adhere to this):**
```
Sentiment: [**POSITIVE**/**NEGATIVE**/**NEUTRAL**]
Explanation: [**Mandatory 2-3 sentences.** Briefly justify the sentiment classification by *explicitly referencing* key points from the **Analysis Framework** (e.g., "Significant surprise factor suggests...", "No direct impact identified..."). Explain *why* this specific news will (or won't) move the price *in the next 60 minutes*.]
```

**DO NOT:**
*   Provide long-term analysis or price targets.
*   Offer investment advice.
*   Summarize the news article.
*   Analyze general market conditions unless the news *directly* interacts with them.
*   Use any sentiment labels other than POSITIVE, NEGATIVE, or NEUTRAL.
"""

    try:
        response = query_gemini(prompt)
        return parse_sentiment_response(response)
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise

def parse_sentiment_response(response: str) -> dict:
    """
    Parse the sentiment analysis response from Gemini API.
    
    Args:
        response: Raw response from the API
        
    Returns:
        Dictionary containing sentiment and explanation
    """
    # Try different patterns to extract sentiment and explanation
    patterns = [
        # Pattern 1: Standard format
        r"Sentiment:\s*(positive|neutral|negative)\s*Explanation:\s*(.*)",
        # Pattern 2: With newlines
        r"Sentiment:\s*(positive|neutral|negative)\s*\n\s*Explanation:\s*(.*)",
        # Pattern 3: With dashes
        r"Sentiment:\s*(positive|neutral|negative)\s*-\s*(.*)",
        # Pattern 4: Just sentiment followed by explanation
        r"(positive|neutral|negative)\s*:\s*(.*)",
    ]
    
    for pattern in patterns:
        import re
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            sentiment = match.group(1).lower()
            explanation = match.group(2).strip()
            return {"sentiment": sentiment, "explanation": explanation}
    
    # Fallback: Try to find sentiment in the first line
    first_line = response.split('\n')[0].lower()
    if 'positive' in first_line:
        sentiment = 'positive'
    elif 'negative' in first_line:
        sentiment = 'negative'
    elif 'neutral' in first_line:
        sentiment = 'neutral'
    else:
        sentiment = 'neutral'  # Default if we can't determine
    
    # Get the rest as explanation
    lines = response.split('\n')
    if len(lines) > 1:
        explanation = ' '.join(lines[1:]).strip()
    else:
        explanation = "No detailed explanation provided."
    
    return {"sentiment": sentiment, "explanation": explanation}

def analyze_sentiments_batch(news_items: List[Dict[str, Any]], ticker: str, batch_size: int = None) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for multiple news items in parallel, automatically optimizing batch size
    to use the full API rate limit of 2000 requests per minute.
    
    Args:
        news_items: List of news items with 'title' and 'description' keys
        ticker: Stock ticker symbol
        batch_size: Optional manual batch size override. If None, will be calculated automatically.
        
    Returns:
        List of sentiment analysis results
    """
    results = []
    total_items = len(news_items)
    
    # Calculate optimal batch size if not provided
    if batch_size is None:
        # Default to the configured batch size if not specified
        batch_size = API_CONFIG["gemini_api"]["batch_size"]
        
        # We aim to process everything in about a minute if possible
        # But we limit to MAX_PARALLEL_REQUESTS for efficient execution
        optimal_batch_size = min(MAX_REQUESTS_PER_MINUTE, total_items, MAX_PARALLEL_REQUESTS)
        batch_size = optimal_batch_size
    
    logger.info(f"Analyzing sentiment for {total_items} news items with batch size of {batch_size}")
    logger.info(f"Using maximum rate of {MAX_REQUESTS_PER_MINUTE} requests per minute")
    
    # Calculate estimated time
    estimated_time = (total_items / batch_size) * 60 if batch_size < MAX_REQUESTS_PER_MINUTE else total_items / MAX_REQUESTS_PER_MINUTE * 60
    logger.info(f"Estimated processing time: {estimated_time:.1f} seconds")
    
    # Process in batches
    for i in range(0, total_items, batch_size):
        batch = news_items[i:i+batch_size]
        batch_results = []
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size} ({len(batch)} items)")
        
        # Adjust max_workers to batch size but cap at MAX_PARALLEL_REQUESTS
        max_workers = min(len(batch), MAX_PARALLEL_REQUESTS)
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Start the load operations and mark each future with its index
            future_to_index = {
                executor.submit(
                    analyze_sentiment, 
                    item.get('title', '').strip(), 
                    item.get('description', '').strip(), 
                    ticker
                ): idx 
                for idx, item in enumerate(batch)
            }
            
            # Process as they complete
            with tqdm(total=len(batch), desc="Sentiment Analysis") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        # Create a complete result including the news and sentiment
                        full_result = {
                            'title': batch[idx].get('title', '').strip(),
                            'date': datetime.fromtimestamp(batch[idx].get('time', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                            'sentiment': result['sentiment'],
                            'explanation': result['explanation']
                        }
                        batch_results.append((idx, full_result))
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error analyzing item {idx + i}: {e}")
                        # Add a placeholder for failed analysis
                        batch_results.append((idx, {
                            'title': batch[idx].get('title', '').strip(),
                            'date': datetime.fromtimestamp(batch[idx].get('time', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                            'sentiment': 'neutral',  # Default to neutral if analysis fails
                            'explanation': f"Analysis failed: {str(e)}"
                        }))
                        pbar.update(1)
        
        # Sort batch results by their original index and append to final results
        batch_results.sort(key=lambda x: x[0])
        results.extend([result for _, result in batch_results])
        
        # If we have more batches to process, check if we need to wait for rate limit
        if i + batch_size < total_items:
            current_time = datetime.now()
            # Count requests in the last minute
            recent_requests = sum(1 for ts in REQUEST_TIMESTAMPS 
                               if current_time - ts < timedelta(minutes=1))
            
            # If we're close to the rate limit, wait a bit
            if recent_requests > MAX_REQUESTS_PER_MINUTE * 0.9:
                wait_time = 5  # seconds
                logger.info(f"Approaching rate limit ({recent_requests}/{MAX_REQUESTS_PER_MINUTE}). Waiting {wait_time}s...")
                time.sleep(wait_time)
    
    logger.info(f"Completed sentiment analysis for {len(results)} news items")
    
    # Return only valid results
    return [r for r in results if r.get('sentiment') in ['positive', 'neutral', 'negative']] 