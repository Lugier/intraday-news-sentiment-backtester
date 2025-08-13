"""
Common utilities for the News Sentiment Analyzer and Trading Strategy Backtester.

This module contains utilities that are used across the codebase to reduce duplication and
ensure consistent behavior.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Union, Tuple

import numpy as np
from tqdm.contrib.logging import logging_redirect_tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logger
logger = logging.getLogger('news_module')

# Only display info and above for libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('tqdm').setLevel(logging.WARNING)

# Configure logging to not break tqdm progress bars
from tqdm.contrib.logging import logging_redirect_tqdm

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder for datetime objects and NumPy types."""
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.number) and np.isnan(obj):
            return None
        return super().default(obj)

def prepare_data_for_serialization(data: Any) -> Any:
    """
    Recursively prepare data for JSON serialization by converting NumPy types
    to native Python types.
    
    Args:
        data: Any Python data structure that might contain NumPy types
        
    Returns:
        Data structure with all NumPy types converted to native Python types
    """
    if isinstance(data, datetime):
        return data.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, list):
        return [prepare_data_for_serialization(item) for item in data]
    elif isinstance(data, dict):
        return {key: prepare_data_for_serialization(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(prepare_data_for_serialization(item) for item in data)
    elif hasattr(data, 'dtype') and np.issubdtype(data.dtype, np.number) and np.isnan(data):
        return None
    elif hasattr(data, '__dict__'):  # Handle custom objects
        return str(data)
    else:
        return data

def filter_valid_stories(stories: List[Dict[str, Any]], max_stories: int = None) -> List[Dict[str, Any]]:
    """
    Filter valid stories for sentiment analysis.
    
    Args:
        stories: List of news stories
        max_stories: Maximum number of stories to return (optional)
        
    Returns:
        Filtered list of valid stories
    """
    filtered_stories = []
    
    for story in stories:
        # Check for required fields
        if not all(key in story for key in ['title', 'url', 'time']):
            continue
            
        # Check for empty titles
        if not story.get('title', '').strip():
            continue
            
        # Add to filtered list
        filtered_stories.append(story)
        
        # Check if we've reached the maximum
        if max_stories and len(filtered_stories) >= max_stories:
            break
    
    logger.info(f"Filtered {len(stories) - len(filtered_stories)} invalid stories. {len(filtered_stories)} stories remaining.")
    return filtered_stories

def create_timestamped_output_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Create a timestamped output directory.
    
    Args:
        base_dir: Base directory
        prefix: Prefix for the directory name
        
    Returns:
        Path to the created directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir

def save_json(data: Union[Dict, List], filename: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filename: Path to output file
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Loaded data or None if loading failed
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None 