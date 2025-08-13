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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects and NumPy types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
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

def filter_valid_stories(stories: List[Dict[str, Any]], max_stories: int = 100) -> List[Dict[str, Any]]:
    """
    Filter out stories that don't have both title and description.
    
    Args:
        stories: List of news stories
        max_stories: Maximum number of stories to return
        
    Returns:
        List of valid stories with both title and description
    """
    valid_stories = []
    for story in stories:
        title = story.get('title', '').strip()
        description = story.get('description', '').strip()
        if title and description:
            valid_stories.append(story)
            if len(valid_stories) >= max_stories:
                break
        else:
            logger.debug(f"Skipping story due to missing data - Title: {bool(title)}, Description: {bool(description)}")
    
    logger.info(f"Filtered {len(stories) - len(valid_stories)} invalid stories. {len(valid_stories)} stories remaining.")
    return valid_stories

def create_timestamped_output_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Create a timestamped output directory.
    
    Args:
        base_dir: Base directory
        prefix: Prefix for the directory name
        
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir

def save_json(data: Any, filepath: str, indent: int = 4) -> bool:
    """
    Save data as JSON with consistent serialization.
    
    Args:
        data: Data to save
        filepath: Path to save the file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=DateTimeEncoder, ensure_ascii=False, indent=indent)
        logger.info(f"Successfully saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        return False

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to load the file from
        
    Returns:
        Loaded data or None if an error occurred
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None 