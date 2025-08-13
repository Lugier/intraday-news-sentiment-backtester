"""
Visualize sentiment analysis results.

This module generates various plots to visualize the sentiment data, including distributions
and time trends.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import logging
from typing import List, Dict, Any
from datetime import datetime

from src.llm.helpers.utilities import save_json

# Configure logging
logger = logging.getLogger(__name__)

# Configure Matplotlib for consistent styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

class SentimentVisualizer:
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the SentimentVisualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_sentiment_distribution_chart(self, sentiment_data: List[Dict[str, Any]], 
                                          ticker: str) -> None:
        """Create and save a sentiment distribution chart."""
        df = pd.DataFrame(sentiment_data)
        sentiment_counts = df['sentiment'].value_counts()
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        
        # Add color to the bars
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        for i, sentiment in enumerate(sentiment_counts.index):
            if sentiment in colors:
                ax.patches[i].set_facecolor(colors[sentiment])
        
        plt.title(f'Sentiment Distribution for {ticker} News')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f'sentiment_distribution_{ticker}.png'))
        plt.close()
    
    def create_sentiment_trend_chart(self, sentiment_data: List[Dict[str, Any]], 
                                   ticker: str) -> None:
        """Create and save a sentiment trend chart over time."""
        df = pd.DataFrame(sentiment_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create a rolling average of sentiment scores
        df['sentiment_score'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        df['rolling_avg'] = df['sentiment_score'].rolling(window=5).mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['rolling_avg'], marker='o', color='blue')
        plt.title(f'Sentiment Trend for {ticker} News')
        plt.xlabel('Date')
        plt.ylabel('Rolling Average Sentiment Score')
        plt.grid(True)
        
        # Add horizontal lines to indicate sentiment thresholds
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.6, label='Positive Threshold')
        plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.6, label='Negative Threshold')
        plt.legend()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f'sentiment_trend_{ticker}.png'))
        plt.close()
    
    def create_daily_sentiment_summary(self, sentiment_data: List[Dict[str, Any]], 
                                      ticker: str) -> None:
        """
        Create and save a daily sentiment summary chart showing the predominant sentiment for each day.
        
        Args:
            sentiment_data: List of sentiment analysis results
            ticker: Stock ticker symbol
        """
        df = pd.DataFrame(sentiment_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract just the date part (without time)
        df['day'] = df['date'].dt.date
        
        # Group by day and get the most common sentiment for each day
        daily_sentiment = df.groupby('day')['sentiment'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral'
        ).reset_index()
        
        # Sort by date
        daily_sentiment = daily_sentiment.sort_values('day')
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Define colors for each sentiment
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        
        # Create a categorical plot
        for i, row in daily_sentiment.iterrows():
            plt.bar(i, 1, color=colors[row['sentiment']])
        
        # Set x-axis ticks and labels
        plt.xticks(range(len(daily_sentiment)), [d.strftime('%Y-%m-%d') for d in daily_sentiment['day']], 
                   rotation=45, ha='right')
        
        # Set title and labels
        plt.title(f'Daily Predominant Sentiment for {ticker}')
        plt.ylabel('Sentiment')
        plt.yticks([])  # Hide y-axis ticks
        plt.tight_layout()
        
        # Add a legend
        patches = [mpatches.Patch(color=color, label=sentiment.capitalize()) 
                  for sentiment, color in colors.items()]
        plt.legend(handles=patches, loc='upper right')
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f'daily_sentiment_{ticker}.png'))
        plt.close()
    
    def save_results(self, sentiment_data: List[Dict[str, Any]], ticker: str) -> None:
        """Save sentiment analysis results in various formats."""
        # Save as CSV
        df = pd.DataFrame(sentiment_data)
        df.to_csv(os.path.join(self.output_dir, f'sentiment_results_{ticker}.csv'), index=False)
        
        # Save as JSON
        with open(os.path.join(self.output_dir, f'sentiment_results_{ticker}.json'), 'w') as f:
            json.dump(sentiment_data, f, indent=4)
        
        # Save summary as text
        with open(os.path.join(self.output_dir, f'sentiment_summary_{ticker}.txt'), 'w') as f:
            f.write(f"Sentiment Analysis Summary for {ticker}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            sentiment_counts = df['sentiment'].value_counts()
            f.write("Sentiment Distribution:\n")
            for sentiment, count in sentiment_counts.items():
                f.write(f"{sentiment}: {count}\n")
            
            f.write("\nDetailed Results:\n")
            for _, row in df.iterrows():
                f.write(f"\nTitle: {row['title']}\n")
                f.write(f"Date: {row['date']}\n")
                f.write(f"Sentiment: {row['sentiment']}\n")
                f.write(f"Explanation: {row['explanation']}\n")
                f.write("-" * 80 + "\n") 