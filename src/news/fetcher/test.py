#!/usr/bin/env python3

import requests
import time
import datetime
import argparse
from typing import List, Dict, Any, Optional

class OldestNewsFinder:
    """Class to find the oldest available news for a ticker using TickerTick API."""
    
    BASE_URL = "https://api.tickertick.com/feed"
    
    def __init__(self, ticker: str = "AAPL"):
        self.ticker = ticker.lower()
        self.oldest_story_id = None
        self.oldest_story_time = None
        self.stories_fetched = 0
        self.no_more_results = False
        self.fetch_size = 100  # Maximum allowed by the API
    
    def fetch_stories(self, last_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch a batch of stories from the TickerTick API."""
        params = {
            "q": f"(and z:{self.ticker} T:curated)",  # Using z: for exact ticker matches AND T:curated for curated stories
            "n": self.fetch_size
        }
        
        if last_id:
            params["last"] = last_id
            
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("stories"):
                self.no_more_results = True
                return []
                
            return data.get("stories", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return []
    
    def find_oldest_news(self, max_iterations: int = 100) -> Dict[str, Any]:
        """Find the oldest news story for the ticker by paginating through results."""
        print(f"Searching for oldest news for {self.ticker.upper()}...")
        print("This may take some time due to rate limits (10 requests per minute)")
        
        iteration = 0
        oldest_story = None
        
        while not self.no_more_results and iteration < max_iterations:
            iteration += 1
            
            # Fetch next batch of stories
            stories = self.fetch_stories(self.oldest_story_id)
            self.stories_fetched += len(stories)
            
            if not stories:
                break
                
            # Update oldest story information
            oldest_in_batch = min(stories, key=lambda x: x.get("time", 0))
            
            if not oldest_story or oldest_in_batch["time"] < oldest_story["time"]:
                oldest_story = oldest_in_batch
                self.oldest_story_id = oldest_story["id"]
                self.oldest_story_time = oldest_story["time"]
            
            # Print progress
            oldest_date = self.format_timestamp(self.oldest_story_time)
            print(f"Iteration {iteration} | Stories so far: {self.stories_fetched} | Oldest: {oldest_date}")
            
            # Respect rate limits (10 requests per minute) by adding a 13-second pause between requests
            if iteration < max_iterations and not self.no_more_results:
                print("Waiting 13 seconds to respect rate limits...")
                time.sleep(13)
                
        return oldest_story
    
    @staticmethod
    def format_timestamp(timestamp_ms: int) -> str:
        """Format a timestamp in milliseconds to a human-readable date string."""
        dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def print_results(self, oldest_story: Dict[str, Any]):
        """Print the results of the search."""
        if not oldest_story:
            print(f"No news found for ticker: {self.ticker.upper()}")
            return
            
        print("\n" + "="*60)
        print(f"Results for ticker: {self.ticker.upper()}")
        print("="*60)
        print(f"Total stories fetched: {self.stories_fetched}")
        
        oldest_date = self.format_timestamp(oldest_story["time"])
        print(f"Oldest news date: {oldest_date}")
        print(f"Title: {oldest_story['title']}")
        print(f"Source: {oldest_story['site']}")
        print(f"URL: {oldest_story['url']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Find the oldest news available for a stock ticker")
    parser.add_argument("--ticker", type=str, default="AAPL", 
                        help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--max-iterations", type=int, default=100,
                        help="Maximum number of API requests to make (default: 100)")
    args = parser.parse_args()
    
    finder = OldestNewsFinder(ticker=args.ticker)
    oldest_story = finder.find_oldest_news(max_iterations=args.max_iterations)
    finder.print_results(oldest_story)


if __name__ == "__main__":
    main() 