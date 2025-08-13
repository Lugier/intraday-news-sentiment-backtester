#!/usr/bin/env python3
"""
Standalone script to run portfolio-level statistical analysis.
Can be called from main.py or executed independently.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root directory to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.portfolio_event_study_statistics import run_portfolio_statistics_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run portfolio statistics analysis."""
    if len(sys.argv) < 2:
        print("Usage: python run_portfolio_statistics.py <run_directory>")
        print("Example: python run_portfolio_statistics.py output/run_LOG_20250617_162150")
        return
    
    run_dir = sys.argv[1]
    
    if not os.path.exists(run_dir):
        logger.error(f"Run directory does not exist: {run_dir}")
        return
    
    logger.info(f"Starting portfolio statistical analysis for: {run_dir}")
    
    try:
        results = run_portfolio_statistics_analysis(run_dir)
        
        if results:
            logger.info("Portfolio statistical analysis completed successfully!")
            
            # Print quick summary
            meta = results.get('meta_statistics', {})
            print(f"\nQuick Summary:")
            print(f"- Stocks analyzed: {meta.get('total_stocks_analyzed', 0)}")
            print(f"- Total events: {meta.get('total_events', 0)}")
            print(f"- Event distribution: {meta.get('event_distribution', {})}")
            
            # Show significant results
            t_tests = results.get('portfolio_t_tests', {}).get('aar_tests', {})
            significant_results = []
            
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in t_tests:
                    for window, test_result in t_tests[sentiment].items():
                        if test_result.get('significant_5pct', False):
                            significant_results.append(
                                f"{sentiment} {window}min: t={test_result['t_statistic']:.3f}, "
                                f"p={test_result['p_value']:.4f}"
                            )
            
            if significant_results:
                print(f"\nStatistically Significant Results (p < 0.05):")
                for result in significant_results:
                    print(f"- {result}")
            else:
                print(f"\nNo statistically significant results found at p < 0.05 level.")
                
        else:
            logger.error("Portfolio statistical analysis failed!")
            
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 