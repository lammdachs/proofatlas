#!/usr/bin/env python3
"""
Summarize extracted TPTP problems from JSON files.

This script reads the extracted JSON files and provides statistics
about the problems that were successfully parsed.
"""

import json
import sys
import os
from pathlib import Path
import argparse
from collections import Counter

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def analyze_problems(problems):
    """Analyze a list of problems and return statistics."""
    stats = {
        'total_problems': len(problems),
        'total_clauses': 0,
        'clause_size_distribution': Counter(),
        'predicate_distribution': Counter(),
        'function_distribution': Counter(),
        'constant_distribution': Counter(),
        'problems_by_clause_count': Counter(),
        'domain_distribution': Counter()
    }
    
    for problem in problems:
        domain = problem['source_file'].split('/')[0]
        stats['domain_distribution'][domain] += 1
        
        num_clauses = problem['statistics']['num_clauses']
        stats['total_clauses'] += num_clauses
        stats['problems_by_clause_count'][num_clauses] += 1
        
        max_clause_size = problem['statistics']['max_clause_size']
        stats['clause_size_distribution'][max_clause_size] += 1
        
        # Count unique predicates/functions
        for pred_name, arity in problem['statistics']['predicates']:
            stats['predicate_distribution'][f"{pred_name}/{arity}"] += 1
            
        for func_name, arity in problem['statistics']['functions']:
            stats['function_distribution'][f"{func_name}/{arity}"] += 1
            
        for const_name in problem['statistics']['constants']:
            stats['constant_distribution'][const_name] += 1
    
    return stats


def print_top_items(counter, name, limit=10):
    """Print top items from a counter."""
    print(f"\nTop {limit} {name}:")
    for item, count in counter.most_common(limit):
        print(f"  {item}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Summarize extracted TPTP problems')
    parser.add_argument('input_file', type=str, help='Input JSON file with extracted problems')
    parser.add_argument('--top', type=int, default=10, help='Number of top items to show')
    args = parser.parse_args()
    
    # Load problems
    print(f"Loading problems from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        problems = json.load(f)
    
    # Analyze
    stats = analyze_problems(problems)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TPTP EXTRACTION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nTotal problems: {stats['total_problems']}")
    print(f"Total clauses: {stats['total_clauses']}")
    print(f"Average clauses per problem: {stats['total_clauses'] / stats['total_problems']:.1f}")
    
    print("\nProblems by domain:")
    for domain, count in sorted(stats['domain_distribution'].items()):
        print(f"  {domain}: {count}")
    
    print_top_items(stats['clause_size_distribution'], "clause sizes", args.top)
    print_top_items(stats['problems_by_clause_count'], "problems by clause count", args.top)
    print_top_items(stats['predicate_distribution'], "predicates", args.top)
    print_top_items(stats['function_distribution'], "functions", args.top)
    
    if stats['constant_distribution']:
        print_top_items(stats['constant_distribution'], "constants", args.top)
    
    # Find some interesting problems
    print("\n\nExample problems:")
    
    # Small problem
    small_problems = [p for p in problems if p['statistics']['num_clauses'] < 10]
    if small_problems:
        print(f"\nSmall problem ({small_problems[0]['source_file']}):")
        print(f"  Clauses: {small_problems[0]['statistics']['num_clauses']}")
        print(f"  Predicates: {small_problems[0]['statistics']['predicates']}")
    
    # Large problem
    large_problems = sorted(problems, key=lambda p: p['statistics']['num_clauses'], reverse=True)
    if large_problems:
        print(f"\nLarge problem ({large_problems[0]['source_file']}):")
        print(f"  Clauses: {large_problems[0]['statistics']['num_clauses']}")
        print(f"  Max clause size: {large_problems[0]['statistics']['max_clause_size']}")


if __name__ == "__main__":
    main()