#!/usr/bin/env python3
"""
Extract only TPTP problems that can be successfully parsed (CNF/FOF only).

This script pre-filters TPTP files to find only those that our parser can handle,
avoiding TFF/THF and other unsupported formats.
"""

import sys
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def check_file_format(file_path):
    """Quick check if file contains only CNF/FOF formulas."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for unsupported formats
        if 'tff(' in content or 'thf(' in content or 'tcf(' in content:
            return False, "Contains unsupported format (TFF/THF/TCF)"
        
        # Check if it has any formulas
        if 'cnf(' not in content and 'fof(' not in content:
            return False, "No CNF or FOF formulas found"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def find_parseable_files(tptp_dir, domain=None, max_files=None):
    """Find TPTP files that can be parsed by our parser."""
    tptp_path = Path(tptp_dir)
    
    # Find Problems directory
    if (tptp_path / "Problems").exists():
        problems_dir = tptp_path / "Problems"
    else:
        problems_dir = tptp_path
    
    # Find all .p files
    if domain:
        pattern = f"{domain}/**/*.p"
        tptp_files = list(problems_dir.glob(pattern))
    else:
        tptp_files = list(problems_dir.glob("**/*.p"))
    
    print(f"Found {len(tptp_files)} .p files")
    
    if max_files:
        tptp_files = tptp_files[:max_files]
    
    # Check each file
    parseable = []
    unparseable = []
    
    for file_path in tqdm(tptp_files, desc="Checking files"):
        ok, reason = check_file_format(file_path)
        if ok:
            parseable.append(file_path)
        else:
            unparseable.append((file_path, reason))
    
    return parseable, unparseable


def main():
    parser = argparse.ArgumentParser(description='Find parseable TPTP problems')
    parser.add_argument('--tptp-dir', type=str, 
                       default=os.getenv('TPTP_PATH', './.data/problems/tptp/TPTP-v9.0.0'),
                       help='Path to TPTP directory')
    parser.add_argument('--domain', type=str, help='Only check specific domain')
    parser.add_argument('--max-files', type=int, help='Maximum files to check')
    parser.add_argument('--output', type=str, help='Output file for parseable file list')
    args = parser.parse_args()
    
    parseable, unparseable = find_parseable_files(args.tptp_dir, args.domain, args.max_files)
    
    print(f"\nSummary:")
    print(f"Parseable files: {len(parseable)}")
    print(f"Unparseable files: {len(unparseable)}")
    
    # Show reasons for unparseable files
    from collections import Counter
    reasons = Counter(reason for _, reason in unparseable)
    print("\nReasons for unparseable files:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")
    
    # Save parseable file list if requested
    if args.output:
        with open(args.output, 'w') as f:
            for file_path in parseable:
                f.write(str(file_path) + '\n')
        print(f"\nParseable file list saved to {args.output}")
    
    # Show some examples
    if parseable:
        print("\nExample parseable files:")
        for f in parseable[:5]:
            print(f"  {f}")


if __name__ == "__main__":
    main()