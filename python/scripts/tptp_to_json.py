#!/usr/bin/env python3
"""
Convert TPTP files to JSON format.

This script parses TPTP problem files and converts them to JSON format,
preserving clause structure and conjecture information.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import proofatlas_rust
except ImportError:
    print("Error: proofatlas_rust module not found.")
    print("Please build it with: cd rust && maturin develop")
    sys.exit(1)


def convert_tptp_to_json(tptp_path: str, include_path: str = None, output_path: str = None) -> dict:
    """
    Convert a TPTP file to JSON format.
    
    Args:
        tptp_path: Path to the TPTP file
        include_path: Path to TPTP library for includes (optional)
        output_path: Path to save JSON file (optional)
    
    Returns:
        Dictionary containing the parsed problem data
    """
    # Parse the TPTP file
    try:
        problem = proofatlas_rust.parser.parse_file(tptp_path, include_path)
    except Exception as e:
        raise ValueError(f"Failed to parse TPTP file: {e}")
    
    # Get JSON representation
    json_str = problem.to_json()
    data = json.loads(json_str)
    
    # Add metadata
    data['metadata'] = {
        'source_file': os.path.basename(tptp_path),
        'source_path': os.path.abspath(tptp_path),
        'num_clauses': len(problem),
        'num_literals': sum(len(clause['literals']) for clause in data['clauses']),
        'num_conjectures': len(data['conjecture_indices'])
    }
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {output_path}")
    
    return data


def batch_convert(tptp_dir: str, output_dir: str, include_path: str = None, pattern: str = "*.p"):
    """
    Convert multiple TPTP files in a directory.
    
    Args:
        tptp_dir: Directory containing TPTP files
        output_dir: Directory to save JSON files
        include_path: Path to TPTP library for includes
        pattern: File pattern to match (default: *.p)
    """
    tptp_path = Path(tptp_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    files = list(tptp_path.glob(pattern))
    if not files:
        files = list(tptp_path.rglob(pattern))  # Recursive search
    
    print(f"Found {len(files)} TPTP files to convert")
    
    successful = 0
    failed = 0
    
    for i, tptp_file in enumerate(files, 1):
        print(f"\r[{i}/{len(files)}] Converting {tptp_file.name}...", end='', flush=True)
        
        # Generate output filename
        relative_path = tptp_file.relative_to(tptp_path)
        json_path = output_path / relative_path.with_suffix('.json')
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            convert_tptp_to_json(str(tptp_file), include_path, str(json_path))
            successful += 1
        except Exception as e:
            print(f"\n  Failed: {e}")
            failed += 1
    
    print(f"\n\nConversion complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TPTP files to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  %(prog)s problem.p -o problem.json
  
  # Convert with TPTP includes
  %(prog)s problem.p -i /path/to/TPTP -o problem.json
  
  # Batch convert a directory
  %(prog)s --batch /path/to/problems -o /path/to/json -i /path/to/TPTP
  
  # Convert specific category
  %(prog)s --batch /path/to/TPTP/Problems/SET -o ./json/SET
"""
    )
    
    parser.add_argument('input', nargs='?', help='Input TPTP file')
    parser.add_argument('-o', '--output', help='Output JSON file or directory')
    parser.add_argument('-i', '--include', help='TPTP include directory')
    parser.add_argument('--batch', help='Batch convert directory')
    parser.add_argument('--pattern', default='*.p', help='File pattern for batch mode (default: *.p)')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.batch:
        parser.error("Either input file or --batch directory must be specified")
    
    if args.batch:
        if not args.output:
            parser.error("--output directory required for batch mode")
        batch_convert(args.batch, args.output, args.include, args.pattern)
    else:
        # Single file conversion
        try:
            data = convert_tptp_to_json(args.input, args.include, args.output)
            
            # Print to stdout if no output file specified
            if not args.output:
                if args.pretty:
                    print(json.dumps(data, indent=2))
                else:
                    print(json.dumps(data))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()