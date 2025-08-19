#!/usr/bin/env python3
"""
Categorize TPTP problems into different types for benchmarking.

Categories:
1. Unit equalities: All clauses are unit clauses with equality
2. CNF without equality: CNF problems with no equality
3. CNF with equality: CNF problems containing equality
4. FOF without equality: FOF problems with no equality
5. FOF with equality: FOF problems containing equality
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Set, List, Tuple

def get_tptp_path() -> Path:
    """Get the TPTP directory path."""
    base_path = Path(__file__).parent.parent / ".data/problems/tptp/TPTP-v9.0.0/Problems"
    if not base_path.exists():
        raise FileNotFoundError(f"TPTP directory not found: {base_path}")
    return base_path

def get_included_files(content: str, problem_path: Path) -> List[Path]:
    """Extract included files from a TPTP problem."""
    includes = []
    include_pattern = re.compile(r"include\s*\(\s*'([^']+)'")
    
    # Get the TPTP root directory (up to TPTP-v9.0.0)
    tptp_root = problem_path
    while tptp_root.name not in ['TPTP-v9.0.0', 'tptp'] and tptp_root.parent != tptp_root:
        tptp_root = tptp_root.parent
    
    for match in include_pattern.finditer(content):
        include_file = match.group(1)
        
        # Try different resolution strategies
        possible_paths = []
        
        # 1. Relative to current file's directory
        possible_paths.append(problem_path.parent / include_file)
        
        # 2. Relative to TPTP root
        if tptp_root != problem_path:
            possible_paths.append(tptp_root / include_file)
        
        # 3. In the Axioms directory (common pattern)
        if 'Axioms/' not in include_file:
            possible_paths.append(tptp_root / 'Axioms' / include_file)
        
        # Find the first existing path
        for path in possible_paths:
            try:
                resolved_path = path.resolve()
                if resolved_path.exists():
                    includes.append(resolved_path)
                    break
            except:
                pass
    
    return includes

def has_equality_recursive(content: str, problem_path: Path, visited: Set[Path]) -> bool:
    """Check if the problem or its includes contain equality."""
    # Avoid infinite recursion
    if problem_path in visited:
        return False
    visited.add(problem_path)
    
    # Check this file
    lines = content.split('\n')
    for line in lines:
        # Skip comments
        if line.strip().startswith('%'):
            continue
        # Skip include lines themselves
        if line.strip().startswith('include'):
            continue
        
        # Look for equality patterns but not implications
        # We need to find = that is not part of => or <=
        if '=' in line:
            # Make sure it's not in a comment
            comment_pos = line.find('%')
            if comment_pos != -1 and line.index('=') >= comment_pos:
                continue
                
            # Check for actual equality (not part of => or <=)
            for i, char in enumerate(line):
                if char == '=':
                    # Check it's not part of => or <=
                    prev_char = line[i-1] if i > 0 else ' '
                    next_char = line[i+1] if i < len(line)-1 else ' '
                    
                    if prev_char not in ['<', '!', '='] and next_char != '>':
                        # This is a real equality
                        return True
                    elif prev_char == '!' and next_char != '>':
                        # This is != (inequality)
                        return True
    
    # Check included files
    for include_path in get_included_files(content, problem_path):
        try:
            include_content = include_path.read_text(encoding='utf-8', errors='ignore')
            if has_equality_recursive(include_content, include_path, visited):
                return True
        except:
            pass
    
    return False

def has_equality(content: str, problem_path: Path) -> bool:
    """Check if the problem contains equality."""
    visited = set()
    return has_equality_recursive(content, problem_path, visited)

def is_unit_clause(clause_content: str) -> bool:
    """Check if a CNF clause is a unit clause."""
    # Remove surrounding parentheses and whitespace
    clause_content = clause_content.strip()
    if clause_content.startswith('(') and clause_content.endswith(')'):
        clause_content = clause_content[1:-1].strip()
    
    # Check for disjunction (|)
    # If there's a | at the top level, it's not a unit clause
    paren_depth = 0
    for char in clause_content:
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == '|' and paren_depth == 0:
            return False
    
    return True


def is_all_unit_equalities(content: str, problem_path: Path) -> bool:
    """Check if this is a unit equality problem.
    
    A unit equality problem has:
    1. Only CNF clauses (no FOF)
    2. All clauses are unit clauses (no disjunctions)
    3. Most clauses contain equality (allowing for negated conjectures without equality)
    """
    visited = set()
    cnf_pattern = re.compile(r'cnf\s*\([^,]+,\s*[^,]+,\s*(.+?)\s*\)\s*\.', re.DOTALL)
    
    # Count statistics
    total_clauses = 0
    unit_clauses = 0
    clauses_with_equality = 0
    
    # Check main file and includes
    def count_clauses_recursive(content: str, path: Path):
        nonlocal total_clauses, unit_clauses, clauses_with_equality
        
        if path in visited:
            return
        visited.add(path)
        
        for match in cnf_pattern.finditer(content):
            total_clauses += 1
            clause_content = match.group(1).strip()
            
            # Check if unit clause
            if is_unit_clause(clause_content):
                unit_clauses += 1
                
                # Check for equality or inequality
                has_eq = False
                for i, char in enumerate(clause_content):
                    if char == '=':
                        prev_char = clause_content[i-1] if i > 0 else ' '
                        next_char = clause_content[i+1] if i < len(clause_content)-1 else ' '
                        if prev_char not in ['<', '='] and next_char != '>':
                            # This is either = or !=
                            has_eq = True
                            break
                
                if has_eq:
                    clauses_with_equality += 1
        
        # Check includes
        for include_path in get_included_files(content, path):
            try:
                include_content = include_path.read_text(encoding='utf-8', errors='ignore')
                count_clauses_recursive(include_content, include_path)
            except:
                pass
    
    count_clauses_recursive(content, problem_path)
    
    # Criteria for unit equality problem:
    # 1. Must have CNF clauses
    # 2. All clauses must be unit clauses
    # 3. At least 80% of clauses should have equality (to allow for negated conjectures)
    if total_clauses == 0:
        return False
    
    if unit_clauses != total_clauses:
        return False
    
    # Require ALL clauses to have equality/inequality
    equality_ratio = clauses_with_equality / total_clauses
    return equality_ratio == 1.0

def has_fof_recursive(content: str, problem_path: Path, visited: Set[Path]) -> bool:
    """Check if the problem or its includes contain FOF formulas."""
    if problem_path in visited:
        return False
    visited.add(problem_path)
    
    if 'fof(' in content:
        return True
    
    # Check included files
    for include_path in get_included_files(content, problem_path):
        try:
            include_content = include_path.read_text(encoding='utf-8', errors='ignore')
            if has_fof_recursive(include_content, include_path, visited):
                return True
        except:
            pass
    
    return False

def get_problem_type(content: str, problem_path: Path) -> str:
    """Determine the type of a TPTP problem."""
    # Check for FOF formulas (including in included files)
    visited_fof = set()
    has_fof = has_fof_recursive(content, problem_path, visited_fof)
    has_cnf = 'cnf(' in content
    
    has_eq = has_equality(content, problem_path)
    
    # If there's any FOF (in main file or includes), treat it as a FOF problem
    if has_fof:
        if has_eq:
            return "fof_with_equality"
        else:
            return "fof_without_equality"
    elif has_cnf:
        # Pure CNF problem
        # Check if all clauses are unit equalities
        if is_all_unit_equalities(content, problem_path):
            return "unit_equalities"
        elif has_eq:
            return "cnf_with_equality"
        else:
            return "cnf_without_equality"
    else:
        return "other"

def categorize_problems() -> dict:
    """Categorize all TPTP problems."""
    tptp_path = get_tptp_path()
    categories = defaultdict(list)
    
    # Iterate through all problem directories
    for domain_dir in sorted(tptp_path.iterdir()):
        if not domain_dir.is_dir():
            continue
            
        print(f"Processing domain: {domain_dir.name}")
        
        # Process all .p files in the domain
        for problem_file in sorted(domain_dir.glob("*.p")):
            try:
                content = problem_file.read_text(encoding='utf-8', errors='ignore')
                problem_type = get_problem_type(content, problem_file)
                
                if problem_type != "other":
                    # Store relative path from Problems directory
                    relative_path = problem_file.relative_to(tptp_path)
                    categories[problem_type].append(str(relative_path))
                    
            except Exception as e:
                print(f"Error processing {problem_file}: {e}")
    
    return categories

def write_problem_lists(categories: dict):
    """Write categorized problem lists to files."""
    output_dir = Path(__file__).parent.parent / ".data/benchmark_lists"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write each category to a separate file
    for category, problems in categories.items():
        output_file = output_dir / f"{category}_problems.txt"
        with open(output_file, 'w') as f:
            f.write(f"# {category.replace('_', ' ').title()} Problems\n")
            f.write(f"# Total: {len(problems)} problems\n\n")
            for problem in sorted(problems):
                f.write(f"{problem}\n")
        
        print(f"Wrote {len(problems)} {category} problems to {output_file}")
    
    # Write summary
    summary_file = output_dir / "problem_categories_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("TPTP Problem Categories Summary\n")
        f.write("=" * 40 + "\n\n")
        
        total = 0
        for category, problems in sorted(categories.items()):
            f.write(f"{category.replace('_', ' ').title()}: {len(problems)} problems\n")
            total += len(problems)
        
        f.write(f"\nTotal categorized problems: {total}\n")
    
    print(f"\nSummary written to {summary_file}")

def main():
    """Main function."""
    print("Categorizing TPTP problems...")
    print("=" * 40)
    
    categories = categorize_problems()
    
    print("\nCategories found:")
    for category, problems in sorted(categories.items()):
        print(f"  {category}: {len(problems)} problems")
    
    print("\nWriting problem lists...")
    write_problem_lists(categories)
    
    print("\nDone!")

if __name__ == "__main__":
    main()