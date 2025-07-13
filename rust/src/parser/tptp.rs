//! TPTP parser for the standard formula representation

use crate::core::{Term, Variable, Constant, FunctionSymbol, PredicateSymbol, Atom, Literal, Clause, CNFFormula};
use super::fof::{FOFFormula, Quantifier, FormulaRole, NamedFormula};
use super::cnf_conversion::fof_to_cnf;
use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_while1, take_until},
    character::complete::{char, multispace0},
    combinator::{map, value},
    multi::{separated_list0, separated_list1},
    sequence::{preceded, tuple, delimited},
};
use std::path::{Path, PathBuf};
use std::fs;
use std::collections::HashSet;

/// Parse result containing formulas and included files
#[derive(Debug)]
pub struct ParseResult {
    pub cnf_formulas: Vec<Clause>,
    pub fof_formulas: Vec<NamedFormula>,
}

/// Parse a TPTP file with include support
pub fn parse_tptp_file(file_path: &str, include_dirs: &[&str]) -> Result<CNFFormula, String> {
    let mut visited = HashSet::new();
    let result = parse_file_recursive(file_path, include_dirs, &mut visited)?;
    
    // Convert all formulas to CNF
    let mut all_clauses = result.cnf_formulas;
    
    // Convert FOF formulas to CNF
    for named_fof in result.fof_formulas {
        // Negate conjectures
        let formula = match named_fof.role {
            FormulaRole::Conjecture => FOFFormula::Not(Box::new(named_fof.formula)),
            _ => named_fof.formula,
        };
        let cnf = fof_to_cnf(formula);
        all_clauses.extend(cnf.clauses);
    }
    
    Ok(CNFFormula { clauses: all_clauses })
}

/// Parse a TPTP string
pub fn parse_tptp(input: &str) -> Result<CNFFormula, String> {
    parse_tptp_with_includes(input, &[])
}

/// Parse a TPTP string with include directories
pub fn parse_tptp_with_includes(input: &str, include_dirs: &[&str]) -> Result<CNFFormula, String> {
    let mut visited = HashSet::new();
    let result = parse_content(input, include_dirs, &PathBuf::from("."), &mut visited)?;
    
    // Convert all formulas to CNF
    let mut all_clauses = result.cnf_formulas;
    
    // Convert FOF formulas to CNF
    for named_fof in result.fof_formulas {
        // Negate conjectures
        let formula = match named_fof.role {
            FormulaRole::Conjecture => FOFFormula::Not(Box::new(named_fof.formula)),
            _ => named_fof.formula,
        };
        let cnf = fof_to_cnf(formula);
        all_clauses.extend(cnf.clauses);
    }
    
    Ok(CNFFormula { clauses: all_clauses })
}

fn parse_file_recursive(
    file_path: &str,
    include_dirs: &[&str],
    visited: &mut HashSet<PathBuf>,
) -> Result<ParseResult, String> {
    let path = PathBuf::from(file_path);
    
    // Check if we've already visited this file
    if visited.contains(&path) {
        return Ok(ParseResult {
            cnf_formulas: vec![],
            fof_formulas: vec![],
        });
    }
    visited.insert(path.clone());
    
    // Read file
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read file {}: {}", file_path, e))?;
    
    let parent_dir = path.parent().unwrap_or(Path::new("."));
    parse_content(&content, include_dirs, parent_dir, visited)
}

fn parse_content(
    input: &str,
    include_dirs: &[&str],
    current_dir: &Path,
    visited: &mut HashSet<PathBuf>,
) -> Result<ParseResult, String> {
    let mut cnf_formulas = Vec::new();
    let mut fof_formulas = Vec::new();
    
    // Split input into logical statements (ending with '.')
    let mut current_statement = String::new();
    
    for line in input.lines() {
        let line = line.trim();
        
        // Skip comments
        if line.starts_with('%') {
            continue;
        }
        
        // Add line to current statement
        if !line.is_empty() {
            if !current_statement.is_empty() {
                current_statement.push(' ');
            }
            current_statement.push_str(line);
        }
        
        // Check if statement is complete (ends with '.')
        if current_statement.ends_with('.') {
            let statement = current_statement.trim();
            
            // Parse include directive
            if statement.starts_with("include(") {
                let included = parse_include_directive(statement, include_dirs, current_dir, visited)?;
                cnf_formulas.extend(included.cnf_formulas);
                fof_formulas.extend(included.fof_formulas);
            }
            // Parse CNF formula
            else if statement.starts_with("cnf(") {
                match parse_cnf_line(statement) {
                    Ok((_, clause)) => cnf_formulas.push(clause),
                    Err(e) => return Err(format!("Parse error in CNF: {:?}\nStatement: {}", e, statement)),
                }
            }
            // Parse FOF formula
            else if statement.starts_with("fof(") {
                match parse_fof_line(statement) {
                    Ok((_, named_formula)) => fof_formulas.push(named_formula),
                    Err(e) => return Err(format!("Parse error in FOF: {:?}\nStatement: {}", e, statement)),
                }
            }
            
            current_statement.clear();
        }
    }
    
    Ok(ParseResult { cnf_formulas, fof_formulas })
}

fn parse_include_directive(
    line: &str,
    include_dirs: &[&str],
    current_dir: &Path,
    visited: &mut HashSet<PathBuf>,
) -> Result<ParseResult, String> {
    // Parse include('filename').
    let (_, filename) = parse_include(line)
        .map_err(|e| format!("Failed to parse include directive: {:?}", e))?;
    
    // Try to find the file
    let file_path = find_include_file(filename, include_dirs, current_dir)?;
    
    // Recursively parse the included file
    parse_file_recursive(&file_path.to_string_lossy(), include_dirs, visited)
}

fn find_include_file(filename: &str, include_dirs: &[&str], current_dir: &Path) -> Result<PathBuf, String> {
    // Try current directory first
    let path = current_dir.join(filename);
    if path.exists() {
        return Ok(path);
    }
    
    // Try include directories
    for dir in include_dirs {
        let path = Path::new(dir).join(filename);
        if path.exists() {
            return Ok(path);
        }
    }
    
    Err(format!("Include file '{}' not found", filename))
}

/// Parse include directive
fn parse_include(input: &str) -> IResult<&str, &str> {
    let (input, _) = tag("include")(input)?;
    let (input, _) = char('(')(input)?;
    let (input, filename) = delimited(
        char('\''),
        take_until("'"),
        char('\'')
    )(input)?;
    let (input, _) = char(')')(input)?;
    let (input, _) = char('.')(input)?;
    
    Ok((input, filename))
}

/// Parse a single CNF line
fn parse_cnf_line(input: &str) -> IResult<&str, Clause> {
    let (input, _) = tag("cnf")(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = parse_name(input)?; // formula name
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = parse_role(input)?; // formula role
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, clause) = parse_clause(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?;
    let (input, _) = char('.')(input)?;
    
    Ok((input, clause))
}

/// Parse a single FOF line
fn parse_fof_line(input: &str) -> IResult<&str, NamedFormula> {
    let (input, _) = tag("fof")(input)?;
    let (input, _) = char('(')(input)?;
    let (input, name) = parse_name(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, role) = parse_formula_role(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, formula) = parse_fof_formula(input)?;
    let (input, _) = char(')')(input)?;
    let (input, _) = char('.')(input)?;
    
    Ok((input, NamedFormula {
        name: name.to_string(),
        role,
        formula,
    }))
}

/// Parse FOF formula
fn parse_fof_formula(input: &str) -> IResult<&str, FOFFormula> {
    parse_fof_binary(input)
}

/// Parse binary formula (handles precedence)
fn parse_fof_binary(input: &str) -> IResult<&str, FOFFormula> {
    let (input, left) = parse_fof_unary(input)?;
    let (input, _) = multispace0(input)?;
    
    // Try to parse binary operator
    if let Ok((input, _)) = tag::<_, _, nom::error::Error<_>>("<=>")(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::Iff(Box::new(left), Box::new(right))))
    } else if let Ok((input, _)) = tag::<_, _, nom::error::Error<_>>("=>")(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::Implies(Box::new(left), Box::new(right))))
    } else if let Ok((input, _)) = char::<_, nom::error::Error<_>>('|')(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::Or(Box::new(left), Box::new(right))))
    } else if let Ok((input, _)) = char::<_, nom::error::Error<_>>('&')(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::And(Box::new(left), Box::new(right))))
    } else {
        Ok((input, left))
    }
}

/// Parse unary formula
fn parse_fof_unary(input: &str) -> IResult<&str, FOFFormula> {
    alt((
        // Negation
        map(
            preceded(char('~'), parse_fof_unary),
            |f| FOFFormula::Not(Box::new(f))
        ),
        // Quantified formula
        parse_fof_quantified,
        // Parenthesized formula
        delimited(
            tuple((char('('), multispace0)),
            parse_fof_formula,
            tuple((multispace0, char(')')))
        ),
        // Atomic formula
        map(parse_atom, FOFFormula::Atom),
    ))(input)
}

/// Parse quantified formula
fn parse_fof_quantified(input: &str) -> IResult<&str, FOFFormula> {
    let (input, quantifier) = alt((
        value(Quantifier::Forall, char('!')),
        value(Quantifier::Exists, char('?')),
    ))(input)?;
    
    let (input, _) = multispace0(input)?;
    let (input, _) = char('[')(input)?;
    let (input, vars) = separated_list1(
        tuple((multispace0, char(','), multispace0)),
        parse_variable_name
    )(input)?;
    let (input, _) = char(']')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, formula) = parse_fof_unary(input)?;
    
    // Build nested quantifiers for multiple variables
    let mut result = formula;
    for var_name in vars.into_iter().rev() {
        result = FOFFormula::Quantified(
            quantifier.clone(),
            Variable { name: var_name.to_string() },
            Box::new(result)
        );
    }
    
    Ok((input, result))
}

/// Parse variable name
fn parse_variable_name(input: &str) -> IResult<&str, &str> {
    parse_uppercase_ident(input)
}

/// Parse formula role
fn parse_formula_role(input: &str) -> IResult<&str, FormulaRole> {
    alt((
        value(FormulaRole::Axiom, tag("axiom")),
        value(FormulaRole::Hypothesis, tag("hypothesis")),
        value(FormulaRole::Definition, tag("definition")),
        value(FormulaRole::Assumption, tag("assumption")),
        value(FormulaRole::Lemma, tag("lemma")),
        value(FormulaRole::Theorem, tag("theorem")),
        value(FormulaRole::Conjecture, tag("conjecture")),
        value(FormulaRole::NegatedConjecture, tag("negated_conjecture")),
        value(FormulaRole::Plain, tag("plain")),
        value(FormulaRole::Type, tag("type")),
        value(FormulaRole::Unknown, tag("unknown")),
    ))(input)
}

/// Parse a name (alphanumeric + underscore)
fn parse_name(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)
}

/// Parse a role (for backward compatibility)
fn parse_role(input: &str) -> IResult<&str, &str> {
    alt((
        tag("axiom"),
        tag("hypothesis"),
        tag("definition"),
        tag("assumption"),
        tag("lemma"),
        tag("theorem"),
        tag("conjecture"),
        tag("negated_conjecture"),
        tag("plain"),
        tag("type"),
        tag("unknown"),
    ))(input)
}

/// Parse a clause (disjunction of literals)
fn parse_clause(input: &str) -> IResult<&str, Clause> {
    alt((
        // Parenthesized clause
        delimited(
            tuple((char('('), multispace0)),
            separated_list1(
                tuple((multispace0, char('|'), multispace0)),
                parse_literal
            ),
            tuple((multispace0, char(')')))
        ),
        // Non-parenthesized clause
        separated_list1(
            tuple((multispace0, char('|'), multispace0)),
            parse_literal
        ),
    ))(input)
        .map(|(remaining, literals)| (remaining, Clause::new(literals)))
}

/// Parse a literal
fn parse_literal(input: &str) -> IResult<&str, Literal> {
    let (input, _) = multispace0(input)?;
    alt((
        // ~atom
        map(
            preceded(
                tuple((char('~'), multispace0)),
                parse_atom
            ),
            |atom| Literal::negative(atom)
        ),
        // term != term
        parse_negative_equality,
        // atom
        map(parse_atom, |atom| Literal::positive(atom)),
    ))(input)
}

/// Parse a negative equality (term != term)
fn parse_negative_equality(input: &str) -> IResult<&str, Literal> {
    let (input, left) = parse_term(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("!=")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, right) = parse_term(input)?;
    
    let eq_pred = PredicateSymbol {
        name: "=".to_string(),
        arity: 2,
    };
    
    Ok((input, Literal::negative(Atom {
        predicate: eq_pred,
        args: vec![left, right],
    })))
}

/// Parse an atom
fn parse_atom(input: &str) -> IResult<&str, Atom> {
    alt((
        parse_equality,
        parse_predicate,
    ))(input)
}

/// Parse an equality atom
fn parse_equality(input: &str) -> IResult<&str, Atom> {
    let (input, left) = parse_term(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("=")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, right) = parse_term(input)?;
    
    let eq_pred = PredicateSymbol {
        name: "=".to_string(),
        arity: 2,
    };
    
    Ok((input, Atom {
        predicate: eq_pred,
        args: vec![left, right],
    }))
}

/// Parse a predicate
fn parse_predicate(input: &str) -> IResult<&str, Atom> {
    let (input, name) = parse_identifier(input)?;
    
    // Check if it has arguments
    if let Ok((input, _)) = char::<&str, nom::error::Error<&str>>('(')(input) {
        let (input, args) = separated_list0(
            tuple((multispace0, char(','), multispace0)),
            parse_term
        )(input)?;
        let (input, _) = char(')')(input)?;
        
        Ok((input, Atom {
            predicate: PredicateSymbol {
                name: name.to_string(),
                arity: args.len(),
            },
            args,
        }))
    } else {
        // Propositional atom
        Ok((input, Atom {
            predicate: PredicateSymbol {
                name: name.to_string(),
                arity: 0,
            },
            args: vec![],
        }))
    }
}

/// Parse a term
fn parse_term(input: &str) -> IResult<&str, Term> {
    alt((
        parse_function_term,
        parse_variable_term,
        parse_constant_term,
    ))(input)
}

/// Parse a function term
fn parse_function_term(input: &str) -> IResult<&str, Term> {
    let (input, name) = parse_lowercase_ident(input)?;
    let (input, _) = char('(')(input)?;
    let (input, args) = separated_list1(
        tuple((multispace0, char(','), multispace0)),
        parse_term
    )(input)?;
    let (input, _) = char(')')(input)?;
    
    Ok((input, Term::Function(
        FunctionSymbol {
            name: name.to_string(),
            arity: args.len(),
        },
        args,
    )))
}

/// Parse a variable term
fn parse_variable_term(input: &str) -> IResult<&str, Term> {
    let (input, name) = parse_uppercase_ident(input)?;
    Ok((input, Term::Variable(Variable {
        name: name.to_string(),
    })))
}

/// Parse a constant term
fn parse_constant_term(input: &str) -> IResult<&str, Term> {
    let (input, name) = parse_lowercase_ident(input)?;
    Ok((input, Term::Constant(Constant {
        name: name.to_string(),
    })))
}

/// Parse an identifier (letters, digits, underscore)
fn parse_identifier(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)
}

/// Parse a lowercase identifier
fn parse_lowercase_ident(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_lowercase() || (c.is_alphanumeric() && !c.is_uppercase()) || c == '_')(input)
}

/// Parse an uppercase identifier (starts with uppercase letter)
fn parse_uppercase_ident(input: &str) -> IResult<&str, &str> {
    // Match an identifier that starts with an uppercase letter
    let mut chars = input.chars();
    if let Some(first) = chars.next() {
        if first.is_uppercase() {
            // Find the end of the identifier
            let mut end = 1;
            while let Some(ch) = chars.next() {
                if ch.is_alphanumeric() || ch == '_' {
                    end += ch.len_utf8();
                } else {
                    break;
                }
            }
            return Ok((&input[end..], &input[..end]));
        }
    }
    Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::TakeWhile1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_simple_clause() {
        let input = "cnf(test, axiom, p(a)).";
        let result = parse_tptp(input).unwrap();
        assert_eq!(result.clauses.len(), 1);
        assert_eq!(result.clauses[0].literals.len(), 1);
    }
    
    #[test]
    fn test_parse_equality() {
        let input = "cnf(test, axiom, X = f(a)).";
        let result = parse_tptp(input).unwrap();
        assert_eq!(result.clauses.len(), 1);
        assert_eq!(result.clauses[0].literals.len(), 1);
        assert!(result.clauses[0].literals[0].atom.is_equality());
    }
    
    #[test]
    fn test_parse_negation() {
        let input = "cnf(test, axiom, ~p(X) | q(X)).";
        let result = parse_tptp(input).unwrap();
        assert_eq!(result.clauses.len(), 1);
        assert_eq!(result.clauses[0].literals.len(), 2);
        assert!(!result.clauses[0].literals[0].polarity);
        assert!(result.clauses[0].literals[1].polarity);
    }
    
    #[test]
    fn test_parse_fof_conjunction() {
        let input = "fof(test, axiom, p(a) & q(b)).";
        let result = parse_tptp(input).unwrap();
        // Should produce two unit clauses after CNF conversion
        assert_eq!(result.clauses.len(), 2);
    }
    
    #[test]
    fn test_parse_fof_quantified() {
        let input = "fof(test, axiom, ![X]: p(X)).";
        let result = parse_tptp(input).unwrap();
        assert_eq!(result.clauses.len(), 1);
    }
    
    #[test]
    fn test_parse_include() {
        let input = "include('test.ax').";
        let (remaining, filename) = parse_include(input).unwrap();
        assert_eq!(filename, "test.ax");
        assert_eq!(remaining, "");
    }
}