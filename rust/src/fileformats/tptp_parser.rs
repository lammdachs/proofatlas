//! TPTP parser implementation using nom
//! 
//! This parser supports:
//! - CNF (Clause Normal Form) formulas
//! - FOF (First Order Form) formulas with automatic CNF conversion
//! - Include directives
//! 
//! Not supported:
//! - TFF (Typed First-order Form) - not planned
//! - THF (Typed Higher-order Form) - not planned

use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_until, take_while},
    character::complete::{char, multispace0, one_of},
    combinator::{map, opt},
    multi::separated_list1,
    sequence::{delimited, preceded, tuple},
};
use std::fs;
use std::path::Path;
use std::collections::HashSet;
use crate::core::{logic::*, error::{ProofAtlasError, Result}};
use crate::fileformats::fof::FofFormula;

/// Parse a TPTP file
pub fn parse_file(file_path: &str, include_path: Option<&str>) -> Result<Problem> {
    let content = fs::read_to_string(file_path)?;
    let mut parser_state = ParserState::new(include_path);
    parser_state.parse_file_content(&content, Path::new(file_path))
}

/// Parse TPTP string content
pub fn parse_string(content: &str) -> Result<Problem> {
    let mut parser_state = ParserState::new(None);
    parser_state.parse_string_content(content)
}

/// Parser state to handle includes and track parsed content
struct ParserState {
    include_path: Option<String>,
    visited_files: HashSet<String>,
    clauses: Vec<Clause>,
    conjecture_indices: Vec<usize>,
}

impl ParserState {
    fn new(include_path: Option<&str>) -> Self {
        ParserState {
            include_path: include_path.map(String::from),
            visited_files: HashSet::new(),
            clauses: Vec::new(),
            conjecture_indices: Vec::new(),
        }
    }
    
    fn parse_file_content(&mut self, content: &str, file_path: &Path) -> Result<Problem> {
        // Mark file as visited
        if let Ok(canonical) = file_path.canonicalize() {
            self.visited_files.insert(canonical.display().to_string());
        }
        
        self.parse_string_content(content)?;
        
        Ok(Problem::with_conjectures(
            self.clauses.clone(),
            self.conjecture_indices.clone(),
        ))
    }
    
    fn parse_string_content(&mut self, content: &str) -> Result<Problem> {
        let inputs = parse_tptp_inputs(content)
            .map_err(|e| ProofAtlasError::ParseError(format!("Parse error: {:?}", e)))?;
        
        for input in inputs {
            match input {
                TPTPInput::Cnf { role, clause, .. } => {
                    let clause_idx = self.clauses.len();
                    if role == "negated_conjecture" {
                        self.conjecture_indices.push(clause_idx);
                    }
                    self.clauses.push(clause);
                }
                TPTPInput::Include { file_name } => {
                    // Handle include
                    if let Some(inc_path) = &self.include_path {
                        let full_path = Path::new(inc_path).join(&file_name);
                        if full_path.exists() && !self.visited_files.contains(&full_path.display().to_string()) {
                            let inc_content = fs::read_to_string(&full_path)?;
                            self.parse_file_content(&inc_content, &full_path)?;
                        }
                    }
                }
                TPTPInput::Fof { name, role, formula } => {
                    // Convert FOF to CNF
                    let cnf_clauses = formula.to_cnf();
                    
                    // Add all resulting clauses
                    for clause in cnf_clauses {
                        let clause_idx = self.clauses.len();
                        if role == "negated_conjecture" || role == "conjecture" {
                            self.conjecture_indices.push(clause_idx);
                        }
                        self.clauses.push(clause);
                    }
                }
            }
        }
        
        Ok(Problem::with_conjectures(
            self.clauses.clone(),
            self.conjecture_indices.clone(),
        ))
    }
}

/// TPTP input types
#[derive(Debug, Clone)]
enum TPTPInput {
    Cnf {
        #[allow(dead_code)]
        name: String,
        role: String,
        clause: Clause,
    },
    Fof {
        name: String,
        role: String,
        formula: FofFormula,
    },
    Include {
        file_name: String,
    },
}

/// Parse multiple TPTP inputs
fn parse_tptp_inputs(input: &str) -> Result<Vec<TPTPInput>> {
    let mut results = Vec::new();
    let mut remaining = input;
    
    while !remaining.trim().is_empty() {
        // Skip whitespace and comments
        remaining = skip_whitespace_and_comments(remaining);
        if remaining.is_empty() {
            break;
        }
        
        // Try to parse an input
        match parse_tptp_input(remaining) {
            Ok((rest, Some(input))) => {
                results.push(input);
                remaining = rest;
            }
            Ok((rest, None)) => {
                // Skipped input (comment or empty)
                remaining = rest;
            }
            Err(_) => {
                // Check if this is an unsupported format
                if remaining.starts_with("tff(") || remaining.starts_with("thf(") {
                    // Skip TFF/THF formulas - not supported
                    if let Some(pos) = remaining.find('\n') {
                        remaining = &remaining[pos + 1..];
                    } else {
                        break;
                    }
                } else {
                    // Skip to next line if we can't parse
                    if let Some(pos) = remaining.find('\n') {
                        remaining = &remaining[pos + 1..];
                    } else {
                        break;
                    }
                }
            }
        }
    }
    
    Ok(results)
}

/// Skip whitespace and comments
fn skip_whitespace_and_comments(input: &str) -> &str {
    let mut remaining = input;
    loop {
        let before = remaining.len();
        remaining = remaining.trim_start();
        
        // Skip line comments
        if remaining.starts_with('%') {
            if let Some(pos) = remaining.find('\n') {
                remaining = &remaining[pos + 1..];
            } else {
                return "";
            }
        }
        
        // Skip block comments
        if remaining.starts_with("/*") {
            if let Some(pos) = remaining.find("*/") {
                remaining = &remaining[pos + 2..];
            } else {
                return "";
            }
        }
        
        if remaining.len() == before {
            break;
        }
    }
    remaining
}

/// Parse a single TPTP input
fn parse_tptp_input(input: &str) -> IResult<&str, Option<TPTPInput>> {
    // Skip leading whitespace
    let (input, _) = multispace0(input)?;
    
    alt((
        map(parse_cnf, Some),
        map(parse_fof, Some),
        map(parse_include, Some),
    ))(input)
}

/// Parse CNF formula
fn parse_cnf(input: &str) -> IResult<&str, TPTPInput> {
    let (input, _) = tag("cnf")(input)?;
    let (input, _) = ws(char('('))(input)?;
    let (input, name) = ws(parse_name)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, role) = ws(parse_name)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, clause) = ws(parse_clause)(input)?;
    let (input, _) = ws(char(')'))(input)?;
    let (input, _) = ws(char('.'))(input)?;
    
    Ok((input, TPTPInput::Cnf { name, role, clause }))
}

/// Parse include directive
fn parse_include(input: &str) -> IResult<&str, TPTPInput> {
    let (input, _) = tag("include")(input)?;
    let (input, _) = ws(char('('))(input)?;
    let (input, file_name) = ws(parse_quoted_string)(input)?;
    let (input, _) = ws(char(')'))(input)?;
    let (input, _) = ws(char('.'))(input)?;
    
    Ok((input, TPTPInput::Include { file_name }))
}

/// Parse a clause (disjunction of literals)
fn parse_clause(input: &str) -> IResult<&str, Clause> {
    alt((
        // Empty clause
        map(tag("$false"), |_| Clause::new(vec![])),
        // Parenthesized clause
        delimited(
            ws(char('(')),
            map(
                separated_list1(ws(char('|')), parse_literal),
                Clause::new
            ),
            ws(char(')'))
        ),
        // Single literal
        map(parse_literal, |lit| Clause::new(vec![lit])),
    ))(input)
}

/// Parse a literal
fn parse_literal(input: &str) -> IResult<&str, Literal> {
    alt((
        // Negative literal
        map(
            preceded(ws(char('~')), parse_atom_or_equality),
            |pred| Literal::negative(pred)
        ),
        // Inequality (A != B becomes ~(A = B))
        map(parse_inequality, |pred| Literal::negative(pred)),
        // Positive literal (including equality)
        map(parse_atom_or_equality, |pred| Literal::positive(pred)),
    ))(input)
}

/// Parse atom or equality
fn parse_atom_or_equality(input: &str) -> IResult<&str, Predicate> {
    alt((
        parse_equality,
        parse_atom,
    ))(input)
}

/// Parse equality or inequality
fn parse_equality(input: &str) -> IResult<&str, Predicate> {
    let (input, left) = parse_term(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, right) = parse_term(input)?;
    
    Ok((input, Predicate::new("=".to_string(), vec![left, right])))
}

/// Parse inequality (returns equality predicate, caller will negate)
fn parse_inequality(input: &str) -> IResult<&str, Predicate> {
    let (input, left) = parse_term(input)?;
    let (input, _) = ws(tag("!="))(input)?;
    let (input, right) = parse_term(input)?;
    
    Ok((input, Predicate::new("=".to_string(), vec![left, right])))
}

/// Parse atom (predicate)
fn parse_atom(input: &str) -> IResult<&str, Predicate> {
    let (input, name) = parse_name(input)?;
    let (input, args) = opt(delimited(
        ws(char('(')),
        separated_list1(ws(char(',')), parse_term),
        ws(char(')'))
    ))(input)?;
    
    Ok((input, Predicate::new(name, args.unwrap_or_default())))
}

/// Parse term
fn parse_term(input: &str) -> IResult<&str, Term> {
    alt((
        // Variable (uppercase)
        map(parse_variable, |name| Term::Variable(name)),
        // Function or constant
        parse_function_or_constant,
    ))(input)
}

/// Parse function or constant
fn parse_function_or_constant(input: &str) -> IResult<&str, Term> {
    let (input, name) = parse_name(input)?;
    let (input, args) = opt(delimited(
        ws(char('(')),
        separated_list1(ws(char(',')), parse_term),
        ws(char(')'))
    ))(input)?;
    
    Ok((input, match args {
        Some(args) => Term::Function { name, args },
        None => Term::Constant(name),
    }))
}

/// Parse variable (uppercase letter followed by alphanumerics)
fn parse_variable(input: &str) -> IResult<&str, String> {
    let (input, first) = one_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ")(input)?;
    let (input, rest) = take_while(|c: char| c.is_alphanumeric() || c == '_')(input)?;
    Ok((input, format!("{}{}", first, rest)))
}

/// Parse name (lowercase or quoted)
fn parse_name(input: &str) -> IResult<&str, String> {
    alt((
        parse_quoted_string,
        parse_unquoted_name,
    ))(input)
}

/// Parse unquoted name
fn parse_unquoted_name(input: &str) -> IResult<&str, String> {
    let (input, first) = one_of("abcdefghijklmnopqrstuvwxyz$")(input)?;
    let (input, rest) = take_while(|c: char| c.is_alphanumeric() || c == '_')(input)?;
    Ok((input, format!("{}{}", first, rest)))
}

/// Parse quoted string
fn parse_quoted_string(input: &str) -> IResult<&str, String> {
    alt((
        delimited(char('\''), take_until("'"), char('\'')),
        delimited(char('"'), take_until("\""), char('"')),
    ))(input)
    .map(|(i, s)| (i, s.to_string()))
}

/// Parse FOF formula
fn parse_fof(input: &str) -> IResult<&str, TPTPInput> {
    let (input, _) = tag("fof")(input)?;
    let (input, _) = ws(char('('))(input)?;
    let (input, name) = ws(parse_name)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, role) = ws(parse_name)(input)?;
    let (input, _) = ws(char(','))(input)?;
    let (input, formula) = ws(parse_fof_formula)(input)?;
    let (input, _) = ws(char(')'))(input)?;
    let (input, _) = ws(char('.'))(input)?;
    
    Ok((input, TPTPInput::Fof { name, role, formula }))
}

/// Parse FOF formula
fn parse_fof_formula(input: &str) -> IResult<&str, FofFormula> {
    parse_fof_binary_formula(input)
}

/// Parse FOF binary formula (handles precedence)
fn parse_fof_binary_formula(input: &str) -> IResult<&str, FofFormula> {
    // Start with lowest precedence: <=> (biconditional)
    parse_fof_iff(input)
}

/// Parse biconditional (<=>)
fn parse_fof_iff(input: &str) -> IResult<&str, FofFormula> {
    let (input, left) = parse_fof_implies(input)?;
    
    // Try to parse <=>
    let mut parse_iff_op = ws(tag("<=>"));
    if let Ok((input2, _)) = parse_iff_op(input) {
        let (input2, right) = parse_fof_iff(input2)?;
        Ok((input2, FofFormula::Iff(Box::new(left), Box::new(right))))
    } else {
        Ok((input, left))
    }
}

/// Parse implication (=>)
fn parse_fof_implies(input: &str) -> IResult<&str, FofFormula> {
    let (input, left) = parse_fof_or(input)?;
    
    // Try to parse =>
    let mut parse_implies_op = ws(tag("=>"));
    if let Ok((input2, _)) = parse_implies_op(input) {
        let (input2, right) = parse_fof_implies(input2)?;
        Ok((input2, FofFormula::Implies(Box::new(left), Box::new(right))))
    } else {
        Ok((input, left))
    }
}

/// Parse disjunction (|)
fn parse_fof_or(input: &str) -> IResult<&str, FofFormula> {
    let (input, first) = parse_fof_and(input)?;
    
    // Parse additional | terms
    let mut terms = vec![first];
    let mut remaining = input;
    
    let mut parse_or_op = ws(char('|'));
    while let Ok((input2, _)) = parse_or_op(remaining) {
        let (input2, term) = parse_fof_and(input2)?;
        terms.push(term);
        remaining = input2;
    }
    
    Ok((remaining, if terms.len() == 1 {
        terms.into_iter().next().unwrap()
    } else {
        FofFormula::Or(terms)
    }))
}

/// Parse conjunction (&)
fn parse_fof_and(input: &str) -> IResult<&str, FofFormula> {
    let (input, first) = parse_fof_unary(input)?;
    
    // Parse additional & terms
    let mut terms = vec![first];
    let mut remaining = input;
    
    let mut parse_and_op = ws(char('&'));
    while let Ok((input2, _)) = parse_and_op(remaining) {
        let (input2, term) = parse_fof_unary(input2)?;
        terms.push(term);
        remaining = input2;
    }
    
    Ok((remaining, if terms.len() == 1 {
        terms.into_iter().next().unwrap()
    } else {
        FofFormula::And(terms)
    }))
}

/// Parse unary formula (negation)
fn parse_fof_unary(input: &str) -> IResult<&str, FofFormula> {
    alt((
        // Negation
        map(
            preceded(ws(char('~')), parse_fof_unary),
            |f| FofFormula::Not(Box::new(f))
        ),
        // Continue to higher precedence
        parse_fof_quantified,
    ))(input)
}

/// Parse quantified formula
fn parse_fof_quantified(input: &str) -> IResult<&str, FofFormula> {
    alt((
        // Universal quantification
        map(
            tuple((
                ws(char('!')),
                ws(delimited(char('['), parse_variable_list, char(']'))),
                ws(char(':')),
                parse_fof_unary,
            )),
            |(_, vars, _, formula)| FofFormula::Forall(vars, Box::new(formula))
        ),
        // Existential quantification
        map(
            tuple((
                ws(char('?')),
                ws(delimited(char('['), parse_variable_list, char(']'))),
                ws(char(':')),
                parse_fof_unary,
            )),
            |(_, vars, _, formula)| FofFormula::Exists(vars, Box::new(formula))
        ),
        // Continue to atomic
        parse_fof_atomic,
    ))(input)
}

/// Parse variable list
fn parse_variable_list(input: &str) -> IResult<&str, Vec<String>> {
    separated_list1(ws(char(',')), parse_variable)(input)
}

/// Parse atomic formula
fn parse_fof_atomic(input: &str) -> IResult<&str, FofFormula> {
    alt((
        // Parenthesized formula
        delimited(
            ws(char('(')),
            parse_fof_formula,
            ws(char(')'))
        ),
        // Inequality (parsed as negated equality)
        map(parse_inequality, |pred| FofFormula::Not(Box::new(FofFormula::Atom(pred)))),
        // Equality
        map(parse_equality, FofFormula::Atom),
        // Regular atom
        map(parse_atom, FofFormula::Atom),
    ))(input)
}

/// Whitespace wrapper
fn ws<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(multispace0, inner, multispace0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_variable() {
        assert_eq!(parse_variable("X"), Ok(("", "X".to_string())));
        assert_eq!(parse_variable("X123"), Ok(("", "X123".to_string())));
        assert_eq!(parse_variable("X_var"), Ok(("", "X_var".to_string())));
        assert!(parse_variable("x").is_err());
    }
    
    #[test]
    fn test_parse_name() {
        assert_eq!(parse_name("foo"), Ok(("", "foo".to_string())));
        assert_eq!(parse_name("'quoted name'"), Ok(("", "quoted name".to_string())));
        assert_eq!(parse_name("$true"), Ok(("", "$true".to_string())));
    }
    
    #[test]
    fn test_parse_literal() {
        let (_, lit) = parse_literal("p(X)").unwrap();
        assert!(lit.polarity);
        assert_eq!(lit.predicate.name, "p");
        
        let (_, lit) = parse_literal("~q(a,b)").unwrap();
        assert!(!lit.polarity);
        assert_eq!(lit.predicate.name, "q");
    }
    
    #[test]
    fn test_parse_clause() {
        let (_, clause) = parse_clause("(p(X) | ~q(Y))").unwrap();
        assert_eq!(clause.literals.len(), 2);
        
        let (_, clause) = parse_clause("$false").unwrap();
        assert!(clause.is_empty());
    }
    
    #[test]
    fn test_parse_cnf() {
        let input = "cnf(test, axiom, p(a)).";
        let result = parse_cnf(input);
        assert!(result.is_ok(), "Failed to parse CNF: {:?}", result);
        
        let (remaining, tptp_input) = result.unwrap();
        assert_eq!(remaining, "");
        match tptp_input {
            TPTPInput::Cnf { name, role, clause } => {
                assert_eq!(name, "test");
                assert_eq!(role, "axiom");
                assert_eq!(clause.literals.len(), 1);
            }
            _ => panic!("Expected CNF input"),
        }
    }
    
    #[test]
    fn test_parse_fof_atomic() {
        let (_, formula) = parse_fof_atomic("p(X)").unwrap();
        match formula {
            FofFormula::Atom(pred) => {
                assert_eq!(pred.name, "p");
                assert_eq!(pred.args.len(), 1);
            }
            _ => panic!("Expected atomic formula"),
        }
    }
    
    #[test]
    fn test_parse_fof_negation() {
        let (_, formula) = parse_fof_formula("~p(X)").unwrap();
        match formula {
            FofFormula::Not(f) => {
                match f.as_ref() {
                    FofFormula::Atom(pred) => assert_eq!(pred.name, "p"),
                    _ => panic!("Expected atom inside negation"),
                }
            }
            _ => panic!("Expected negation"),
        }
    }
    
    #[test]
    fn test_parse_fof_binary() {
        // Test conjunction
        let (_, formula) = parse_fof_formula("p(X) & q(Y)").unwrap();
        match formula {
            FofFormula::And(terms) => assert_eq!(terms.len(), 2),
            _ => panic!("Expected conjunction"),
        }
        
        // Test disjunction
        let (_, formula) = parse_fof_formula("p(X) | q(Y)").unwrap();
        match formula {
            FofFormula::Or(terms) => assert_eq!(terms.len(), 2),
            _ => panic!("Expected disjunction"),
        }
        
        // Test implication
        let (_, formula) = parse_fof_formula("p(X) => q(Y)").unwrap();
        match formula {
            FofFormula::Implies(_, _) => {},
            _ => panic!("Expected implication"),
        }
        
        // Test biconditional
        let (_, formula) = parse_fof_formula("p(X) <=> q(Y)").unwrap();
        match formula {
            FofFormula::Iff(_, _) => {},
            _ => panic!("Expected biconditional"),
        }
    }
    
    #[test]
    fn test_parse_fof_quantified() {
        // Test universal quantification
        let (_, formula) = parse_fof_formula("![X]: p(X)").unwrap();
        match formula {
            FofFormula::Forall(vars, _) => {
                assert_eq!(vars.len(), 1);
                assert_eq!(vars[0], "X");
            }
            _ => panic!("Expected universal quantification"),
        }
        
        // Test existential quantification
        let (_, formula) = parse_fof_formula("?[X,Y]: (p(X) & q(Y))").unwrap();
        match formula {
            FofFormula::Exists(vars, _) => {
                assert_eq!(vars.len(), 2);
                assert_eq!(vars[0], "X");
                assert_eq!(vars[1], "Y");
            }
            _ => panic!("Expected existential quantification"),
        }
    }
    
    #[test]
    fn test_parse_fof() {
        let input = "fof(test1, axiom, ![X]: (p(X) => q(X))).";
        let (_, tptp) = parse_fof(input).unwrap();
        match tptp {
            TPTPInput::Fof { name, role, formula } => {
                assert_eq!(name, "test1");
                assert_eq!(role, "axiom");
                match formula {
                    FofFormula::Forall(_, _) => {},
                    _ => panic!("Expected quantified formula"),
                }
            }
            _ => panic!("Expected FOF"),
        }
    }
    
    #[test]
    fn test_parse_mixed_fof_cnf() {
        let input = r#"
% Simple FOF test file

fof(axiom1, axiom, ![X]: (human(X) => mortal(X))).
fof(axiom2, axiom, human(socrates)).
fof(conjecture1, conjecture, mortal(socrates)).

% Test more complex formulas
fof(complex1, axiom, ![X,Y]: ((p(X) & q(Y)) => (r(X) | s(Y)))).
fof(complex2, axiom, ?[X]: (p(X) & ![Y]: (q(Y) => r(X,Y)))).
fof(biconditional, axiom, ![X]: (p(X) <=> (q(X) & r(X)))).

% Mixed CNF and FOF
cnf(cnf_clause, axiom, (p(a) | ~q(b) | r(c))).
"#;
        
        let problem = parse_string(input).unwrap();
        
        // Should have parsed all formulas
        // FOF formulas get converted to CNF, so count the resulting clauses
        assert!(problem.clauses.len() > 0);
        
        // Check that we have at least one conjecture
        assert!(problem.conjecture_indices.len() > 0);
    }
    
    #[test]
    fn test_fof_to_cnf_conversion() {
        // Test that FOF formulas are properly converted to CNF
        let input = "fof(test, axiom, ![X]: (p(X) => q(X))).";
        let problem = parse_string(input).unwrap();
        
        // The implication should be converted to ~p(X) | q(X)
        assert_eq!(problem.clauses.len(), 1);
        let clause = &problem.clauses[0];
        assert_eq!(clause.literals.len(), 2);
        
        // Should have one negative and one positive literal
        let polarities: Vec<_> = clause.literals.iter().map(|l| l.polarity).collect();
        assert!(polarities.contains(&true));
        assert!(polarities.contains(&false));
    }
    
    #[test]
    #[ignore = "requires external test file"]
    fn test_parse_file_with_fof() {
        // Test parsing the actual test file
        let result = parse_file("../test_fof.p", None);
        
        match result {
            Ok(problem) => {
                // Should have parsed multiple clauses
                assert!(problem.clauses.len() > 0);
                // Should have one conjecture
                assert_eq!(problem.conjecture_indices.len(), 1);
            },
            Err(e) => panic!("Failed to parse file: {:?}", e),
        }
    }
}