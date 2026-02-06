//! TPTP parser for the standard formula representation

use super::cnf_conversion::fof_to_cnf_with_role;
use super::fof::{FOFFormula, FormulaRole, NamedFormula, Quantifier};
use crate::logic::ordering::orient_equalities::orient_all_equalities;
use crate::logic::{
    Atom, CNFFormula, Clause, ClauseRole, Constant, FunctionSymbol, Interner, Literal,
    PredicateSymbol, Term, Variable,
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{char, multispace0},
    combinator::{map, value},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, preceded, tuple},
    IResult,
};
use std::cell::RefCell;
use std::ops::Deref;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Result of parsing a TPTP problem
#[derive(Debug)]
pub struct ParsedProblem {
    pub formula: CNFFormula,
    pub interner: Interner,
}

/// Parse result containing formulas and included files (internal)
#[derive(Debug)]
struct ParseResult {
    cnf_formulas: Vec<Clause>,
    fof_formulas: Vec<NamedFormula>,
}

/// Parsing context holding the interner
struct ParseContext {
    interner: RefCell<Interner>,
}

impl ParseContext {
    fn new() -> Self {
        ParseContext {
            interner: RefCell::new(Interner::new()),
        }
    }

    fn intern_variable(&self, name: &str) -> Variable {
        Variable::new(self.interner.borrow_mut().intern_variable(name))
    }

    fn intern_constant(&self, name: &str) -> Constant {
        Constant::new(self.interner.borrow_mut().intern_constant(name))
    }

    fn intern_function(&self, name: &str, arity: usize) -> FunctionSymbol {
        FunctionSymbol::new(
            self.interner.borrow_mut().intern_function(name),
            arity as u8,
        )
    }

    fn intern_predicate(&self, name: &str, arity: usize) -> PredicateSymbol {
        PredicateSymbol::new(
            self.interner.borrow_mut().intern_predicate(name),
            arity as u8,
        )
    }

    fn into_interner(self) -> Interner {
        self.interner.into_inner()
    }
}

// Thread-local parsing context for use during parsing
thread_local! {
    static PARSE_CTX: RefCell<Option<ParseContext>> = const { RefCell::new(None) };
}

fn with_ctx<F, R>(f: F) -> R
where
    F: FnOnce(&ParseContext) -> R,
{
    PARSE_CTX.with(|ctx| {
        let ctx_ref = ctx.borrow();
        f(ctx_ref.as_ref().expect("ParseContext not initialized"))
    })
}

/// Parse a TPTP file and return the CNF formula with interner
///
/// Args:
///   - file_path: Path to the TPTP file
///   - include_dirs: Directories to search for included files
///   - timeout: Optional timeout instant for CNF conversion
pub fn parse_tptp_file(
    file_path: &str,
    include_dirs: &[&str],
    timeout: Option<Instant>,
) -> Result<ParsedProblem, String> {
    // Initialize parsing context
    PARSE_CTX.with(|ctx| {
        *ctx.borrow_mut() = Some(ParseContext::new());
    });

    let mut visited = HashSet::new();
    let result = parse_file_recursive(file_path, include_dirs, &mut visited, timeout)?;
    let formula = convert_to_cnf(result, timeout)?;

    // Extract interner from context
    let interner = PARSE_CTX.with(|ctx| ctx.borrow_mut().take().unwrap().into_interner());

    Ok(ParsedProblem { formula, interner })
}

/// Parse a TPTP string and return the CNF formula with interner
///
/// Args:
///   - input: TPTP content as string
///   - include_dirs: Directories to search for included files
///   - timeout: Optional timeout instant for CNF conversion
pub fn parse_tptp(
    input: &str,
    include_dirs: &[&str],
    timeout: Option<Instant>,
) -> Result<ParsedProblem, String> {
    // Initialize parsing context
    PARSE_CTX.with(|ctx| {
        *ctx.borrow_mut() = Some(ParseContext::new());
    });

    let mut visited = HashSet::new();
    let result = parse_content(input, include_dirs, &PathBuf::from("."), &mut visited, timeout)?;
    let formula = convert_to_cnf(result, timeout)?;

    // Extract interner from context
    let interner = PARSE_CTX.with(|ctx| ctx.borrow_mut().take().unwrap().into_interner());

    Ok(ParsedProblem { formula, interner })
}

/// Convert parsed FOF formulas to CNF
fn convert_to_cnf(result: ParseResult, timeout: Option<Instant>) -> Result<CNFFormula, String> {
    let mut all_clauses = result.cnf_formulas;

    // Separate conjectures from other formulas
    let mut conjectures = Vec::new();
    let mut other_formulas = Vec::new();

    for named_fof in result.fof_formulas {
        match named_fof.role {
            FormulaRole::Conjecture => conjectures.push(named_fof.formula),
            _ => other_formulas.push((named_fof.formula, named_fof.role)),
        }
    }

    // Convert non-conjecture formulas to CNF with their roles
    for (formula, role) in other_formulas {
        let clause_role = formula_role_to_clause_role(&role);
        let cnf = PARSE_CTX.with(|ctx| {
            let mut ctx_ref = ctx.borrow_mut();
            let parse_ctx = ctx_ref.as_mut().unwrap();
            fof_to_cnf_with_role(formula, clause_role, timeout, parse_ctx.interner.get_mut())
        }).map_err(|e| e.to_string())?;
        all_clauses.extend(cnf.clauses);
    }

    // Handle conjectures: form the negation of their conjunction
    if !conjectures.is_empty() {
        let conjecture_formula = if conjectures.len() == 1 {
            FOFFormula::Not(Box::new(conjectures.into_iter().next().unwrap()))
        } else {
            // Multiple conjectures: ¬(C₁ ∧ C₂ ∧ ...) = ¬C₁ ∨ ¬C₂ ∨ ...
            let negated_conjectures: Vec<FOFFormula> = conjectures
                .into_iter()
                .map(|c| FOFFormula::Not(Box::new(c)))
                .collect();

            let mut disjunction = negated_conjectures.into_iter().rev();
            let mut result = disjunction.next().unwrap();
            for negated in disjunction {
                result = FOFFormula::Or(Box::new(negated), Box::new(result));
            }
            result
        };

        let cnf = PARSE_CTX.with(|ctx| {
            let mut ctx_ref = ctx.borrow_mut();
            let parse_ctx = ctx_ref.as_mut().unwrap();
            fof_to_cnf_with_role(conjecture_formula, ClauseRole::NegatedConjecture, timeout, parse_ctx.interner.get_mut())
        }).map_err(|e| e.to_string())?;
        all_clauses.extend(cnf.clauses);
    }

    // Get interner reference for orient_all_equalities
    PARSE_CTX.with(|ctx| {
        let ctx_ref = ctx.borrow();
        let parse_ctx = ctx_ref.as_ref().unwrap();
        orient_all_equalities(&mut all_clauses, parse_ctx.interner.borrow().deref());
    });

    Ok(CNFFormula {
        clauses: all_clauses,
    })
}

/// Convert a FormulaRole to a ClauseRole
fn formula_role_to_clause_role(role: &FormulaRole) -> ClauseRole {
    match role {
        FormulaRole::Axiom
        | FormulaRole::Lemma
        | FormulaRole::Theorem
        | FormulaRole::Corollary
        | FormulaRole::Assumption => ClauseRole::Axiom,
        FormulaRole::Hypothesis => ClauseRole::Hypothesis,
        FormulaRole::Definition => ClauseRole::Definition,
        FormulaRole::Conjecture | FormulaRole::NegatedConjecture => ClauseRole::NegatedConjecture,
    }
}

fn parse_file_recursive(
    file_path: &str,
    include_dirs: &[&str],
    visited: &mut HashSet<PathBuf>,
    timeout: Option<Instant>,
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
    parse_content(&content, include_dirs, parent_dir, visited, timeout)
}

fn parse_content(
    input: &str,
    include_dirs: &[&str],
    current_dir: &Path,
    visited: &mut HashSet<PathBuf>,
    timeout: Option<Instant>,
) -> Result<ParseResult, String> {
    let mut cnf_formulas = Vec::new();
    let mut fof_formulas = Vec::new();

    // Split input into logical statements (ending with '.')
    let mut current_statement = String::new();

    for line in input.lines() {
        // Check timeout before processing each line
        if let Some(deadline) = timeout {
            if Instant::now() > deadline {
                return Err("Parsing timed out".to_string());
            }
        }

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
                let included =
                    parse_include_directive(statement, include_dirs, current_dir, visited, timeout)?;
                cnf_formulas.extend(included.cnf_formulas);
                fof_formulas.extend(included.fof_formulas);
            }
            // Parse CNF formula
            else if statement.starts_with("cnf(") {
                match parse_cnf_line(statement) {
                    Ok((_, clause)) => cnf_formulas.push(clause),
                    Err(e) => {
                        return Err(format!(
                            "Parse error in CNF: {:?}\nStatement: {}",
                            e, statement
                        ))
                    }
                }
            }
            // Parse FOF formula
            else if statement.starts_with("fof(") {
                // Check timeout before parsing potentially huge formulas
                if let Some(deadline) = timeout {
                    if Instant::now() > deadline {
                        return Err("Parsing timed out".to_string());
                    }
                }

                // Skip extremely large statements (> 100KB) as they will timeout anyway
                if statement.len() > 100_000 {
                    return Err(format!(
                        "Formula too large to parse ({} bytes, max 100KB)",
                        statement.len()
                    ));
                }

                match parse_fof_line(statement) {
                    Ok((_, named_formula)) => fof_formulas.push(named_formula),
                    Err(e) => {
                        return Err(format!(
                            "Parse error in FOF: {:?}\nStatement: {}",
                            e, statement
                        ))
                    }
                }
            }

            current_statement.clear();
        }
    }

    Ok(ParseResult {
        cnf_formulas,
        fof_formulas,
    })
}

fn parse_include_directive(
    line: &str,
    include_dirs: &[&str],
    current_dir: &Path,
    visited: &mut HashSet<PathBuf>,
    timeout: Option<Instant>,
) -> Result<ParseResult, String> {
    // Parse include('filename').
    let (_, filename) =
        parse_include(line).map_err(|e| format!("Failed to parse include directive: {:?}", e))?;

    // Try to find the file
    let file_path = find_include_file(filename, include_dirs, current_dir)?;

    // Recursively parse the included file
    parse_file_recursive(&file_path.to_string_lossy(), include_dirs, visited, timeout)
}

fn find_include_file(
    filename: &str,
    include_dirs: &[&str],
    current_dir: &Path,
) -> Result<PathBuf, String> {
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
    let (input, filename) = delimited(char('\''), take_until("'"), char('\''))(input)?;
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
    let (input, role_str) = parse_role(input)?; // formula role
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, mut clause) = parse_clause(input)?;
    let (input, _) = multispace0(input)?;

    // Parse optional annotations
    let (input, _annotations) = parse_annotations(input)?;

    let (input, _) = char(')')(input)?;
    let (input, _) = char('.')(input)?;

    // Set the role on the clause
    clause.role = ClauseRole::from_tptp_role(role_str);

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

    // Parse optional annotations
    let (input, _annotations) = parse_annotations(input)?;

    let (input, _) = char(')')(input)?;
    let (input, _) = char('.')(input)?;

    Ok((
        input,
        NamedFormula {
            name: name.to_string(),
            role,
            formula,
        },
    ))
}

/// Parse FOF formula
fn parse_fof_formula(input: &str) -> IResult<&str, FOFFormula> {
    parse_fof_binary(input)
}

/// Parse binary formula (handles precedence)
fn parse_fof_binary(input: &str) -> IResult<&str, FOFFormula> {
    let (input, left) = parse_fof_unary(input)?;
    let (input, _) = multispace0(input)?;

    // Try to parse binary operator (order matters for precedence)
    if let Ok((input, _)) = tag::<_, _, nom::error::Error<_>>("<=>")(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::Iff(Box::new(left), Box::new(right))))
    } else if let Ok((input, _)) = tag::<_, _, nom::error::Error<_>>("<~>")(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::Xor(Box::new(left), Box::new(right))))
    } else if let Ok((input, _)) = tag::<_, _, nom::error::Error<_>>("=>")(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::Implies(Box::new(left), Box::new(right))))
    } else if let Ok((input, _)) = tag::<_, _, nom::error::Error<_>>("<=")(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        // Reverse implication: P <= Q is Q => P
        Ok((input, FOFFormula::Implies(Box::new(right), Box::new(left))))
    } else if let Ok((input, _)) = tag::<_, _, nom::error::Error<_>>("~|")(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::Nand(Box::new(left), Box::new(right))))
    } else if let Ok((input, _)) = tag::<_, _, nom::error::Error<_>>("~&")(input) {
        let (input, _) = multispace0(input)?;
        let (input, right) = parse_fof_binary(input)?;
        Ok((input, FOFFormula::Nor(Box::new(left), Box::new(right))))
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

/// Parse $true as a special atom
fn parse_fof_true(input: &str) -> IResult<&str, FOFFormula> {
    let (input, _) = tag("$true")(input)?;
    let pred = with_ctx(|ctx| ctx.intern_predicate("$true", 0));
    Ok((
        input,
        FOFFormula::Atom(Atom {
            predicate: pred,
            args: vec![],
        }),
    ))
}

/// Parse $false as a special atom
fn parse_fof_false(input: &str) -> IResult<&str, FOFFormula> {
    let (input, _) = tag("$false")(input)?;
    let pred = with_ctx(|ctx| ctx.intern_predicate("$false", 0));
    Ok((
        input,
        FOFFormula::Atom(Atom {
            predicate: pred,
            args: vec![],
        }),
    ))
}

/// Parse unary formula
fn parse_fof_unary(input: &str) -> IResult<&str, FOFFormula> {
    alt((
        // Negation
        map(
            preceded(tuple((char('~'), multispace0)), parse_fof_unary),
            |f| FOFFormula::Not(Box::new(f)),
        ),
        // $true and $false (before other atoms)
        parse_fof_true,
        parse_fof_false,
        // Infix inequality (try before atomic to catch != operator)
        parse_fof_infix_unary,
        // Quantified formula
        parse_fof_quantified,
        // Parenthesized formula
        delimited(
            tuple((char('('), multispace0)),
            parse_fof_formula,
            tuple((multispace0, char(')'))),
        ),
        // Atomic formula
        map(parse_atom, FOFFormula::Atom),
    ))(input)
}

/// Parse FOF infix unary (inequality)
fn parse_fof_infix_unary(input: &str) -> IResult<&str, FOFFormula> {
    let (input, left) = parse_term(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("!=")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, right) = parse_term(input)?;

    // Create negated equality
    let eq_pred = with_ctx(|ctx| ctx.intern_predicate("=", 2));
    let eq_atom = Atom {
        predicate: eq_pred,
        args: vec![left, right],
    };

    Ok((input, FOFFormula::Not(Box::new(FOFFormula::Atom(eq_atom)))))
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
        parse_variable_name,
    )(input)?;
    let (input, _) = char(']')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, formula) = parse_fof_unary(input)?;

    // Build nested quantifiers for multiple variables
    let mut result = formula;
    for var_name in vars.into_iter().rev() {
        let var = with_ctx(|ctx| ctx.intern_variable(var_name));
        result = FOFFormula::Quantified(quantifier.clone(), var, Box::new(result));
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
        value(FormulaRole::Corollary, tag("corollary")),
        value(FormulaRole::Conjecture, tag("conjecture")),
        value(FormulaRole::NegatedConjecture, tag("negated_conjecture")),
    ))(input)
}

/// Parse a name (atomic_word or integer)
fn parse_name(input: &str) -> IResult<&str, &str> {
    alt((parse_single_quoted, parse_integer_name, parse_lower_word))(input)
}

/// Parse a single-quoted name
fn parse_single_quoted(input: &str) -> IResult<&str, &str> {
    if !input.starts_with('\'') {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let mut pos = 1;
    let chars: Vec<char> = input.chars().collect();

    while pos < chars.len() {
        if chars[pos] == '\\' && pos + 1 < chars.len() {
            pos += 2;
        } else if chars[pos] == '\'' {
            if pos + 1 < chars.len() && chars[pos + 1] == '\'' {
                pos += 2;
            } else {
                let byte_pos: usize = chars[..=pos].iter().map(|c| c.len_utf8()).sum();
                return Ok((&input[byte_pos..], &input[0..byte_pos]));
            }
        } else {
            pos += 1;
        }
    }

    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Tag,
    )))
}

/// Parse an integer as a name
fn parse_integer_name(input: &str) -> IResult<&str, &str> {
    let (remaining, digits) = take_while1(|c: char| c.is_numeric())(input)?;

    if let Some(c) = remaining.chars().next() {
        if c.is_alphabetic() || c == '_' {
            return Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Digit,
            )));
        }
    }

    Ok((remaining, digits))
}

/// Parse a double-quoted string (distinct object)
fn parse_double_quoted(input: &str) -> IResult<&str, &str> {
    if !input.starts_with('"') {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let mut pos = 1;
    let chars: Vec<char> = input.chars().collect();

    while pos < chars.len() {
        if chars[pos] == '\\' && pos + 1 < chars.len() {
            pos += 2;
        } else if chars[pos] == '"' {
            let byte_pos: usize = chars[..=pos].iter().map(|c| c.len_utf8()).sum();
            return Ok((&input[byte_pos..], &input[0..byte_pos]));
        } else {
            pos += 1;
        }
    }

    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Tag,
    )))
}

/// Strip double quotes and unescape content
fn strip_double_quotes(s: &str) -> String {
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        let inner = &s[1..s.len() - 1];
        let mut result = String::with_capacity(inner.len());
        let mut chars = inner.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(&next) = chars.peek() {
                    if next == '"' || next == '\\' {
                        result.push(chars.next().unwrap());
                        continue;
                    }
                }
                result.push(c);
            } else {
                result.push(c);
            }
        }
        result
    } else {
        s.to_string()
    }
}

/// Parse a lower word (starts with lowercase)
fn parse_lower_word(input: &str) -> IResult<&str, &str> {
    let mut chars = input.chars();
    if let Some(first) = chars.next() {
        if first.is_lowercase() {
            let mut end = first.len_utf8();
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
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Alpha,
    )))
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
        tag("corollary"),
        tag("conjecture"),
        tag("negated_conjecture"),
    ))(input)
}

/// Parse a clause (disjunction of literals)
fn parse_clause(input: &str) -> IResult<&str, Clause> {
    parse_cnf_formula(input)
}

/// Parse CNF formula (supports arbitrary parenthesization)
fn parse_cnf_formula(input: &str) -> IResult<&str, Clause> {
    alt((
        delimited(
            tuple((char('('), multispace0)),
            parse_cnf_formula,
            tuple((multispace0, char(')'))),
        ),
        parse_cnf_disjunction,
    ))(input)
}

/// Get the name of a distinct object (without the '"' prefix)
fn distinct_object_name(term: &Term, interner: &Interner) -> Option<String> {
    match term {
        Term::Constant(c) => {
            let name = interner.resolve_constant(c.id);
            if name.starts_with('"') {
                Some(name[1..].to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if a literal is trivially true or false due to distinct object semantics
fn eval_distinct_object_equality(lit: &Literal, interner: &Interner) -> Option<bool> {
    if !lit.is_equality(interner) || lit.args.len() != 2 {
        return None;
    }

    let left = &lit.args[0];
    let right = &lit.args[1];

    if let (Some(left_name), Some(right_name)) =
        (distinct_object_name(left, interner), distinct_object_name(right, interner))
    {
        let same = left_name == right_name;
        if lit.polarity {
            Some(same)
        } else {
            Some(!same)
        }
    } else {
        None
    }
}

/// Parse CNF disjunction
fn parse_cnf_disjunction(input: &str) -> IResult<&str, Clause> {
    separated_list1(tuple((multispace0, char('|'), multispace0)), parse_literal)(input).map(
        |(remaining, literals)| {
            // Get interner reference
            let (is_tautology, filtered) = with_ctx(|ctx| {
                let interner = ctx.interner.borrow();

                let is_tautology = literals.iter().any(|lit| {
                    if lit.polarity {
                        let pred_name = interner.resolve_predicate(lit.predicate.id);
                        if pred_name == "$true" {
                            return true;
                        }
                    }
                    if let Some(true) = eval_distinct_object_equality(lit, &interner) {
                        return true;
                    }
                    false
                });

                let filtered: Vec<Literal> = literals
                    .into_iter()
                    .filter(|lit| {
                        if lit.polarity {
                            let pred_name = interner.resolve_predicate(lit.predicate.id);
                            if pred_name == "$false" {
                                return false;
                            }
                        }
                        if let Some(false) = eval_distinct_object_equality(lit, &interner) {
                            return false;
                        }
                        true
                    })
                    .collect();

                (is_tautology, filtered)
            });

            if is_tautology {
                let pred = with_ctx(|ctx| ctx.intern_predicate("$true", 0));
                return (
                    remaining,
                    Clause::new(vec![Literal::positive(pred, vec![])]),
                );
            }

            (remaining, Clause::new(filtered))
        },
    )
}

/// Parse $true as a literal
fn parse_cnf_true(input: &str) -> IResult<&str, Literal> {
    let (input, _) = tag("$true")(input)?;
    let pred = with_ctx(|ctx| ctx.intern_predicate("$true", 0));
    Ok((
        input,
        Literal::positive(pred, vec![]),
    ))
}

/// Parse $false as a literal
fn parse_cnf_false(input: &str) -> IResult<&str, Literal> {
    let (input, _) = tag("$false")(input)?;
    let pred = with_ctx(|ctx| ctx.intern_predicate("$false", 0));
    Ok((
        input,
        Literal::positive(pred, vec![]),
    ))
}

/// Parse a literal
fn parse_literal(input: &str) -> IResult<&str, Literal> {
    let (input, _) = multispace0(input)?;
    alt((
        parse_cnf_true,
        parse_cnf_false,
        map(
            preceded(tuple((char('~'), multispace0)), parse_cnf_true),
            |lit| lit.complement(),
        ),
        map(
            preceded(tuple((char('~'), multispace0)), parse_cnf_false),
            |lit| lit.complement(),
        ),
        map(preceded(tuple((char('~'), multispace0)), parse_atom), |atom| {
            Literal::from_atom(atom, false)
        }),
        parse_negative_equality,
        map(parse_atom, |atom| Literal::from_atom(atom, true)),
    ))(input)
}

/// Parse a negative equality (term != term)
fn parse_negative_equality(input: &str) -> IResult<&str, Literal> {
    let (input, left) = parse_term(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("!=")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, right) = parse_term(input)?;

    let eq_pred = with_ctx(|ctx| ctx.intern_predicate("=", 2));

    Ok((
        input,
        Literal::negative(eq_pred, vec![left, right]),
    ))
}

/// Parse an atom
fn parse_atom(input: &str) -> IResult<&str, Atom> {
    alt((parse_equality, parse_predicate))(input)
}

/// Parse an equality atom
fn parse_equality(input: &str) -> IResult<&str, Atom> {
    let (input, left) = parse_term(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("=")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, right) = parse_term(input)?;

    let eq_pred = with_ctx(|ctx| ctx.intern_predicate("=", 2));

    Ok((
        input,
        Atom {
            predicate: eq_pred,
            args: vec![left, right],
        },
    ))
}

/// Parse a predicate
fn parse_predicate(input: &str) -> IResult<&str, Atom> {
    let (input, name) = parse_identifier(input)?;
    let name = strip_quotes(name);

    if let Ok((input, _)) = char::<&str, nom::error::Error<&str>>('(')(input) {
        let (input, args) =
            separated_list0(tuple((multispace0, char(','), multispace0)), parse_term)(input)?;
        let (input, _) = char(')')(input)?;

        let pred = with_ctx(|ctx| ctx.intern_predicate(&name, args.len()));

        Ok((
            input,
            Atom {
                predicate: pred,
                args,
            },
        ))
    } else {
        let pred = with_ctx(|ctx| ctx.intern_predicate(&name, 0));
        Ok((
            input,
            Atom {
                predicate: pred,
                args: vec![],
            },
        ))
    }
}

/// Parse a term
fn parse_term(input: &str) -> IResult<&str, Term> {
    alt((
        parse_function_term,
        parse_variable_term,
        parse_distinct_object_term,
        parse_constant_term,
    ))(input)
}

/// Parse a distinct object term
fn parse_distinct_object_term(input: &str) -> IResult<&str, Term> {
    let (input, quoted) = parse_double_quoted(input)?;
    let inner = strip_double_quotes(quoted);
    let name = format!("\"{}", inner);
    let constant = with_ctx(|ctx| ctx.intern_constant(&name));
    Ok((input, Term::Constant(constant)))
}

/// Parse a function term
fn parse_function_term(input: &str) -> IResult<&str, Term> {
    let (input, name) = parse_functor_name(input)?;
    let (input, _) = char('(')(input)?;
    let (input, args) =
        separated_list1(tuple((multispace0, char(','), multispace0)), parse_term)(input)?;
    let (input, _) = char(')')(input)?;

    let func = with_ctx(|ctx| ctx.intern_function(&name, args.len()));

    Ok((input, Term::Function(func, args)))
}

/// Parse a variable term
fn parse_variable_term(input: &str) -> IResult<&str, Term> {
    let (input, name) = parse_uppercase_ident(input)?;
    let var = with_ctx(|ctx| ctx.intern_variable(name));
    Ok((input, Term::Variable(var)))
}

/// Parse a constant term
fn parse_constant_term(input: &str) -> IResult<&str, Term> {
    let (input, name) = parse_functor_name(input)?;
    let constant = with_ctx(|ctx| ctx.intern_constant(&name));
    Ok((input, Term::Constant(constant)))
}

/// Parse an identifier
fn parse_identifier(input: &str) -> IResult<&str, &str> {
    alt((
        parse_single_quoted,
        take_while1(|c: char| c.is_alphanumeric() || c == '_'),
    ))(input)
}

/// Parse a lowercase identifier
fn parse_lowercase_ident(input: &str) -> IResult<&str, &str> {
    use nom::bytes::complete::take_while;
    use nom::character::complete::satisfy;
    use nom::combinator::recognize;
    use nom::sequence::pair;

    recognize(pair(
        satisfy(|c| c.is_lowercase()),
        take_while(|c: char| c.is_alphanumeric() || c == '_'),
    ))(input)
}

/// Strip quotes from a single-quoted string
fn strip_quotes(s: &str) -> String {
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        let inner = &s[1..s.len() - 1];
        let mut result = String::with_capacity(inner.len());
        let mut chars = inner.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(&next) = chars.peek() {
                    if next == '\'' || next == '\\' {
                        result.push(chars.next().unwrap());
                        continue;
                    }
                }
                result.push(c);
            } else if c == '\'' {
                if chars.peek() == Some(&'\'') {
                    chars.next();
                }
                result.push('\'');
            } else {
                result.push(c);
            }
        }
        result
    } else {
        s.to_string()
    }
}

/// Parse a functor name
fn parse_functor_name(input: &str) -> IResult<&str, String> {
    alt((
        map(parse_single_quoted, |s| strip_quotes(s)),
        map(parse_lowercase_ident, |s| s.to_string()),
    ))(input)
}

/// Parse an uppercase identifier
fn parse_uppercase_ident(input: &str) -> IResult<&str, &str> {
    let mut chars = input.chars();
    if let Some(first) = chars.next() {
        if first.is_uppercase() {
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
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::TakeWhile1,
    )))
}

/// Parse annotations
fn parse_annotations(input: &str) -> IResult<&str, ()> {
    if let Ok((input, _)) = char::<_, nom::error::Error<_>>(',')(input) {
        let (input, _) = multispace0(input)?;
        let (input, _) = alt((
            map(parse_general_term, |_| ()),
            map(parse_general_list, |_| ()),
        ))(input)?;
        let (input, _) = parse_optional_info(input)?;
        Ok((input, ()))
    } else {
        Ok((input, ()))
    }
}

/// Parse optional info
fn parse_optional_info(input: &str) -> IResult<&str, ()> {
    if let Ok((input, _)) =
        tuple::<_, _, nom::error::Error<_>, _>((multispace0, char(','), multispace0))(input)
    {
        let (input, _) = alt((
            map(parse_general_list, |_| ()),
            map(parse_general_term, |_| ()),
        ))(input)?;
        Ok((input, ()))
    } else {
        Ok((input, ()))
    }
}

/// Parse a general term
fn parse_general_term(input: &str) -> IResult<&str, ()> {
    alt((
        map(
            tuple((parse_general_data, char('('), parse_general_args, char(')'))),
            |_| (),
        ),
        map(parse_general_data, |_| ()),
    ))(input)
}

/// Parse general data
fn parse_general_data(input: &str) -> IResult<&str, &str> {
    alt((parse_single_quoted, parse_identifier))(input)
}

/// Parse general arguments
fn parse_general_args(input: &str) -> IResult<&str, ()> {
    let mut depth = 0;
    let mut pos = 0;
    let chars: Vec<char> = input.chars().collect();

    while pos < chars.len() {
        match chars[pos] {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            '\'' => {
                pos += 1;
                while pos < chars.len() && chars[pos] != '\'' {
                    if pos + 1 < chars.len() && chars[pos] == '\'' && chars[pos + 1] == '\'' {
                        pos += 2;
                    } else {
                        pos += 1;
                    }
                }
            }
            _ => {}
        }
        pos += 1;
    }

    Ok((&input[pos..], ()))
}

/// Parse a general list
fn parse_general_list(input: &str) -> IResult<&str, ()> {
    let (input, _) = char('[')(input)?;

    let mut depth = 1;
    let mut pos = 0;
    let chars: Vec<char> = input.chars().collect();

    while pos < chars.len() && depth > 0 {
        match chars[pos] {
            '[' => depth += 1,
            ']' => depth -= 1,
            '\'' => {
                pos += 1;
                while pos < chars.len() && chars[pos] != '\'' {
                    if pos + 1 < chars.len() && chars[pos] == '\'' && chars[pos + 1] == '\'' {
                        pos += 2;
                    } else {
                        pos += 1;
                    }
                }
            }
            _ => {}
        }
        pos += 1;
    }

    Ok((&input[pos..], ()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_clause() {
        let result = parse_tptp("cnf(test, axiom, p(a)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses.len(), 1);
        assert_eq!(result.formula.clauses[0].literals.len(), 1);
    }

    #[test]
    fn test_parse_equality() {
        let result = parse_tptp("cnf(test, axiom, X = f(a)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses.len(), 1);
        assert_eq!(result.formula.clauses[0].literals.len(), 1);
        assert!(result.formula.clauses[0].literals[0]
            .is_equality(&result.interner));
    }

    #[test]
    fn test_parse_negation() {
        let result = parse_tptp("cnf(test, axiom, ~p(X) | q(X)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses.len(), 1);
        assert_eq!(result.formula.clauses[0].literals.len(), 2);
        assert!(!result.formula.clauses[0].literals[0].polarity);
        assert!(result.formula.clauses[0].literals[1].polarity);
    }

    #[test]
    fn test_parse_fof_conjunction() {
        let result = parse_tptp("fof(test, axiom, p(a) & q(b)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses.len(), 2);
    }

    #[test]
    fn test_parse_fof_quantified() {
        let result = parse_tptp("fof(test, axiom, ![X]: p(X)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses.len(), 1);
    }

    #[test]
    fn test_parse_include() {
        let input = "include('test.ax').";
        let (remaining, filename) = parse_include(input).unwrap();
        assert_eq!(filename, "test.ax");
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_clause_role_parsing() {
        use crate::logic::ClauseRole;

        let result = parse_tptp("cnf(c1, axiom, p(a)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses[0].role, ClauseRole::Axiom);

        let result = parse_tptp("cnf(c2, negated_conjecture, ~p(a)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses[0].role, ClauseRole::NegatedConjecture);

        let result = parse_tptp("cnf(c3, hypothesis, q(X)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses[0].role, ClauseRole::Hypothesis);
    }

    #[test]
    fn test_fof_role_propagation() {
        use crate::logic::ClauseRole;

        let result = parse_tptp("fof(ax1, axiom, p(a) & q(b)).", &[], None).unwrap();
        for clause in &result.formula.clauses {
            assert_eq!(clause.role, ClauseRole::Axiom);
        }

        let result = parse_tptp("fof(conj, conjecture, p(a)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses[0].role, ClauseRole::NegatedConjecture);

        let result = parse_tptp("fof(hyp1, hypothesis, r(X)).", &[], None).unwrap();
        assert_eq!(result.formula.clauses[0].role, ClauseRole::Hypothesis);
    }

    #[test]
    fn test_interner_populated() {
        let result = parse_tptp("cnf(test, axiom, p(X, f(a))).", &[], None).unwrap();

        // Check that interner has correct symbols
        assert!(result.interner.get_variable("X").is_some());
        assert!(result.interner.get_constant("a").is_some());
        assert!(result.interner.get_function("f").is_some());
        assert!(result.interner.get_predicate("p").is_some());
    }
}
