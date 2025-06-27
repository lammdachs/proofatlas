//! Type conversion utilities between Rust and Python

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::core::logic::*;

/// Convert Rust Problem to Python dictionary matching the expected format
pub fn problem_to_python(py: Python, problem: &Problem) -> PyResult<PyObject> {
    // Import Python Problem class
    let logic_module = py.import("proofatlas.core.logic")?;
    let problem_class = logic_module.getattr("Problem")?;
    
    // Convert clauses
    let py_clauses = PyList::empty(py);
    for clause in &problem.clauses {
        let py_clause = clause_to_python(py, clause)?;
        py_clauses.append(py_clause)?;
    }
    
    // Convert conjecture indices to list
    let conjecture_list = PyList::new(py, problem.conjecture_indices.iter().cloned());
    
    // Create Problem instance
    let args = (py_clauses,);
    let kwargs = PyDict::new(py);
    kwargs.set_item("conjecture_indices", conjecture_list)?;
    
    let problem_instance = problem_class.call(args, Some(kwargs))?;
    Ok(problem_instance.into())
}

/// Convert Rust Clause to Python Clause
fn clause_to_python(py: Python, clause: &Clause) -> PyResult<PyObject> {
    let logic_module = py.import("proofatlas.core.logic")?;
    let clause_class = logic_module.getattr("Clause")?;
    
    // Convert literals
    let py_literals = PyList::empty(py);
    for literal in &clause.literals {
        let py_literal = literal_to_python(py, literal)?;
        py_literals.append(py_literal)?;
    }
    
    // Create Clause(*literals)
    let args = py_literals.iter().collect::<Vec<_>>();
    clause_class.call1(args).map(Into::into)
}

/// Convert Rust Literal to Python Literal
fn literal_to_python(py: Python, literal: &Literal) -> PyResult<PyObject> {
    let logic_module = py.import("proofatlas.core.logic")?;
    let literal_class = logic_module.getattr("Literal")?;
    
    // Convert predicate
    let py_predicate = predicate_to_python(py, &literal.predicate)?;
    
    // Create Literal(predicate, polarity)
    literal_class.call1((py_predicate, literal.polarity)).map(Into::into)
}

/// Convert Rust Predicate to Python Predicate
fn predicate_to_python(py: Python, predicate: &Predicate) -> PyResult<PyObject> {
    let logic_module = py.import("proofatlas.core.logic")?;
    let predicate_class = logic_module.getattr("Predicate")?;
    
    // Convert terms
    let py_terms = PyList::empty(py);
    for term in &predicate.args {
        let py_term = term_to_python(py, term)?;
        py_terms.append(py_term)?;
    }
    
    // Create Predicate(name, arity)(*terms)
    let predicate_symbol = predicate_class.call1((&predicate.name, predicate.arity()))?;
    
    // Call the symbol with terms to create the predicate application
    let args = py_terms.iter().collect::<Vec<_>>();
    predicate_symbol.call1(args).map(Into::into)
}

/// Convert Rust Term to Python Term
fn term_to_python(py: Python, term: &Term) -> PyResult<PyObject> {
    let logic_module = py.import("proofatlas.core.logic")?;
    
    match term {
        Term::Variable(name) => {
            let variable_class = logic_module.getattr("Variable")?;
            variable_class.call1((name,)).map(Into::into)
        }
        Term::Constant(name) => {
            let constant_class = logic_module.getattr("Constant")?;
            constant_class.call1((name,)).map(Into::into)
        }
        Term::Function { name, args } => {
            let function_class = logic_module.getattr("Function")?;
            
            // Convert argument terms
            let py_args = PyList::empty(py);
            for arg in args {
                let py_arg = term_to_python(py, arg)?;
                py_args.append(py_arg)?;
            }
            
            // Create Function(name, arity)
            let func_symbol = function_class.call1((name, args.len()))?;
            
            // Call the symbol with args to create the function application
            let arg_vec = py_args.iter().collect::<Vec<_>>();
            func_symbol.call1(arg_vec).map(Into::into)
        }
    }
}

/// Alternative: Convert to dictionary format for JSON serialization
pub fn problem_to_dict(py: Python, problem: &Problem) -> PyResult<&PyDict> {
    let dict = PyDict::new(py);
    
    dict.set_item("num_clauses", problem.clauses.len())?;
    dict.set_item("num_literals", problem.count_literals())?;
    
    // Convert clauses to list of dicts
    let clauses_list = PyList::empty(py);
    for clause in &problem.clauses {
        let clause_dict = clause_to_dict(py, clause)?;
        clauses_list.append(clause_dict)?;
    }
    dict.set_item("clauses", clauses_list)?;
    
    // Add conjecture indices
    let conjecture_list = PyList::new(py, problem.conjecture_indices.iter().cloned());
    dict.set_item("conjecture_indices", conjecture_list)?;
    
    Ok(dict)
}

fn clause_to_dict(py: Python, clause: &Clause) -> PyResult<&PyDict> {
    let dict = PyDict::new(py);
    
    let literals_list = PyList::empty(py);
    for literal in &clause.literals {
        let lit_dict = literal_to_dict(py, literal)?;
        literals_list.append(lit_dict)?;
    }
    dict.set_item("literals", literals_list)?;
    
    Ok(dict)
}

fn literal_to_dict(py: Python, literal: &Literal) -> PyResult<&PyDict> {
    let dict = PyDict::new(py);
    
    dict.set_item("polarity", literal.polarity)?;
    
    // Predicate as dict
    let pred_dict = PyDict::new(py);
    pred_dict.set_item("symbol", &literal.predicate.name)?;
    pred_dict.set_item("arity", literal.predicate.arity())?;
    dict.set_item("predicate", pred_dict)?;
    
    // Terms
    let terms_list = PyList::empty(py);
    for term in &literal.predicate.args {
        let term_dict = term_to_dict(py, term)?;
        terms_list.append(term_dict)?;
    }
    dict.set_item("terms", terms_list)?;
    
    Ok(dict)
}

fn term_to_dict(py: Python, term: &Term) -> PyResult<&PyDict> {
    let dict = PyDict::new(py);
    
    match term {
        Term::Variable(name) => {
            dict.set_item("type", "variable")?;
            dict.set_item("name", name)?;
        }
        Term::Constant(name) => {
            dict.set_item("type", "constant")?;
            dict.set_item("symbol", name)?;
        }
        Term::Function { name, args } => {
            dict.set_item("type", "function")?;
            dict.set_item("symbol", name)?;
            dict.set_item("arity", args.len())?;
            
            let args_list = PyList::empty(py);
            for arg in args {
                let arg_dict = term_to_dict(py, arg)?;
                args_list.append(arg_dict)?;
            }
            dict.set_item("arguments", args_list)?;
        }
    }
    
    Ok(dict)
}