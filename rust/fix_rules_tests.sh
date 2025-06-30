#!/bin/bash
# Script to fix rules/tests.rs to use pre-allocated arrays

# Replace ArrayProblem::new() with create_test_problem()
sed -i 's/ArrayProblem::new()/create_test_problem()/g' src/rules/tests.rs

# Add .unwrap() to create_test_clause calls
sed -i 's/create_test_clause(&mut problem, /create_test_clause(\&mut problem, /g' src/rules/tests.rs
sed -i 's/create_test_clause(\([^)]*\))/create_test_clause(\1).unwrap()/g' src/rules/tests.rs

# Add .unwrap() to create_function_term calls
sed -i 's/create_function_term(&mut problem, /create_function_term(\&mut problem, /g' src/rules/tests.rs
sed -i 's/create_function_term(\([^)]*\))/create_function_term(\1).unwrap()/g' src/rules/tests.rs

# Fix double unwrap cases
sed -i 's/.unwrap().unwrap()/.unwrap()/g' src/rules/tests.rs

echo "Fixed basic patterns. Manual fixes still needed for:"
echo "1. create_equality_clause function needs to be updated to use ArrayBuilder"
echo "2. Direct push() calls in tests need manual conversion"
echo "3. NodeType comparisons need to add 'as u8'"