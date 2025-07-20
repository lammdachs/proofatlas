# TPTP FOF and CNF Grammar Rules

This document extracts the grammar rules for First-Order Form (FOF) and Conjunctive Normal Form (CNF) from the TPTP v9.0.0 BNF specification.

## Notation

- `::=` regular grammar rules for syntactic parsing
- `:==` semantic grammar rules (specific values that make semantic sense)
- `::−` rules that produce tokens
- `:::` rules that define character classes

## Top-Level Formula Structure

```bnf
<fof_annotated>        ::= fof(<name>,<formula_role>,<fof_formula><annotations>).
<cnf_annotated>        ::= cnf(<name>,<formula_role>,<cnf_formula><annotations>).
<annotations>          ::= ,<source><optional_info> | <nothing>
```

## Formula Roles

```bnf
<formula_role>         ::= <lower_word>
<formula_role>         :== axiom | hypothesis | definition | assumption | lemma | theorem |
                           corollary | conjecture | negated_conjecture
```

Note: The roles `plain`, `type`, `interpretation`, `fi_domain`, `fi_functors`, `fi_predicates`, and `unknown` are explicitly excluded from this implementation.

## FOF Grammar Rules

### FOF Formula Structure

```bnf
<fof_formula>          ::= <fof_logic_formula>
<fof_logic_formula>    ::= <fof_binary_formula> | <fof_unary_formula> | <fof_unitary_formula>
```

### FOF Binary Formulas

```bnf
<fof_binary_formula>   ::= <fof_binary_nonassoc> | <fof_binary_assoc>
<fof_binary_nonassoc>  ::= <fof_unit_formula> <nonassoc_connective> <fof_unit_formula>
<fof_binary_assoc>     ::= <fof_or_formula> | <fof_and_formula>
<fof_or_formula>       ::= <fof_unit_formula> <vline> <fof_unit_formula> |
                           <fof_or_formula> <vline> <fof_unit_formula>
<fof_and_formula>      ::= <fof_unit_formula> & <fof_unit_formula> |
                           <fof_and_formula> & <fof_unit_formula>
```

### FOF Unary Formulas

```bnf
<fof_unary_formula>    ::= <unary_connective> <fof_unit_formula> | <fof_infix_unary>
<fof_infix_unary>      ::= <fof_term> <infix_inequality> <fof_term>
<fof_unit_formula>     ::= <fof_unitary_formula> | <fof_unary_formula>
```

### FOF Unitary Formulas

```bnf
<fof_unitary_formula>  ::= <fof_quantified_formula> | <fof_atomic_formula> | (<fof_logic_formula>)
<fof_quantified_formula> ::= <fof_quantifier> [<fof_variable_list>] : <fof_unit_formula>
<fof_variable_list>    ::= <variable> | <variable>,<fof_variable_list>
```

### FOF Atomic Formulas

```bnf
<fof_atomic_formula>   ::= <fof_plain_atomic_formula> | <fof_defined_atomic_formula>
<fof_plain_atomic_formula> ::= <fof_plain_term>
<fof_plain_atomic_formula> :== <proposition> | <predicate>(<fof_arguments>)
<fof_defined_atomic_formula> ::= <fof_defined_infix_formula>
<fof_defined_infix_formula> ::= <fof_term> <defined_infix_pred> <fof_term>
```

### FOF Terms

```bnf
<fof_term>             ::= <fof_function_term> | <variable>
<fof_function_term>    ::= <fof_plain_term>
<fof_plain_term>       ::= <constant> | <functor>(<fof_arguments>)
<fof_arguments>        ::= <fof_term> | <fof_term>,<fof_arguments>
```


## CNF Grammar Rules

### CNF Formula Structure

```bnf
<cnf_formula>          ::= <cnf_disjunction> | ( <cnf_formula> )
<cnf_disjunction>      ::= <cnf_literal> | <cnf_disjunction> <vline> <cnf_literal>
<cnf_literal>          ::= <fof_atomic_formula> | ~ <fof_atomic_formula> |
                           ~ (<fof_atomic_formula>) | <fof_infix_unary>
```

Note: CNF formulas reuse FOF atomic formulas and terms.

## Connectives and Operators

### FOF Connectives

```bnf
<fof_quantifier>       ::= ! | ?
<nonassoc_connective>  ::= <=> | => | <= | <~> | ~<vline> | ~&
<assoc_connective>     ::= <vline> | &
<unary_connective>     ::= ~
<defined_infix_pred>   ::= <infix_equality>
<infix_equality>       ::= =
<infix_inequality>     ::= !=
```

### Other Operators

```bnf
<vline>                ::: [|]
```

## Atoms and Basic Elements

### Predicates and Propositions

```bnf
<proposition>          :== <predicate>
<predicate>            :== <atomic_word>
```

### Constants and Functions

```bnf
<constant>             ::= <functor>
<functor>              ::= <atomic_word>
```

### Terms and Variables

```bnf
<variable>             ::= <upper_word>
```

## Names and Identifiers

```bnf
<name>                 ::= <atomic_word> | <integer>
<atomic_word>          ::= <lower_word> | <single_quoted>
```

## Lexical Rules (Tokens)

### Words and Identifiers

```bnf
<upper_word>           ::- <upper_alpha><alpha_numeric>*
<lower_word>           ::- <lower_alpha><alpha_numeric>*
<single_quoted>        ::- <single_quote><sq_char><sq_char>*<single_quote>
```


### Character Classes

```bnf
<alpha_numeric>        ::: (<lower_alpha>|<upper_alpha>|<numeric>|<underscore>)
<lower_alpha>          ::: [a-z]
<upper_alpha>          ::: [A-Z]
<numeric>              ::: [0-9]
<underscore>           ::: [_]
<dollar>               ::: [$]
```

## Notes

1. Variables in CNF formulas are implicitly universally quantified
2. FOF formulas require all variables to be explicitly quantified
3. CNF literals can use `!=` directly, while FOF must handle it as `<fof_infix_unary>`
4. Both FOF and CNF share the same atomic formula and term structures
5. The grammar allows for system-specific and defined constants/functions/predicates