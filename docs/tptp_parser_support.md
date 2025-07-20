# TPTP Parser Support Documentation

This document specifies which parts of the TPTP (Thousands of Problems for Theorem Provers) format are supported by the ProofAtlas parser.

## Overview

The ProofAtlas TPTP parser supports the core FOF (First-Order Form) and CNF (Conjunctive Normal Form) dialects of TPTP, focusing on the essential features needed for theorem proving while omitting advanced or rarely-used features.

## ✅ Supported Features

### Formula Types
- **FOF (First-Order Form)** - `fof(name, role, formula).`
- **CNF (Conjunctive Normal Form)** - `cnf(name, role, clause).`

### Formula Names
- **Lower-case identifiers** - `axiom1`, `my_formula`
- **Single-quoted names** - `'complex-name!@#'`, `'name with spaces'`
- **Integer names** - `123`, `456`

### Formula Roles
- `axiom`
- `hypothesis`
- `definition`
- `assumption`
- `lemma`
- `theorem`
- `corollary`
- `conjecture`
- `negated_conjecture`

### FOF Connectives
- **Negation** - `~` (e.g., `~p(X)`)
- **Conjunction** - `&` (e.g., `p(X) & q(Y)`)
- **Disjunction** - `|` (e.g., `p(X) | q(Y)`)
- **Implication** - `=>` (e.g., `p(X) => q(Y)`)
- **Reverse implication** - `<=` (e.g., `p(X) <= q(Y)`)
- **Biconditional** - `<=>` (e.g., `p(X) <=> q(Y)`)
- **XOR** - `<~>` (e.g., `p(X) <~> q(Y)`)
- **NAND** - `~&` (e.g., `p(X) ~& q(Y)`)
- **NOR** - `~|` (e.g., `p(X) ~| q(Y)`)

### Quantifiers
- **Universal** - `!` (e.g., `![X,Y]: p(X,Y)`)
- **Existential** - `?` (e.g., `?[X]: p(X)`)

### Terms
- **Variables** - Upper-case identifiers (e.g., `X`, `Var1`, `X_1`)
- **Constants** - Lower-case identifiers (e.g., `a`, `const1`)
- **Functions** - Lower-case identifier with arguments (e.g., `f(X,Y)`, `sum(a,b)`)

### Atoms
- **Predicates** - With or without arguments (e.g., `p`, `q(X,Y)`)
- **Equality** - `=` (e.g., `X = Y`, `f(a) = g(b)`)
- **Inequality** - `!=` in both FOF and CNF (e.g., `X != Y`)

### CNF Features
- **Disjunction of literals** - `p(X) | ~q(Y) | r(Z)`
- **Parenthesized clauses** - `(p(X) | q(Y))`
- **Arbitrary parenthesization** - `((p | q))`, `(((p)))`

### Special Features
- **Include directives** - `include('filename.ax').`
- **Annotations** - Source and inference information (parsed but ignored)
  - Simple: `introduced(definition)`
  - Complex: `inference(resolution, [status(thm)], [source1, source2])`
  - Lists: `[author('John'), date(2024)]`
- **Multiple conjectures** - Properly handled by negating their conjunction

### Lexical Features
- **Comments** - Lines starting with `%`
- **Multi-line formulas** - Formulas can span multiple lines
- **Whitespace** - Flexible whitespace handling
- **Escape sequences in quotes** - `''` for single quote in single-quoted names

## ❌ Not Supported Features

### Formula Types
- **TFF** (Typed First-order Form)
- **THF** (Typed Higher-order Form)
- **TPI** (TPTP Process Instruction)
- **FOF Sequents**

### Formula Roles
- `plain`
- `type`
- `interpretation`
- `fi_domain`
- `fi_functors`
- `fi_predicates`
- `unknown`

### Advanced Features
- **Numbers as terms** - No support for integers, rationals, or reals as term values
- **Distinct objects** - No support for `"distinct_object"` syntax
- **Dollar constants** - No `$true`, `$false`, or other dollar-prefixed constants
- **System functions/predicates** - No `$$system_words`
- **Defined functions** - No arithmetic operators like `$sum`, `$difference`, etc.
- **Defined predicates** - No `$less`, `$greater`, etc.

### Type System
- **Type declarations** - No support for typed formulas
- **Sort declarations** - No support for many-sorted logic

### Extended Syntax
- **Formula role extensions** - No `role-general_term` syntax
- **Conditional formulas** - No `$ite` (if-then-else)
- **Let expressions** - No `$let` bindings
- **Global definitions** - No `$def` declarations

### Annotations (Parsed but Ignored)
While the parser accepts annotations for compatibility, it does not:
- Store annotation data
- Use source information
- Track inference history
- Validate annotation format beyond basic syntax

## Usage Notes

1. **Conjecture Handling**: Multiple conjectures are automatically combined by negating their conjunction, following standard refutation-based proving.

2. **Equality Orientation**: All equalities are automatically oriented using Knuth-Bendix Ordering (KBO) for efficiency.

3. **Implicit Universal Quantification**: In CNF formulas, all variables are implicitly universally quantified.

4. **Include Path Resolution**: Include files are searched in:
   - The directory of the including file
   - User-specified include directories (via `--include` option)

## Examples

### Supported TPTP Input
```tptp
% A simple group theory problem
fof('left-identity', axiom, ![X]: mult(e, X) = X).
fof(left_inverse, axiom, ![X]: mult(inv(X), X) = e).
fof(associativity, axiom, ![X,Y,Z]: mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).

% Using new operators
fof(xor_example, axiom, p <~> q).
fof(nand_example, axiom, ~(p ~& q)).

% CNF with parentheses
cnf(clause1, axiom, ((p(X) | q(Y)))).

% With annotations (parsed but ignored)
fof(derived, lemma, r(a), inference(modus_ponens, [status(thm)], [1,2])).
```

### Not Supported
```tptp
% TFF syntax
tff(type_decl, type, person: $tType).

% Numbers as terms
fof(arithmetic, axiom, sum(2, 3) = 5).

% Distinct objects
fof(people, axiom, "John" != "Mary").

% Dollar constants
fof(truth, axiom, $true).

% System predicates
fof(system, axiom, $$myPred(X)).

% Unsupported roles
fof(data, plain, p(a)).
cnf(typing, type, q(b)).
```

## Compatibility

This parser implementation is designed to handle the vast majority of problems in the TPTP library that use standard FOF and CNF syntax without advanced features. Problems using typed formulas, arithmetic, or other extended features will need to be converted to standard first-order syntax before parsing.