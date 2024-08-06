from itertools import chain
from lark import Transformer, Tree, Token, Visitor
import os

from foreduce.fol.logic import *
from foreduce.tptp.lexer import tptplexer


class _Maxsize:
    def __init__(self, max_size):
        if max_size is not None and max_size < 0:
            raise ValueError("max_size reached")
        self.max_size = max_size

    def use(self, size):
        if self.max_size is not None:
            self.max_size -= size
            if self.max_size < 0:
                raise Exception("File is too large")


def read_file(file, include_path='/', max_size=None):
    _max_size = _Maxsize(max_size)
    _max_size.use(os.path.getsize(file))
    with open(file, "r") as f:
        data = f.read()
    return read_string(data, include_path, _max_size=_max_size)


def read_string(string, include_path='/', _max_size=None):
    tree = tptplexer.parse(string)
    return read_tree(tree, include_path, _max_size=_max_size)


def read_tree(tree, include_path='/', _max_size=_Maxsize(None)):
    tree = include(tree, include_path, _max_size=_max_size)
    tree = opsimp(tree)
    tree = simplify(tree)
    tree = nnf(tree)
    tree = simplify(tree)
    tree = eliminate_equivalences(tree)
    tree = nnf(tree)
    tree = skolemize(tree)
    tree = simplify(tree)
    tree = cnf(tree)
    counter = 0
    for formula in tree.children:
        renamer = VariableRenamer(counter)   
        renamer.visit(formula)
        counter = renamer.counter
    return FOLConverter().transform(tree)


def include(tree, include_path='/', _max_size=_Maxsize(None)):
    visitor = Include(include_path=include_path, _max_size=_max_size)
    visitor.visit(tree)
    return tree


class Include(Visitor):
    def __init__(self, include_path='/', _max_size=_Maxsize(None)):
        self.include_path = include_path
        self._max_size = _max_size
    
    def include(self, tree):
        self._max_size.use(os.path.getsize(self.include_path + tree.children[0].value[1:-1]))
        with open(self.include_path + tree.children[0].value[1:-1], "r") as f:
            data = f.read()
        input = tptplexer.parse(data)
        Include(include_path=self.include_path, _max_size=self._max_size).visit(input)
        tree.children = [entry for entry in input.children if entry.data == "formula"]
    
    def tptp_file(self, tree):
        formulas  = [formula for formula in tree.children if formula.data == "formula"]
        included = list(chain(*[include.children for include in tree.children if include.data == "include"]))
        tree.children = formulas + included


# Remove all connectives except for ~, |, &, <=>, <~>
def opsimp(tree):
    OpSimplify().visit(tree)
    return tree

class OpSimplify(Visitor):
    # p => q -> ~p | q
    # p <= q -> p | ~q
    # p ~| q -> ~(p | q)
    # p ~& q -> ~(p & q)
    def fof_binary(self, tree):
        match tree.children[1].value:
            case "=>":
                tree.children = [Tree("fof_unary", [Tree("fof_negation", [tree.children[0]])]), Token("BINARY_CONNECTIVE", "|"), tree.children[2]]
            case "<=":
                tree.children = [tree.children[0], Token("BINARY_CONNECTIVE", "|"), Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_formula", [tree.children[2]])])])])]
            case "~|":
                tree.children = [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [tree.children[0], Token("BINARY_CONNECTIVE", "|"), tree.children[2]])])])])]
            case "~&":
                tree.children = [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [tree.children[0], Token("BINARY_CONNECTIVE", "&"), tree.children[2]])])])])]

    def fof(self, tree):
        match tree.children[1].value:
            case "conjecture" | "theorem" | "lemma" | "corollary":
                tree.children[1] = Token("FORMULA_ROLE", "negated_conjecture")
                tree.children[2] = Tree("fof_formula", [Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [tree.children[2]])])])])
                

# Simplify occurences of $true, $false and redundant parentheses

def simplify(tree):
    tree = Simplify().visit(tree)
    return tree


class Simplify(Visitor):
    def __init__(self):
        self.unchanged = False

    def fof_negation(self, tree):
        self.unchanged = False
        if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
            tree.data = "fof_atom"
            tree.children = [Token("DEFINED_UNARY_PREDICATE", "$true")]
        elif tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
            tree.data = "fof_atom"
            tree.children = [Token("DEFINED_UNARY_PREDICATE", "$false")]

    def fof_binary(self, tree):
        match tree.children[1].value:
            case "|":
                if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]) or tree.children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]
                    return
                if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    tree.data = tree.children[2].data
                    tree.children = tree.children[2].children
                    return
                if tree.children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = tree.children[0].children
                    return
            case "&":
                if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]) or tree.children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]
                    return
                if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    tree.data = tree.children[2].data
                    tree.children = tree.children[2].children
                    return
                if tree.children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = tree.children[0].children
                    return
            case "<=>":
                if tree.children[0] == tree.children[2]:
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]
                    return
                if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    tree.data = tree.children[2].data
                    tree.children = tree.children[2].children
                    return
                if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = [Tree("fof_negation", [tree.children[2]])]
                    return
                if tree.children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = tree.children[0].children
                    return
                if tree.children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = [Tree("fof_negation", [tree.children[0]])]
                    return
            case "<~>":
                if tree.children[0] == tree.children[2]:
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]
                    return
                if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = [Tree("fof_negation", [tree.children[2]])]
                    return
                if tree.children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    tree.data = tree.children[2].data
                    tree.children = tree.children[2].children
                    return
                if tree.children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = [Tree("fof_negation", [tree.children[0]])]
                    return
                if tree.children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    tree.data = "fof_unary"
                    tree.children = tree.children[0].children
                    return
        if tree.children[2].data == "fof_unary":
            if tree.children[2].children[0].data == "fof_formula":
                if tree.children[2].children[0].children[0].data == "fof_binary":
                    self.unchanged = False
                    tree.children = [tree.children[0], tree.children[1], tree.children[2].children[0].children[0]]
                    return

    def fof_unary(self, tree):
        if tree.children[0].data == "fof_formula":
            if tree.children[0].children[0].data == "fof_unary":
                self.unchanged = False
                tree.children = tree.children[0].children[0].children


# Push negations inwards

def nnf(tree):
    visitor = NNF()
    while not visitor.unchanged:
        visitor.unchanged = True
        visitor.visit(tree)
    return tree


class NNF(Visitor):
    def __init__(self):
        self.unchanged = False
    
    def fof_unary(self, tree):
        if tree.children[0].data == "fof_negation":
            if tree.children[0].children[0].children[0].data == "fof_negation":
                self.unchanged = False
                tree.data = tree.children[0].children[0].children[0].children[0].data
                tree.children = tree.children[0].children[0].children[0].children[0].children
            elif tree.children[0].children[0].children[0].data == "fof_quantified_formula":
                self.unchanged = False
                match tree.children[0].children[0].children[0].children[0].value:
                    case "!":
                        tree.children = [Tree("fof_quantified_formula", [
                            Token("QUANTIFIER", "?"),
                            *tree.children[0].children[0].children[0].children[1:-1],
                            Tree("fof_unary", [Tree("fof_negation", [tree.children[0].children[0].children[0].children[-1]])])
                        ])]
                    case "?":
                        tree.children = [Tree("fof_quantified_formula", [
                            Token("QUANTIFIER", "!"),
                            *tree.children[0].children[0].children[0].children[1:-1],
                            Tree("fof_unary", [Tree("fof_negation", [tree.children[0].children[0].children[0].children[-1]])])
                        ])]
            elif tree.children[0].children[0].children[0].data == "fof_formula":
                f = tree.children[0].children[0].children[0].children[0]
                if f.data == "fof_binary":
                    match f.children[1].value:
                        case "|":
                            self.unchanged = False
                            tree.children = [Tree("fof_formula", [Tree("fof_binary", [Tree("fof_unary", [Tree("fof_negation", [f.children[0]])]), Token("BINARY_CONNECTIVE", "&"), Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_formula", [f.children[2]])])])])])])]
                        case "&":
                            self.unchanged = False
                            tree.children = [Tree("fof_formula", [Tree("fof_binary", [Tree("fof_unary", [Tree("fof_negation", [f.children[0]])]), Token("BINARY_CONNECTIVE", "|"), Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_formula", [f.children[2]])])])])])])]
                        case "<=>":
                            self.unchanged = False
                            tree.children = [Tree("fof_formula", [Tree("fof_binary", [f.children[0], Token("BINARY_CONNECTIVE", "<~>"), f.children[2]])])]
                        case "<~>":
                            self.unchanged = False
                            tree.children = [Tree("fof_formula", [Tree("fof_binary", [f.children[0], Token("BINARY_CONNECTIVE", "<=>"), f.children[2]])])]


class FreeVariables(Transformer):
    def __init__(self):
        super().__init__(visit_tokens=True)

    def fof_quantified_formula(self, children):
        return children[-1] - children[1].union(*children[2:-1])

    def VARIABLE(self, name):
        return {name}

    def __default__(self, data, children, meta):
        return set().union(*children)

    def __default_token__(self, token):
        return {}


## Eliminating <=> and <~> with linear blowup

def eliminate_equivalences(tree):
    polarities = EquivPolarities().transform(tree)
    visitor = EliminateEquivalence()
    visitor.visit(tree)
    for i, (args, polarity) in enumerate(zip(visitor.equivs, polarities)):
        match polarity:
            case 1:
                tree.children += _positive_equiv_axioms(i, *args)
            case -1:
                tree.children += _negative_equiv_axioms(i, *args)
            case 0:
                tree.children += _positive_equiv_axioms(i, *args)
                tree.children += _negative_equiv_axioms(i, *args)
    return tree


class EquivPolarities(Transformer):
    def fof(self, children):
        if children[1] in ["axiom" , "hypothesis" , "definition" , "assumption", "negated_conjecture"]:
            return children[2]
        else:
            return [-p for p in children[2]]

    def cnf(self, children):
        return []
    
    def fof_binary(self, children):
        if children[1] == Token("BINARY_CONNECTIVE", "<=>"):
            return [0 for _ in children[0]] + [0 for _ in children[2]] + [1]
        elif children[1] == Token("BINARY_CONNECTIVE", "<~>"):
            return [0 for _ in children[0]] + [0 for _ in children[2]] + [-1]
        return children[0] + children[2]

    def BINARY_CONNECTIVE(self, token):
        return token

    def FORMULA_ROLE(self, token):
        return token

    def __default__(self, data, children, meta):
        return list(chain(*children))

    def __default_token__(self, token):
        return []


class EliminateEquivalence(Visitor):
    def __init__(self):
        self.counter = 0
        self.unchanged = False
        self.equivs = []
    
    def fof_binary(self, tree):
        if tree.children[1] == "<=>":
            self.unchanged = False
            free_variables = FreeVariables().transform(tree)
            self.equivs.append([tree.children[0], tree.children[2], free_variables])
            tree.data = "fof_unary"
            tree.children = [Tree("fof_atom", [
                Token("FUNCTOR", f"equiv_{self.counter}"),
                *[
                    Tree("fof_term", [Token("VARIABLE", var)])
                    for var in free_variables
                ]
            ])]
            self.counter += 1
        elif tree.children[1] == "<~>":
            self.unchanged = False
            free_variables = FreeVariables().transform(tree)
            self.equivs.append([tree.children[0], tree.children[2], free_variables])
            tree.data = "fof_unary"
            tree.children = [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_atom", [
                Token("FUNCTOR", f"equiv_{self.counter}"),
                *[
                    Tree("fof_term", [Token("VARIABLE", var)])
                    for var in free_variables
                ]
            ])])])]
            self.counter += 1


def _positive_equiv_axioms(counter, left, right, variables):
    return [
        simplify(nnf(opsimp(Tree("formula", [
            Tree("fof", [
                Token("NAME", f"equivalence_{counter}_0_0"),
                Token("FORMULA_ROLE", "axiom"),
                Tree("fof_formula", [
                    Tree("fof_binary", [
                        Tree("fof_unary", [
                            Tree("fof_atom", [
                                Token("FUNCTOR", f"equiv_{counter}"),
                                *[
                                    Tree("fof_term", [Token("VARIABLE", var)])
                                    for var in variables
                                ]
                            ])
                        ]),
                        Token("BINARY_CONNECTIVE", "=>"),
                        Tree("fof_binary", [
                            left,
                            Token("BINARY_CONNECTIVE", "=>"),
                            right
                        ])
                    ])
                ])
            ])
        ])))),
        simplify(nnf(opsimp(Tree("formula", [
            Tree("fof", [
                Token("NAME", f"equivalence_{counter}_0_1"),
                Token("FORMULA_ROLE", "axiom"),
                Tree("fof_formula", [
                    Tree("fof_binary", [
                        Tree("fof_unary", [
                            Tree("fof_atom", [
                                Token("FUNCTOR", f"equiv_{counter}"),
                                *[
                                    Tree("fof_term", [Token("VARIABLE", var)])
                                    for var in variables
                                ]
                            ])
                        ]),
                        Token("BINARY_CONNECTIVE", "=>"),
                        Tree("fof_binary", [
                            left,
                            Token("BINARY_CONNECTIVE", "<="),
                            right
                        ])
                    ])
                ])
            ])
        ]))))
    ]


def _negative_equiv_axioms(counter, left, right, variables):
    return [
        simplify(nnf(opsimp(Tree("formula", [
            Tree("fof", [
                Token("NAME", f"equivalence_{counter}_1"),
                Token("FORMULA_ROLE", "axiom"),
                Tree("fof_formula", [
                    Tree("fof_binary", [
                        Tree("fof_unary", [
                                Tree("fof_atom", [
                                    Token("FUNCTOR", f"equiv_{counter}"),
                                    *[
                                        Tree("fof_term", [Token("VARIABLE", var)])
                                        for var in variables
                                    ]
                                ])
                            ]),
                        Token("BINARY_CONNECTIVE", "<="),
                        Tree("fof_binary", [
                            Tree("fof_unary", [Tree("fof_formula", [
                                Tree("fof_binary", [
                                    left,
                                    Token("BINARY_CONNECTIVE", "=>"),
                                    right
                                ])
                            ])]),
                            Token("BINARY_CONNECTIVE", "&"),
                            Tree("fof_binary", [
                                left,
                                Token("BINARY_CONNECTIVE", "<="),
                                right
                            ])
                        ]),
                    ])
                ])
            ])
        ]))))
    ]


def skolemize(tree):
    Skolemize().visit_topdown(tree)
    return tree


class Skolemize(Visitor):
    def __init__(self):
        self.skolem_count = 0
        self.var_count = 0

    def tptp_file(self, tree):
        for child in tree.children:
            child.sub = {}
            child.var = []

    def fof(self, tree):
        self.var_count = 0
        variables = FreeVariables().transform(tree)
        tree.sub = {
            var.value : [Token("VARIABLE", f"X{self.var_count + i}")]
            for i, var in enumerate(variables)
        }
        tree.var = [Token("VARIABLE", f"X{self.var_count + i}") for i in range(len(variables))]
        self.var_count += len(variables)
        Skolemize.__default__(self, tree)

    def fof_unary(self, tree):
        if tree.children[0].data == "fof_quantified_formula":
            f = tree.children[0]
            quantifier = f.children[0].value
            variables = f.children[1:-1]
            match quantifier:
                case "!":
                    tree.sub = {v : s for v, s in tree.sub.items()}
                    tree.sub.update({
                        var.value : [Token("VARIABLE", f"X{self.var_count + i}")]
                        for i, var in enumerate(variables)
                    })
                    tree.var = [v for v in tree.var] + [Token("VARIABLE", f"X{self.var_count + i}") for i in range(len(variables))]
                    self.var_count += len(variables)
                case "?":
                    tree.sub = {v : s for v, s in tree.sub.items()}
                    tree.sub.update({
                        var.value : [Token("FUNCTOR", f"sk{self.skolem_count + i}"), *[Tree("fof_term", [v]) for v in tree.var]]
                        for i, var in enumerate(variables)
                    })
                    self.skolem_count += len(variables)
            tree.children = f.children[-1].children
            Skolemize.fof_unary(self, tree)
        else:
            Skolemize.__default__(self, tree)

    def fof_term(self, tree):
        if tree.children[0].type == "VARIABLE":
            if tree.children[0].value in tree.sub:
                tree.children = tree.sub[tree.children[0].value]
        Skolemize.__default__(self, tree)

    def __default__(self, tree):
        for child in tree.children:
            if not isinstance(child, Token):
                child.sub = tree.sub
                child.var = tree.var
        delattr(tree, "sub")
        delattr(tree, "var")


def cnf(tree):
    tree = CDA().visit(tree)
    visitor = CDAReduce()
    while not visitor.unchanged:
        visitor.unchanged = True
        visitor.visit(tree)
    return tree

class CDA(Visitor):
    def fof_unary(self, tree):
        tree.data = tree.children[0].data
        tree.children = tree.children[0].children

    def fof_negation(self, tree):
        tree.data = "fof_negated_atom"

    def fof_binary(self, tree):
        if tree.children[1] == Token("BINARY_CONNECTIVE", "|"):
            match tree.children[0].data:
                case "fof_atom" | "fof_negated_atom" | "conjunction":
                    disjuncts1 = [tree.children[0]]
                case "disjunction":
                    disjuncts1 = tree.children[0].children
            match tree.children[2].data:
                case "fof_atom" | "fof_negated_atom" | "conjunction":
                    disjuncts2 = [tree.children[2]]
                case "disjunction":
                    disjuncts2 = tree.children[2].children
            tree.data = "disjunction"
            tree.children = disjuncts1 + disjuncts2
        elif tree.children[1] == Token("BINARY_CONNECTIVE", "&"):
            match tree.children[0].data:
                case "fof_atom" | "fof_negated_atom" | "disjunction":
                    conjuncts1 = [tree.children[0]]
                case "conjunction":
                    conjuncts1 = tree.children[0].children
            match tree.children[2].data:
                case "fof_atom" | "fof_negated_atom" | "disjunction":
                    conjuncts2 = [tree.children[2]]
                case "conjunction":
                    conjuncts2 = tree.children[2].children
            tree.data = "conjunction"
            tree.children = conjuncts1 + conjuncts2
    
    def fof_formula(self, tree):
        tree.data = tree.children[0].data
        tree.children = tree.children[0].children

    def fof(self, tree):
        tree.data = "cda"
        tree.dependencies = []


def negated(tree):
    match tree.data:
        case "conjunction":
            return Tree("disjunction", [negated(child) for child in tree.children])
        case "diction":
            return Tree("conjunction", [negated(child) for child in tree.children])
        case "fof_atom":
            return Tree("fof_negated_atom", [tree])
        case "fof_negated_atom":
            return tree.children[0]


class VariableCopy(Transformer):
    def __init__(self):
        super().__init__(visit_tokens=True)

    def VARIABLE(self, tree):
        return Token("VARIABLE", tree.value)


class CDAReduce(Visitor):
    def __init__(self):
        self.unchanged = False
        self.new = []

    def cda(self, tree):
        self.unchanged = False
        index = 0
        match tree.children[2].data:
            case "fof_atom" | "fof_negated_atom":
                tree.data = "cnf"
                tree.children = [tree.children[0], tree.children[1], Tree("cnf_formula", [Tree("disjunction", [Tree("literal", [tree.children[2]])])])]
            case "disjunction":
                disjuncts = [Tree("literal", [VariableCopy().transform(f)]) for f in tree.dependencies]
                for disjunct in tree.children[2].children:
                    match disjunct.data:
                        case "fof_atom" | "fof_negated_atom":
                            disjuncts.append(Tree("literal", [disjunct]))
                        case "conjunction":
                            variables = FreeVariables().transform(disjunct)
                            atom = Tree("fof_atom", [
                                Token("FUNCTOR", f"{tree.children[0].value}_{index}"),
                                *[
                                    Tree("fof_term", [Token("VARIABLE", var)])
                                    for var in variables
                                ]
                            ])
                            disjuncts.append(Tree("literal", [atom]))
                            new = Tree("cda", [
                                Token("NAME", f"{tree.children[0].value}_{index}"),
                                Token("FORMULA_ROLE", "axiom"),
                                disjunct
                            ])
                            match tree.children[1].value:
                                case "axiom" | "hypothesis" | "definition" | "assumption" | "lemma" | "negated_conjecture":
                                    new.dependencies = tree.dependencies + [negated(atom)]
                                case "theorem" | "corollary" | "conjecture":
                                    new.dependencies = [negated(f) for f in tree.dependencies] + [atom]
                            self.new.append(Tree("formula", [new]))
                            index += 1
                tree.data = "cnf"
                tree.children = [tree.children[0], tree.children[1], Tree("cnf_formula", [Tree("disjunction", disjuncts)])]
            case "conjunction":
                formulas = []
                for conjunct in tree.children[2].children:
                    match conjunct.data:
                        case "fof_atom" | "fof_negated_atom":
                            formulas.append(Tree("formula", [Tree("cnf", [
                                Token("NAME", f"{tree.children[0].value}_{index}"),
                                tree.children[1],
                                Tree("cnf_formula", [Tree("disjunction", [Tree("literal", [VariableCopy().transform(f)]) for f in tree.dependencies] + [Tree("literal", [conjunct])])])
                            ])]))
                            index += 1
                        case "disjunction":
                            variables = FreeVariables().transform(conjunct)
                            atom = Tree("fof_atom", [
                                Token("FUNCTOR", f"{tree.children[0].value}_{index}"),
                                *[
                                    Tree("fof_term", [Token("VARIABLE", var)])
                                    for var in variables
                                ]
                            ])
                            formulas.append(Tree("formula", [Tree("cnf", [
                                Token("NAME", f"{tree.children[0].value}_{index}"),
                                tree.children[1],
                                Tree("cnf_formula", [Tree("disjunction", [Tree("literal", [VariableCopy().transform(f)]) for f in tree.dependencies] + [Tree("literal", [atom])])])
                            ])]))
                            new = Tree("cda", [
                                Token("NAME", f"{tree.children[1].value}_{index}"),
                                Token("FORMULA_ROLE", "axiom"),
                                conjunct
                            ])
                            match tree.children[1].value:
                                case "axiom" | "hypothesis" | "definition" | "assumption" | "lemma" | "negated_conjecture":
                                    new.dependencies = tree.dependencies + [negated(atom)]
                                case "theorem" | "corollary" | "conjecture":
                                    new.dependencies = [negated(f) for f in tree.dependencies] + [atom]
                            self.new.append(Tree("formula", [new]))
                            index += 1
                self.new += formulas
                tree.data = "delete"

    def tptp_file(self, tree):
        tree.children = [child for child in tree.children if child.children[0].data != "delete"]
        tree.children += self.new
        self.new = []


class VariableRenamer(Visitor):
    def __init__(self, counter=0):
        self.name = dict()
        self.counter = counter

    def fof_term(self, token):
        if token.children[0].type == "VARIABLE":
            if token.children[0].value not in self.name:
                self.name[token.children[0].value] = f"X{self.counter}"
                self.counter += 1
            token.children[0].value = self.name[token.children[0].value]


class FOLConverter(Transformer):
    def __init__(self):
        self.n_goals = 0
    
    def fof_term(self, children):
        if children[0].type == "FUNCTOR":
            return Function(children[0].value, len(children) - 1)(*children[1:])
        if children[0].type == "VARIABLE":
            return Variable(children[0].value)
    
    def fof_atom(self, children):
        if len(children) == 3 and type(children[1]) == Token:
            return Literal(eq(children[0], children[2]), True)
        else:
            return Literal(Predicate(children[0].value, len(children) - 1)(*children[1:]))

    def fof_negated_atom(self, children):
        children[0].polarity = False
        return children[0]

    def literal(self, children):
        return children[0]

    def disjunction(self, children):
        return Clause(*children)

    def cnf_formula(self, children):
        return children[0]

    def cnf(self, children):
        if children[1] == Token("FORMULA_ROLE", "negated_conjecture"):
            children[2].literals += (Literal(Predicate(f"goal_{self.n_goals}", 0)(), True),)
            self.n_goals += 1
        return children[2]

    def formula(self, children):
        return children[0]

    def tptp_file(self, children):
        return Problem(*children, *[
            Clause(Literal(Predicate(f"goal_{i}", 0)(), False))
            for i in range(self.n_goals)
        ])

