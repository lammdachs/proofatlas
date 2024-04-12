from itertools import chain
from lark import Transformer, Tree, Token, Visitor
from lark.visitors import Transformer_InPlace

from foreduce.tptp.lexer import tptplexer


def normalform(tree, include_path='/'):
    tree = include(tree, include_path)
    tree = opsimp(tree)
    tree = simplify(tree)
    tree = nnf(tree)
    tree = simplify(tree)
    tree = eliminate_equivalences(tree)
    tree = nnf(tree)
    tree = skolemize(tree)
    tree = simplify(tree)
    tree = cnf(tree)
    return tree


def include(tree, include_path='/'):
    transformer = Include(include_path=include_path)
    transformer.transform(tree)
    return tree

class Include(Transformer_InPlace):
    def __init__(self, include_path='/'):
        self.include_path = include_path
    
    def include(self, tree):
        with open(self.include_path + tree.children[0].value, "r") as f:
            data = f.read()
        input = tptplexer.parse(data)
        Include(include_path=self.include_path).transform(input)
        return Tree("include", [i for i in input.children if input.data == "formula"])
    
    def tptp_file(self, children):
        formulas  = [formula for formula in children if formula.data == "formula"]
        included = list(chain(*[include.children for include in children if include.data == "include"]))
        return Tree("tptp_file", formulas + included)

# Remove all connectives except for ~, |, &, <=>, <~>

def opsimp(tree):
    transformer = OpSimplify()
    transformer.transform(tree)
    return tree

class OpSimplify(Transformer_InPlace):
    # p => q -> ~p | q
    # p <= q -> p | ~q
    # p ~| q -> ~(p | q)
    # p ~& q -> ~(p & q)
    def fof_binary(self, children):
        match children[1].value:
            case "=>":
                return Tree("fof_binary", [Tree("fof_unary", [Tree("fof_negation", [children[0]])]), Token("BINARY_CONNECTIVE", "|"), children[2]])  
            case "<=":
                return Tree("fof_binary", [children[0], Token("BINARY_CONNECTIVE", "|"), Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_formula", [children[2]])])])])])
            case "~|":
                return Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [children[0], Token("BINARY_CONNECTIVE", "|"), children[2]])])])])])
            case "~&":
                return Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [children[0], Token("BINARY_CONNECTIVE", "&"), children[2]])])])])])
        return Tree("fof_binary", children)
            

# Simplify occurences of $true, $false and redundant parentheses

def simplify(tree):
    transformer = Simplify()
    transformer.transform(tree)
    return tree


class Simplify(Transformer_InPlace):
    def __init__(self):
        self.unchanged = False

    def fof_negation(self, children):
        if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
            self.unchanged = False
            return Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])
        elif children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
            self.unchanged = False
            return Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])
        return Tree("fof_negation", children)

    def fof_binary(self, children):
        match children[1].value:
            case "|":
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]) or children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])])
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return children[2]
                if children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return children[0]
            case "&":
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]) or children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])])
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return children[2]
                if children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return children[0]
            case "<=>":
                if children[0] == children[2]:
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])])
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return children[2]
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_negation", [children[2]])])
                if children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return children[0]
                if children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_negation", [children[0]])])
            case "<~>":
                if children[0] == children[2]:
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])])
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_negation", [children[2]])])
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return children[2]
                if children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_negation", [children[0]])])
                if children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return children[0]
        if children[2].data == "fof_unary":
            if children[2].children[0].data == "fof_formula":
                if children[2].children[0].children[0].data == "fof_binary":
                    self.unchanged = False
                    return Tree("fof_binary", [children[0], children[1], children[2].children[0].children[0]])
        return Tree("fof_binary", children)

    def fof_unary(self, children):
        if children[0].data == "fof_formula":
            if children[0].children[0].data == "fof_unary":
                self.unchanged = False
                return children[0].children[0]
        return Tree("fof_unary", children)


# Push negations inwards

def nnf(tree):
    transformer = NNF()
    while not transformer.unchanged:
        transformer.unchanged = True
        transformer.transform(tree)
    return tree

class NNF(Transformer_InPlace):
    def __init__(self):
        self.unchanged = False
    
    def fof_unary(self, children):
        if children[0].data == "fof_negation":
            if children[0].children[0].children[0].data == "fof_atom":
                return Tree("fof_unary", children)
            if children[0].children[0].children[0].data == "fof_negation":
                self.unchanged = False
                return children[0].children[0].children[0].children[0]
            if children[0].children[0].children[0].data == "fof_quantified_formula":
                self.unchanged = False
                match children[0].children[0].children[0].children[0].value:
                    case "!":
                        return Tree("fof_unary", [Tree("fof_quantified_formula", [
                            Token("QUANTIFIER", "?"),
                            *children[0].children[0].children[0].children[1:-1],
                            Tree("fof_unary", [Tree("fof_negation", [children[0].children[0].children[0].children[-1]])])
                        ])])
                    case "?":
                        return Tree("fof_unary", [Tree("fof_quantified_formula", [
                            Token("QUANTIFIER", "!"),
                            *children[0].children[0].children[0].children[1:-1],
                            Tree("fof_unary", [Tree("fof_negation", [children[0].children[0].children[0].children[-1]])])
                        ])])
            if children[0].children[0].children[0].data == "fof_formula":
                f = children[0].children[0].children[0].children[0]
                if f.data == "fof_unary":
                    unchanged = False
                    return Tree("fof_unary", [Tree("fof_negation", [f])])
                if f.data == "fof_binary":
                    match f.children[1].value:
                        case "|":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [Tree("fof_unary", [Tree("fof_negation", [f.children[0]])]), Token("BINARY_CONNECTIVE", "&"), Tree("fof_unary", [Tree("fof_negation", [f.children[2]])])])])])
                        case "&":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [Tree("fof_unary", [Tree("fof_negation", [f.children[0]])]), Token("BINARY_CONNECTIVE", "|"), Tree("fof_unary", [Tree("fof_negation", [f.children[2]])])])])])
                        case "<=>":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [f.children[0], Token("BINARY_CONNECTIVE", "<~>"), f.children[2]])])])
                        case "<~>":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [f.children[0], Token("BINARY_CONNECTIVE", "<=>"), f.children[2]])])])
        return Tree("fof_unary", children)


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
    visitor.transform(tree)
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


class EliminateEquivalence(Transformer_InPlace):
    def __init__(self):
        self.counter = 0
        self.unchanged = False
        self.equivs = []
    
    def fof_binary(self, children):
        if children[1] == "<=>":
            free_variables = FreeVariables().transform(children[0]).union(FreeVariables().transform(children[2]))
            result = Tree("fof_unary", [Tree("fof_atom", [
                Token("FUNCTOR", f"equiv_{self.counter}"),
                *[
                    Tree("fof_term", [Tree("fof_variable", [Token("VARIABLE", var)])])
                    for var in free_variables
                ]
            ])])
            self.unchanged = False
            self.counter += 1
            self.equivs.append([children[0], children[2], free_variables])
            return result
        if children[1] == "<~>":
            free_variables = FreeVariables().transform(children[0]).union(FreeVariables().transform(children[2]))
            result = Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_atom", [
                Token("FUNCTOR", f"equiv_{self.counter}"),
                *[
                    Tree("fof_term", [Tree("fof_variable", [Token("VARIABLE", var)])])
                    for var in free_variables
                ]
            ])])])
            ])
            self.unchanged = False
            self.counter += 1
            self.equivs.append([children[0], children[2], free_variables])
            return result
        return Tree("fof_binary", children)


def _positive_equiv_axioms(counter, left, right, variables):
    return [
        OpSimplify().transform(Tree("tptp_input", [
            Tree("formula", [
                Tree("fof", [
                    Token("NAME", f"equivalence_{counter}_0_0"),
                    Token("FORMULA_ROLE", "axiom"),
                    Tree("fof_formula", [
                        Tree("fof_binary", [
                            Tree("fof_unary", [
                                Tree("fof_atom", [
                                    Token("FUNCTOR", f"equiv_{counter}"),
                                    *[
                                        Tree("fof_term", [Tree("fof_variable", [Token("VARIABLE", var)])])
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
            ])
        ])),
        OpSimplify().transform(Tree("tptp_input", [
            Tree("formula", [
                Tree("fof", [
                    Token("NAME", f"equivalence_{counter}_0_1"),
                    Token("FORMULA_ROLE", "axiom"),
                    Tree("fof_formula", [
                        Tree("fof_binary", [
                            Tree("fof_unary", [
                                Tree("fof_atom", [
                                    Token("FUNCTOR", f"equiv_{counter}"),
                                    *[
                                        Tree("fof_term", [Tree("fof_variable", [Token("VARIABLE", var)])])
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
            ])
        ]))
    ]


def _negative_equiv_axioms(counter, left, right, variables):
    return [
        OpSimplify().transform(Tree("tptp_input", [
            Tree("formula", [
                Tree("fof", [
                    Token("NAME", f"equivalence_{counter}_1"),
                    Token("FORMULA_ROLE", "axiom"),
                    Tree("fof_formula", [
                        Tree("fof_binary", [
                            Tree("fof_unary", [
                                    Tree("fof_atom", [
                                        Token("FUNCTOR", f"equiv_{counter}"),
                                        *[
                                            Tree("fof_term", [Tree("fof_variable", [Token("VARIABLE", var)])])
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
            ])
        ]))
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
    transformer = PushDisjunctionsInward()
    while not transformer.unchanged:
        transformer.unchanged = True
        tree = simplify(transformer.transform(tree))
    return CNF().transform(tree)


class PushDisjunctionsInward(Transformer_InPlace):
    def __init__(self):
        self.unchanged = False    
    
    def fof_binary(self, children):
        if children[1] == Token("BINARY_CONNECTIVE", "|"):
            if children[0].children[0].data == "fof_formula":
                if children[0].children[0].children[0].data == "fof_binary":
                    if children[0].children[0].children[0].children[1] == Token("BINARY_CONNECTIVE", "&"):
                        self.unchanged = False
                        return Tree("fof_binary", [
                            Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [
                                children[0].children[0].children[0].children[0],
                                Token("BINARY_CONNECTIVE", "|"),
                                children[2]
                            ])])]),
                            Token("BINARY_CONNECTIVE", "&"),
                            Tree("fof_binary", [
                                Tree("fof_unary", [Tree("fof_formula", [children[0].children[0].children[0].children[2]])]),
                                Token("BINARY_CONNECTIVE", "|"),
                                children[2]
                            ])
                        ])
            if children[2].data == "fof_binary":
                if children[2].children[1] == Token("BINARY_CONNECTIVE", "&"):
                    self.unchaged = False
                    return Tree("fof_binary", [
                        Tree("fof_unary", [Tree("fof_formula", [Tree("fof_binary", [
                            children[0],
                            Token("BINARY_CONNECTIVE", "|"),
                            children[2].children[0]
                        ])])]),
                        Token("BINARY_CONNECTIVE", "&"),
                        Tree("fof_binary", [
                            children[0],
                            Token("BINARY_CONNECTIVE", "|"),
                            children[2].children[2]
                        ])
                    ])
        
        return Tree("fof_binary", children)


class CNF(Transformer_InPlace):
    def fof_unary(self, children):
        match children[0].data:
            case "fof_atom":
                return children[0]
            case "fof_negation":
                return Tree("fof_negated_atom", children[0].children)
            case "disjunction":
                return children[0]
            case "conjunction":
                return children[0]
    
    def fof_binary(self, children):
        if children[1] == Token("BINARY_CONNECTIVE", "|"):
            match children[0].data:
                case "fof_atom" | "fof_negated_atom":
                    disjunts1 = [children[0]]
                case "disjunction":
                    disjunts1 = children[0].children
            match children[2].data:
                case "fof_atom" | "fof_negated_atom":
                    disjunts2 = [children[2]]
                case "disjunction":
                    disjunts2 = children[2].children
            return Tree("disjunction", disjunts1 + disjunts2)
        if children[1] == Token("BINARY_CONNECTIVE", "&"):
            match children[0].data:
                case "fof_atom" | "fof_negated_atom":
                    conjuncts1 = [Tree("disjunction", [children[0]])]
                case "disjunction":
                    conjuncts1 = [children[0]]
                case "conjunction":
                    conjuncts1 = children[0].children
            match children[2].data:
                case "fof_atom" | "fof_negated_atom":
                    conjuncts2 = [Tree("disjunction", [children[2]])]
                case "disjunction":
                    conjuncts2 = [children[2]]
                case "conjunction":
                    conjuncts2 = children[2].children
            return Tree("conjunction", conjuncts1 + conjuncts2)
    
    def fof_formula(self, children):
        return children[0]

    def fof(self, children):
        match children[2].data:
            case "fof_atom":
                children[2] = [Tree("disjunction", [children[2]])]
            case "fof_negated_atom":
                children[2] = [Tree("disjunction", [children[2]])]
            case "disjunction":
                children[2] = [children[2]]
            case "conjunction":
                children[2] = children[2].children
        return Tree("cnf_collection", [
            Tree("cnf", [Token("NAME", children[0] + f"_{i}"), children[1], Tree("cnf_formula", [disjunction])])
            for i, disjunction in enumerate(children[2])
        ])
    
    def cnf(self, children):
        return Tree("cnf_collection", [Tree("cnf", children)])

    def tptp_file(self, children):
        cnfs = [
            Tree("formula", [cnf])
            for cnf in list(chain(
                *[child.children[0].children
                  for child in children
                  if child.data == "formula"]
                ))
        ]
        includes = [child for child in children if child.data == "include"]
        return Tree("tptp_file", includes + cnfs)
    
    