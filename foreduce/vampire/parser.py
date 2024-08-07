from lark import Transformer, Token
from itertools import chain

from foreduce.fol.logic import _EQ, _TRUE, _FALSE, Function, Literal, Predicate, Variable, Clause, Problem
from foreduce.vampire.lexer import vampire_lexer


def read_file(file):
    with open(file) as f:
        return Formulas().transform(vampire_lexer.parse(f.read()))


def read_string(str):
    return Formulas().transform(vampire_lexer.parse(str))


class Formulas(Transformer):
    def fof_term(self, children):
        if children[0].type == "FUNCTOR":
            return Function(children[0].value, len(children) - 1)(*children[1:])
        if children[0].type == "VARIABLE":
            return Variable(children[0].value)
    
    def fof_atom(self, children):
        if len(children) == 3 and type(children[1]) == Token:
            return Literal(_EQ(children[0], children[2]), True)
        else:
            if children[0].value == "$true":
                return Literal(_TRUE(), True)
            if children[0].value == "$false":
                return Literal(_FALSE(), True)
            return Literal(Predicate(children[0].value, len(children) - 1)(*children[1:]))

    def fof_negated_atom(self, children):
        children[0].polarity = False
        return children[0]

    def literal(self, children):
        return children[0]

    def disjunction(self, children):
        return Clause(*children)

    def step(self, children):
        return children

    def start(self, children):
        clauses = []
        tree = []
        for child in children:
            clauses.append(Clause(*child[1].literals))
            tree.append([i-1 for i in child[2][1]])
        return Problem(*clauses), tree
    
    def NUMBER(self, token):
        return int(token.value)

    def rule(self, children):
        if len (children) == 1:
            return [children[0], []]
        return children
    
    def LOWER_WORD(self, token):
        return token.value

    def name(self, children):
        return " ".join(chain(children))

    def premises(self, children):
        return children
