from lark import Tree, Token
from lark.visitors import Transformer_InPlace


class OpSimplify(Transformer_InPlace):
    # p <= q -> q => p
    # p ~| q -> ~(p | q)
    # p ~& q -> ~(p & q)
    def fof_binary(self, children):
        match children[1].value:
            case "<=":
                return Tree("fof_binary", [children[2], Token("BINARY_CONNECTIVE", "=>"), children[0]])
            case "~|":
                return Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_logic_formula", [Tree("fof_binary", [children[0], Token("BINARY_CONNECTIVE", "|"), children[2]])])])])])
            case "~&":
                return Tree("fof_unary", [Tree("fof_negation", [Tree("fof_unary", [Tree("fof_logic_formula", [Tree("fof_binary", [children[0], Token("BINARY_CONNECTIVE", "&"), children[2]])])])])])
        return Tree("fof_binary", children)


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
            case "=>":
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return children[2]
                if children[0] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])])
                if children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$true")])])
                if children[2] == Tree("fof_unary", [Tree("fof_atom", [Token("DEFINED_UNARY_PREDICATE", "$false")])]):
                    self.unchanged = False
                    return Tree("fof_unary", [Tree("fof_negation", [children[0]])])
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
        return Tree("fof_binary", children)



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
            if children[0].children[0].children[0].data == "fof_logic_formula":
                f = children[0].children[0].children[0].children[0]
                if f.data == "fof_unary":
                    unchanged = False
                    return Tree("fof_unary", [Tree("fof_negation", [f])])
                if f.data == "fof_binary":
                    match f.children[1].value:
                        case "|":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_logic_formula", [Tree("fof_binary", [Tree("fof_unary", [Tree("fof_negation", [f.children[0]])]), Token("BINARY_CONNECTIVE", "&"), Tree("fof_unary", [Tree("fof_negation", [f.children[2]])])])])])
                        case "&":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_logic_formula", [Tree("fof_binary", [Tree("fof_unary", [Tree("fof_negation", [f.children[0]])]), Token("BINARY_CONNECTIVE", "|"), Tree("fof_unary", [Tree("fof_negation", [f.children[2]])])])])])
                        case "=>":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_logic_formula", [Tree("fof_binary", [f.children[0], Token("BINARY_CONNECTIVE", "&"), Tree("fof_unary", [Tree("fof_negation", [f.children[2]])])])])])
                        case "<=>":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_logic_formula", [Tree("fof_binary", [f.children[0], Token("BINARY_CONNECTIVE", "<~>"), f.children[2]])])])
                        case "<~>":
                            self.unchanged = False
                            return Tree("fof_unary", [Tree("fof_logic_formula", [Tree("fof_binary", [f.children[0], Token("BINARY_CONNECTIVE", "<=>"), f.children[2]])])])
        return Tree("fof_unary", children)

