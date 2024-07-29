from lark import Lark

vampire_lexer = Lark(r"""
    %import common.WS
    %ignore WS
    %ignore COMMENT_LINE

    start : step+
    step : "[SA] new:" NUMBER "." disjunction rule "\n"
    rule : "[" name premises "]"
    name : LOWER_WORD+
    premises : NUMBER ("," NUMBER)*

    disjunction : literal ("|" literal)*
    literal : fof_atom | fof_negated_atom
    fof_negated_atom : "~" fof_atom
    fof_atom : FUNCTOR | FUNCTOR "(" fof_term ("," fof_term)* ")" | DEFINED_UNARY_PREDICATE | fof_term DEFINED_BINARY_PREDICATE fof_term
    fof_term : FUNCTOR | FUNCTOR "(" fof_term ("," fof_term)* ")" | VARIABLE

    DEFINED_UNARY_PREDICATE :  "$true" | "$false"
    DEFINED_BINARY_PREDICATE : "=" | "!="

    FUNCTOR : ATOMIC_WORD
    VARIABLE : UPPER_WORD

    ATOMIC_WORD : LOWER_WORD | SINGLE_QUOTED
    UPPER_WORD : UPPER_ALPHA ALPHA_NUMERIC*
    LOWER_WORD : LOWER_ALPHA ALPHA_NUMERIC*

    SINGLE_QUOTED : "'" SQ_CHAR+ "'"
    SQ_CHAR : " ".."&" | "(".."~" | "\'"

    ALPHA_NUMERIC : LOWER_ALPHA | UPPER_ALPHA | "0".."9" | "_"
    UPPER_ALPHA : "A".."Z"
    LOWER_ALPHA : "a".."z"
    NUMBER : "0".."9"+

    COMMENT_LINE : "%" PRINTABLE_CHAR* "\n"
    PRINTABLE_CHAR : " ".."~"
""", start="start")