from lark import Lark

tptplexer = Lark(r"""
    %import common.WS
    %ignore WS
    %ignore COMMENT_LINE
    %ignore COMMENT_BLOCK

    tptp_file : (formula | include)*

    formula : fof | cnf
    
    include : "include(" FILE_NAME formula_selection* ")."
    formula_selection : ",[" NAME ("," NAME)* "]"

    fof : "fof(" NAME "," FORMULA_ROLE "," fof_formula")."
    cnf : "cnf(" NAME "," FORMULA_ROLE "," cnf_formula")."

    FORMULA_ROLE : "axiom" | "hypothesis" | "definition" | "assumption" | "lemma" | "theorem" | "corollary" | "conjecture" | "negated_conjecture"

    fof_formula : fof_binary | fof_unary
    fof_binary : fof_unary BINARY_CONNECTIVE fof_unary | fof_unary BINARY_CONNECTIVE fof_binary
    fof_unary : fof_quantified_formula | fof_negation | fof_atom | "(" fof_formula ")"

    fof_quantified_formula : FOF_QUANTIFIER "[" VARIABLE ("," VARIABLE)* "] :" fof_unary
    fof_negation : "~" fof_unary
    fof_atom : FUNCTOR | FUNCTOR "(" fof_term ("," fof_term)* ")" | DEFINED_UNARY_PREDICATE | fof_term DEFINED_BINARY_PREDICATE fof_term
    fof_term : FUNCTOR | FUNCTOR "(" fof_term ("," fof_term)* ")" | VARIABLE

    FOF_QUANTIFIER : "!" | "?"
    BINARY_CONNECTIVE : "<=>" | "=>" | "<=" | "<~>" | "~|" | "~&" | "&" | "|"
    DEFINED_UNARY_PREDICATE :  "$true" | "$false"
    DEFINED_BINARY_PREDICATE : "=" | "!="

    cnf_formula : disjunction | "(" disjunction ")"
    disjunction : literal ("|" literal)*
    literal : fof_atom | fof_negated_atom
    fof_negated_atom : "~" fof_atom

    FUNCTOR : ATOMIC_WORD
    VARIABLE : UPPER_WORD

    NAME : ATOMIC_WORD | "0".."9"+
    ATOMIC_WORD : LOWER_WORD | SINGLE_QUOTED
    FILE_NAME : SINGLE_QUOTED
    COMMENT_LINE : /%[^\n]*/
    COMMENT_BLOCK : "/*" NOT_STAR_SLASH? "*"+ "/"
    NOT_STAR_SLASH : ("^*"* "*"+ "^/*") ("^*")*
    SINGLE_QUOTED : "'" SQ_CHAR+ "'"
    UPPER_WORD : UPPER_ALPHA ALPHA_NUMERIC*
    LOWER_WORD : LOWER_ALPHA ALPHA_NUMERIC*
    SQ_CHAR : " ".."&" | "(".."~" | "\'"
    LOWER_ALPHA : "a".."z"
    UPPER_ALPHA : "A".."Z"
    ALPHA_NUMERIC : LOWER_ALPHA | UPPER_ALPHA | "0".."9" | "_"
""", start="tptp_file")
