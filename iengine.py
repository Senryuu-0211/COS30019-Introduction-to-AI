import sys
from typing import List, Set, Dict, Optional
from itertools import product

class KnowledgeBase:
    def __init__(self):
        self.sentences = []
        self.symbols = set()
        
    def tell(self, sentence: str):
        """Add a sentence to the knowledge base with improved parsing"""
        # Handle multiple expressions separated by semicolons
        expressions = self._split_expressions(sentence)
        
        for expression in expressions:
            # Clean and normalize the expression
            cleaned_expr = self._normalize_expression(expression)
            if cleaned_expr:
                self.sentences.append(cleaned_expr)
                # Extract symbols more robustly
                self._extract_symbols(cleaned_expr)
    
    def _split_expressions(self, sentence: str) -> List[str]:
        """Split expressions while respecting parentheses"""
        expressions = []
        current_expr = []
        paren_count = 0
        
        for char in sentence:
            if char == '(' or char == '[':
                paren_count += 1
            elif char == ')' or char == ']':
                paren_count -= 1
            
            if char == ';' and paren_count == 0:
                if current_expr:
                    expressions.append(''.join(current_expr))
                    current_expr = []
            else:
                current_expr.append(char)
        
        if current_expr:
            expressions.append(''.join(current_expr))
        
        return [expr.strip() for expr in expressions if expr.strip()]
    
    def _normalize_expression(self, expr: str) -> str:
        """Normalize logical expressions"""
        # Replace alternative symbols with standard ones
        replacements = {
            '!': '~',    # NOT
            '&&': '&',   # AND
            '||': '||',  # OR
            '→': '=>',   # IMPLIES
            '↔': '<=>',  # BICONDITIONAL
            'AND': '&',
            'OR': '||',
            'NOT': '~',
            'IMPLIES': '=>',
            'IFF': '<=>'
        }
        
        normalized = expr
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
            
        # Handle parentheses
        normalized = normalized.replace('[', '(').replace(']', ')')
        
        return normalized.strip()
    
    def _extract_symbols(self, expr: str):
        """Extract symbols more robustly"""
        # Remove all logical operators
        operators = ['<=>', '=>', '&', '||', '~', '(', ')']
        temp_expr = expr
        for op in operators:
            temp_expr = temp_expr.replace(op, ' ')
        
        # Extract valid propositional symbols
        symbols = [word.strip() for word in temp_expr.split()]
        self.symbols.update(s for s in symbols if s and any(c.isalpha() for c in s))

def _clean_expression(expr: str) -> str:
    """Clean and normalize a logical expression"""
    # Remove whitespace
    expr = expr.strip()
    
    # Balance parentheses if needed
    open_count = expr.count('(')
    close_count = expr.count(')')
    if open_count > close_count:
        expr = expr + ')' * (open_count - close_count)
    elif close_count > open_count:
        expr = '(' * (close_count - open_count) + expr
        
    return expr

def evaluate(sentence: str, model: Dict[str, bool]) -> bool:
    """Evaluate a logical sentence given a model"""
    try:
        # Clean the expression first
        sentence = _clean_expression(sentence)
        
        # Remove outer parentheses if they exist
        while sentence.startswith('(') and sentence.endswith(')'):
            if _are_matching_parentheses(sentence):
                sentence = sentence[1:-1].strip()
            else:
                break

        # Handle special case for simple propositions
        if all(c not in sentence for c in ['=>', '&', '<=>', '||', '~', '(', ')']):
            symbol = sentence.strip()
            if symbol in model:
                return model[symbol]
            else:
                raise ValueError(f"Unknown symbol: {symbol}")

        # Handle NOT (~)
        if sentence.startswith('~'):
            inner = sentence[1:].strip()
            return not evaluate(inner, model)

        # Handle parenthesized expressions first
        if '(' in sentence:
            # Find the outermost parentheses
            stack = []
            for i, char in enumerate(sentence):
                if char == '(':
                    stack.append(i)
                elif char == ')' and stack:
                    start = stack.pop()
                    if not stack:  # Found matching outer parentheses
                        before = sentence[:start].strip()
                        inner = sentence[start+1:i].strip()
                        after = sentence[i+1:].strip()
                        
                        # Evaluate the inner expression and substitute back
                        inner_result = evaluate(inner, model)
                        if before and after:
                            new_sentence = f"{before} {str(inner_result).lower()} {after}"
                        elif before:
                            new_sentence = f"{before} {str(inner_result).lower()}"
                        elif after:
                            new_sentence = f"{str(inner_result).lower()} {after}"
                        else:
                            new_sentence = str(inner_result).lower()
                        return evaluate(new_sentence.strip(), model)

        # Handle OR (||)
        if '||' in sentence:
            left, right = _split_on_operator(sentence, '||')
            return evaluate(left, model) or evaluate(right, model)

        # Handle AND (&)
        if '&' in sentence:
            left, right = _split_on_operator(sentence, '&')
            return evaluate(left, model) and evaluate(right, model)

        # Handle biconditional (<=>)
        if '<=>' in sentence:
            left, right = _split_on_operator(sentence, '<=>')
            return evaluate(left, model) == evaluate(right, model)

        # Handle implication (=>)
        if '=>' in sentence:
            left, right = _split_on_operator(sentence, '=>')
            return (not evaluate(left, model)) or evaluate(right, model)

        # If we get here, it should be a simple proposition
        if sentence.strip() in model:
            return model[sentence.strip()]
        else:
            raise ValueError(f"Unable to evaluate expression: {sentence}")

    except Exception as e:
        raise ValueError(f"Error evaluating '{sentence}': {str(e)}")

def _are_matching_parentheses(s: str) -> bool:
    """Check if the outermost parentheses are matching"""
    if not (s.startswith('(') and s.endswith(')')):
        return False
    
    count = 0
    for i, c in enumerate(s[1:-1]):
        if c == '(':
            count += 1
        elif c == ')':
            count -= 1
        if count < 0:  # Found a closing parenthesis without matching opening
            return False
    return count == 0

def _split_on_operator(sentence: str, operator: str) -> tuple[str, str]:
    """Split a sentence on an operator while respecting parentheses"""
    count = 0
    operator_index = -1
    
    for i in range(len(sentence)):
        if sentence[i] == '(':
            count += 1
        elif sentence[i] == ')':
            count -= 1
        elif count == 0 and sentence[i:].startswith(operator):
            operator_index = i
            break
    
    if operator_index != -1:
        left = sentence[:operator_index].strip()
        right = sentence[operator_index + len(operator):].strip()
        return left, right
    
    # If we couldn't find the operator with proper parentheses handling,
    # fall back to simple split
    parts = sentence.split(operator, 1)
    return parts[0].strip(), parts[1].strip()

    
class LogicalExpression:
    @staticmethod
    def parse(expression: str) -> dict:
        """Parse a logical expression into a syntax tree"""
        # Remove whitespace
        expression = ''.join(expression.split())
        
        def parse_biconditional(expr: str) -> dict:
            if '<=>' in expr:
                left, right = expr.split('<=>', 1)
                return {'type': 'biconditional', 'left': parse_biconditional(left), 'right': parse_biconditional(right)}
            return parse_implication(expr)
            
        def parse_implication(expr: str) -> dict:
            if '=>' in expr:
                left, right = expr.split('=>', 1)
                return {'type': 'implication', 'left': parse_implication(left), 'right': parse_implication(right)}
            return parse_or(expr)
            
        def parse_or(expr: str) -> dict:
            if '||' in expr:
                left, right = expr.split('||', 1)
                return {'type': 'or', 'left': parse_or(left), 'right': parse_or(right)}
            return parse_and(expr)
            
        def parse_and(expr: str) -> dict:
            if '&' in expr:
                left, right = expr.split('&', 1)
                return {'type': 'and', 'left': parse_and(left), 'right': parse_and(right)}
            return parse_not(expr)
            
        def parse_not(expr: str) -> dict:
            if expr.startswith('~'):
                return {'type': 'not', 'expr': parse_not(expr[1:])}
            return parse_atom(expr)
            
        def parse_atom(expr: str) -> dict:
            if expr.startswith('(') and expr.endswith(')'):
                return parse_biconditional(expr[1:-1])
            return {'type': 'atom', 'symbol': expr}
            
        return parse_biconditional(expression)

class InferenceEngine:
    @staticmethod
    def tt_check_all(kb: KnowledgeBase, query: str) -> tuple[bool, int]:
        """Truth table enumeration method"""
        symbols = list(kb.symbols)
        
        # Count models where KB is true
        kb_models_count = 0
        kb_and_query_models_count = 0
        
        # Generate all possible combinations of truth values
        for values in product([False, True], repeat=len(symbols)):
            # Create a model mapping symbols to truth values
            model = dict(zip(symbols, values))
            
            try:
                # Check if all clauses in KB are satisfied
                kb_satisfied = all(evaluate(_clean_expression(s), model) for s in kb.sentences)
                
                if kb_satisfied:
                    kb_models_count += 1
                    if evaluate(_clean_expression(query), model):
                        kb_and_query_models_count += 1
            except Exception as e:
                # Only print debug information if needed
                # print(f"Debug - Model: {model}")
                # print(f"Debug - Error: {e}")
                continue
        
        # Query is entailed if it's true in all models where KB is true
        return (kb_models_count > 0 and kb_models_count == kb_and_query_models_count, kb_models_count)

    @staticmethod
    def forward_chaining(kb: KnowledgeBase, query: str) -> tuple[bool, List[str]]:
        """Forward chaining algorithm for Horn clauses"""
        count = {}  # Number of premises for each clause
        inferred = set()  # Set of inferred symbols
        agenda = []  # Symbols known to be true
        entailed_symbols = []  # Symbols in order of inference
        
        # Initialize count and agenda from KB
        for sentence in kb.sentences:
            if '=>' not in sentence:  # Simple fact
                symbol = sentence.strip()
                if symbol not in inferred:
                    agenda.append(symbol)
                    inferred.add(symbol)
                    entailed_symbols.append(symbol)
            else:
                premises, conclusion = sentence.rsplit('=>', 1)
                premises = premises.strip()
                conclusion = conclusion.strip()
                if '&' in premises:
                    count[conclusion] = len(premises.split('&'))
                else:
                    count[conclusion] = 1
        
        while agenda:
            p = agenda.pop(0)
            for sentence in kb.sentences:
                if '=>' in sentence:
                    premises, conclusion = sentence.rsplit('=>', 1)
                    premises = premises.strip()
                    conclusion = conclusion.strip()
                    
                    # Check if p is in premises
                    if '&' in premises:
                        premise_list = [prem.strip() for prem in premises.split('&')]
                        if p in premise_list and conclusion not in inferred:
                            count[conclusion] -= 1
                            if count[conclusion] == 0:
                                inferred.add(conclusion)
                                agenda.append(conclusion)
                                entailed_symbols.append(conclusion)
                    else:
                        if p == premises and conclusion not in inferred:
                            count[conclusion] -= 1
                            if count[conclusion] == 0:
                                inferred.add(conclusion)
                                agenda.append(conclusion)
                                entailed_symbols.append(conclusion)
        
        return query in inferred, entailed_symbols

    @staticmethod
    def backward_chaining(kb: KnowledgeBase, query: str) -> tuple[bool, List[str]]:
        """Backward chaining algorithm for Horn clauses"""
        entailed = set()
        entailed_symbols = []
        
        def bc_rec(symbol: str, visited: Set[str]) -> bool:
            if symbol in visited:
                return False
            
            if symbol in entailed:
                return True
                
            visited.add(symbol)
            
            # Check if symbol is a fact in KB
            for sentence in kb.sentences:
                if '=>' not in sentence and sentence.strip() == symbol:
                    entailed.add(symbol)
                    if symbol not in entailed_symbols:
                        entailed_symbols.append(symbol)
                    return True
            
            # Check implications
            for sentence in kb.sentences:
                if '=>' in sentence:
                    premises, conclusion = sentence.rsplit('=>', 1)
                    premises = premises.strip()
                    conclusion = conclusion.strip()
                    
                    if conclusion == symbol:
                        all_premises_true = True
                        if '&' in premises:
                            for premise in premises.split('&'):
                                premise = premise.strip()
                                if not bc_rec(premise, visited.copy()):
                                    all_premises_true = False
                                    break
                        else:
                            if not bc_rec(premises, visited.copy()):
                                all_premises_true = False
                                
                        if all_premises_true:
                            entailed.add(symbol)
                            if symbol not in entailed_symbols:
                                entailed_symbols.append(symbol)
                            return True
                            
            return False
            
        result = bc_rec(query, set())
        return result, entailed_symbols

def main():
    if len(sys.argv) != 3:
        print("Usage: python iengine.py <filename> <method>")
        sys.exit(1)
        
    filename = sys.argv[1]
    method = sys.argv[2]
    
    kb = KnowledgeBase()
    query = ""
    
    # Read input file
    with open(filename, 'r') as f:
        lines = f.readlines()
        tell_mode = False
        ask_mode = False
        
        for line in lines:
            line = line.strip()
            if line == "TELL":
                tell_mode = True
            elif line == "ASK":
                tell_mode = False
                ask_mode = True
            elif tell_mode and line:
                kb.tell(line)
            elif ask_mode and line:
                query = line
    
    # Apply the specified inference method
    if method == "TT":
        result, models = InferenceEngine.tt_check_all(kb, query)
        print(f"YES: {models}" if result else "NO")
    elif method == "FC":
        result, symbols = InferenceEngine.forward_chaining(kb, query)
        print(f"YES: {', '.join(symbols)}" if result else "NO")
    elif method == "BC":
        result, symbols = InferenceEngine.backward_chaining(kb, query)
        print(f"YES: {', '.join(symbols)}" if result else "NO")
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)

if __name__ == "__main__":
    main()