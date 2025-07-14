#!/usr/bin/env python3
"""
Interactive proof exploration with ProofAtlas

This example shows how to step through a proof search interactively,
examining each inference as it's generated.
"""

from proofatlas import ProofState


def interactive_proof_search():
    """Interactive step-by-step proof exploration"""
    print("ProofAtlas Interactive Proof Explorer")
    print("=" * 60)
    
    state = ProofState()
    
    # Set up an interesting problem
    print("\nProblem: Prove that if f is injective and f(a)=f(b), then a=b")
    print("\nAxioms:")
    print("  1. f is injective: f(X)=f(Y) → X=Y")
    print("  2. f(a) = f(b)")
    print("  3. Goal (negated): a ≠ b")
    
    state.add_clauses_from_tptp("""
    cnf(injective, axiom, f(X) != f(Y) | X = Y).
    cnf(equal_images, axiom, f(a) = f(b)).
    cnf(not_equal, negated_conjecture, a != b).
    """)
    
    # Enable superposition for equality
    state.set_use_superposition(True)
    state.set_literal_selection("max_weight")
    
    print("\nInitial clause set:")
    for i in range(state.num_clauses()):
        info = state.get_clause_info(i)
        print(f"  [{i}] {info.clause_string}")
        print(f"       Type: {'Unit' if info.is_unit else 'Non-unit'}, "
              f"Weight: {info.weight}, "
              f"Equality: {'Yes' if info.is_equality else 'No'}")
    
    print("\nPress Enter to step through the proof search...")
    print("(Type 'q' to quit, 'a' to auto-complete)")
    
    step = 0
    auto = False
    
    while state.num_unprocessed() > 0 and not state.contains_empty_clause():
        if not auto:
            user_input = input("\n> ")
            if user_input.lower() == 'q':
                print("Exiting...")
                return
            elif user_input.lower() == 'a':
                auto = True
                print("Auto-completing proof search...")
        
        step += 1
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        print('='*60)
        
        # Select given clause
        given_id = state.select_given_clause(strategy="smallest")
        if given_id is None:
            print("No more clauses to process - saturated!")
            break
        
        given_info = state.get_clause_info(given_id)
        print(f"\nGiven clause: [{given_id}] {given_info.clause_string}")
        print(f"  Properties: Weight={given_info.weight}, "
              f"Literals={given_info.num_literals}")
        
        # Show what literals are in the clause
        print(f"  Literals: {', '.join(given_info.literal_strings)}")
        
        # Generate inferences
        inferences = state.generate_inferences(given_id)
        print(f"\nGenerated {len(inferences)} potential inferences:")
        
        # Show each inference
        new_clauses = []
        for i, inf in enumerate(inferences):
            parents_str = f"[{','.join(map(str, inf.parent_ids))}]"
            print(f"  {i+1}. {inf.clause_string} "
                  f"(from {parents_str} by {inf.rule_name})")
            
            # Try to add it
            new_id = state.add_inference(inf)
            if new_id is not None:
                new_clauses.append(new_id)
                print(f"     → Added as clause [{new_id}]")
            else:
                print(f"     → Rejected (redundant)")
        
        # Process the given clause
        state.process_clause(given_id)
        
        # Show current statistics
        stats = state.get_statistics()
        print(f"\nStatistics: {stats['total']} total, "
              f"{stats['processed']} processed, "
              f"{stats['unprocessed']} in queue")
        
        # Check if we found the empty clause
        if state.contains_empty_clause():
            print("\n" + "="*60)
            print("*** PROOF FOUND! ***")
            print("="*60)
            break
    
    # Show the proof if found
    if state.contains_empty_clause():
        show_proof_trace(state)
    else:
        print("\nProof search completed without finding a contradiction.")


def show_proof_trace(state: ProofState):
    """Display the proof trace in a nice format"""
    print("\nProof Trace (showing derivation of ⊥):")
    print("-" * 60)
    
    trace = state.get_proof_trace()
    
    # Group by derivation level
    levels = {}
    for step in trace:
        if not step.parent_ids:  # Input clause
            level = 0
        else:
            # Level is max parent level + 1
            level = max(levels.get(pid, 0) for pid in step.parent_ids) + 1
        levels[step.clause_id] = level
    
    # Display by level
    max_level = max(levels.values()) if levels else 0
    
    for level in range(max_level + 1):
        if level == 0:
            print("\nInput clauses:")
        else:
            print(f"\nLevel {level} inferences:")
        
        for step in trace:
            if levels.get(step.clause_id, 0) == level:
                print(f"  [{step.clause_id}] {step.clause_string}")
                if step.parent_ids:
                    parents = ', '.join(str(p) for p in step.parent_ids)
                    print(f"      from [{parents}] by {step.rule_name}")


def guided_example():
    """A guided example with explanations"""
    print("\n\nGuided Example: Understanding Resolution")
    print("=" * 60)
    
    state = ProofState()
    
    print("\nWe'll prove a simple theorem step by step.")
    print("Given: P(a) and ∀X.(P(X) → Q(X))")
    print("Prove: Q(a)")
    print("\nIn CNF form:")
    
    state.add_clauses_from_tptp("""
    cnf(fact, axiom, p(a)).
    cnf(rule, axiom, ~p(X) | q(X)).
    cnf(goal, negated_conjecture, ~q(a)).
    """)
    
    for i in range(state.num_clauses()):
        print(f"  [{i}] {state.clause_to_string(i)}")
    
    input("\nPress Enter to see the first inference...")
    
    # Step 1
    given_id = state.select_given_clause()
    print(f"\n1. Select clause [{given_id}]: {state.clause_to_string(given_id)}")
    
    inferences = state.generate_inferences(given_id)
    print(f"   This is a unit clause with predicate p(a).")
    print(f"   Looking for clauses with ~p(...) to resolve with...")
    
    if inferences:
        inf = inferences[0]
        print(f"\n2. Found matching literal in clause [1]: ~p(X)")
        print(f"   Unifying p(a) with p(X) gives substitution {{X/a}}")
        print(f"   Resolution produces: {inf.clause_string}")
        
        new_id = state.add_inference(inf)
        if new_id:
            print(f"   Added as clause [{new_id}]")
    
    state.process_clause(given_id)
    
    input("\nPress Enter to continue...")
    
    # Continue until proof found
    while state.num_unprocessed() > 0 and not state.contains_empty_clause():
        given_id = state.select_given_clause()
        if given_id is None:
            break
            
        print(f"\n3. Now processing clause [{given_id}]: {state.clause_to_string(given_id)}")
        
        inferences = state.generate_inferences(given_id)
        for inf in inferences:
            print(f"   Can resolve with clause {inf.parent_ids}: {inf.clause_string}")
            new_id = state.add_inference(inf)
            
            if new_id and state.clause_to_string(new_id) == "⊥":
                print(f"\n   *** Found empty clause! ***")
                print(f"   This means we have derived a contradiction.")
                print(f"   Since we negated the goal, the original goal is proven!")
        
        state.process_clause(given_id)
    
    print("\nThe proof is complete!")


if __name__ == "__main__":
    print("Choose an example:")
    print("1. Interactive proof search")
    print("2. Guided resolution example")
    
    choice = input("\nEnter choice (1-2): ")
    
    if choice == "1":
        interactive_proof_search()
    elif choice == "2":
        guided_example()
    else:
        print("Running all examples...")
        interactive_proof_search()
        guided_example()