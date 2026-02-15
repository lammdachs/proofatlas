<script lang="ts">
</script>

<svelte:head>
	<title>ProofAtlas - Documentation</title>
</svelte:head>

<div class="flex flex-col md:flex-row gap-8">
	<!-- Sidebar -->
	<nav class="md:w-56 shrink-0 md:sticky md:top-20 md:self-start">
		<h3 class="font-display font-semibold text-sm uppercase tracking-wide text-text-muted mb-3">Contents</h3>
		<ul class="space-y-1.5 text-sm">
			<li><a href="#overview" class="text-text-muted hover:text-accent transition-colors">Overview</a></li>
			<li><a href="#saturation" class="text-text-muted hover:text-accent transition-colors">Saturation-Based Proving</a></li>
			<li><a href="#calculus" class="text-text-muted hover:text-accent transition-colors">Inference Calculus</a></li>
			<li><a href="#selection" class="text-text-muted hover:text-accent transition-colors">Literal Selection</a></li>
			<li><a href="#simplification" class="text-text-muted hover:text-accent transition-colors">Simplification Rules</a></li>
			<li><a href="#ml" class="text-text-muted hover:text-accent transition-colors">Machine Learning</a></li>
			<li><a href="#examples" class="text-text-muted hover:text-accent transition-colors">Examples</a></li>
		</ul>
	</nav>

	<!-- Content -->
	<div class="flex-1 min-w-0 space-y-12 text-text">
		<section id="overview">
			<h2 class="font-display text-2xl font-bold mb-4 text-text">Overview</h2>
			<p class="mb-3">
				ProofAtlas is a <strong>saturation-based theorem prover</strong> for first-order logic with equality.
				It implements the superposition calculus, a complete inference system for equational reasoning.
			</p>
			<p class="mb-4">
				The prover runs entirely in your browser using WebAssembly, providing a fast and private
				proof search experience without any server-side computation.
			</p>
			<h3 class="font-display font-semibold text-lg mb-2">Key Features</h3>
			<ul class="list-disc list-inside space-y-1 text-text-muted">
				<li>Complete superposition calculus for equality reasoning</li>
				<li>Multiple literal selection strategies</li>
				<li>Clause subsumption and simplification</li>
				<li>Knuth-Bendix Ordering (KBO) for term ordering</li>
				<li>Interactive proof inspection</li>
			</ul>
		</section>

		<section id="saturation">
			<h2 class="font-display text-2xl font-bold mb-4 text-text">Saturation-Based Proving</h2>
			<p class="mb-4">
				The prover uses a <strong>given clause algorithm</strong> to systematically explore the search space:
			</p>

			<div class="bg-surface-light border border-card-border rounded-lg p-5 mb-4">
				<h4 class="font-display font-semibold text-sm uppercase tracking-wide text-text-muted mb-3">Given Clause Loop</h4>
				<ol class="list-decimal list-inside space-y-2 text-sm">
					<li>Start with initial clauses from the problem (negated conjecture + axioms)</li>
					<li>Maintain two sets:
						<ul class="list-disc list-inside ml-4 mt-1 space-y-0.5 text-text-muted">
							<li><strong class="text-text">Processed</strong>: Clauses already used for inference</li>
							<li><strong class="text-text">Unprocessed</strong>: Clauses waiting to be processed</li>
						</ul>
					</li>
					<li>Repeat until proof found or resources exhausted:
						<ul class="list-disc list-inside ml-4 mt-1 space-y-0.5 text-text-muted">
							<li>Select a <em>given clause</em> from unprocessed</li>
							<li>Apply all inference rules between given clause and processed clauses</li>
							<li>Simplify and filter new clauses</li>
							<li>Check if empty clause derived &rarr; <strong class="text-green-400">Proof found!</strong></li>
							<li>Move given clause to processed set</li>
						</ul>
					</li>
				</ol>
			</div>

			<p class="text-sm text-text-muted italic border-l-3 border-blue-400 pl-3">
				<strong>Note:</strong> The empty clause represents a contradiction, proving that the
				original formula is unsatisfiable (and thus the conjecture is a theorem).
			</p>
		</section>

		<section id="calculus">
			<h2 class="font-display text-2xl font-bold mb-4 text-text">Inference Calculus</h2>
			<p class="mb-4">ProofAtlas implements the following inference rules:</p>

			<div class="space-y-6">
				<!-- Binary Resolution -->
				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-display font-semibold text-lg mb-3">Binary Resolution</h3>
					<div class="font-mono text-sm bg-surface rounded p-3 mb-3 text-center space-y-1">
						<div class="text-text-muted">C &or; L &nbsp;&nbsp;&nbsp;&nbsp; D &or; &not;L'</div>
						<div class="border-t border-card-border pt-1 text-text">(C &or; D)&sigma;</div>
					</div>
					<p class="text-sm text-text-muted mb-1">where &sigma; = mgu(L, L') is the most general unifier</p>
					<p class="text-sm">Combines two clauses by resolving complementary literals.</p>
				</div>

				<!-- Factoring -->
				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-display font-semibold text-lg mb-3">Factoring</h3>
					<div class="font-mono text-sm bg-surface rounded p-3 mb-3 text-center space-y-1">
						<div class="text-text-muted">C &or; L &or; L'</div>
						<div class="border-t border-card-border pt-1 text-text">(C &or; L)&sigma;</div>
					</div>
					<p class="text-sm text-text-muted mb-1">where &sigma; = mgu(L, L')</p>
					<p class="text-sm">Merges two identical literals in a clause.</p>
				</div>

				<!-- Superposition -->
				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-display font-semibold text-lg mb-3">Superposition</h3>
					<div class="font-mono text-sm bg-surface rounded p-3 mb-3 text-center space-y-1">
						<div class="text-text-muted">C &or; l &approx; r &nbsp;&nbsp;&nbsp;&nbsp; D &or; s[u] &approx; t</div>
						<div class="border-t border-card-border pt-1 text-text">(C &or; D &or; s[r] &approx; t)&sigma;</div>
					</div>
					<p class="text-sm text-text-muted mb-1">where &sigma; = mgu(l, u), and ordering constraints apply</p>
					<p class="text-sm mb-3">Replaces a subterm in an equality using another equality. This is the key rule for equational reasoning.</p>
					<div class="text-xs text-text-muted space-y-0.5">
						<strong>Constraints:</strong>
						<ul class="list-disc list-inside ml-2">
							<li>l &approx; r must be a positive equality</li>
							<li>l&sigma; &succ; r&sigma; (l is maximal in the equation)</li>
							<li>l &approx; r is selected in C &or; l &approx; r</li>
						</ul>
					</div>
				</div>

				<!-- Equality Resolution -->
				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-display font-semibold text-lg mb-3">Equality Resolution</h3>
					<div class="font-mono text-sm bg-surface rounded p-3 mb-3 text-center space-y-1">
						<div class="text-text-muted">C &or; &not;(s &approx; t)</div>
						<div class="border-t border-card-border pt-1 text-text">C&sigma;</div>
					</div>
					<p class="text-sm text-text-muted mb-1">where &sigma; = mgu(s, t)</p>
					<p class="text-sm">Removes reflexive negative equalities.</p>
				</div>

				<!-- Equality Factoring -->
				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-display font-semibold text-lg mb-3">Equality Factoring</h3>
					<div class="font-mono text-sm bg-surface rounded p-3 mb-3 text-center space-y-1">
						<div class="text-text-muted">C &or; s &approx; t &or; u &approx; v</div>
						<div class="border-t border-card-border pt-1 text-text">(C &or; s &approx; t &or; &not;(t &approx; v))&sigma;</div>
					</div>
					<p class="text-sm text-text-muted mb-1">where &sigma; = mgu(s, u) and s&sigma; &succ; t&sigma;, u&sigma; &succ; v&sigma;</p>
					<p class="text-sm">Special factoring rule for equalities that maintains ordering constraints.</p>
				</div>
			</div>
		</section>

		<section id="selection">
			<h2 class="font-display text-2xl font-bold mb-4 text-text">Literal Selection Strategies</h2>
			<p class="mb-4">
				Literal selection determines which literals in a clause are eligible for inference.
				Based on <a href="https://link.springer.com/chapter/10.1007/978-3-319-40229-1_21" class="text-blue-400 hover:underline">Hoder et al. "Selecting the selection" (2016)</a>,
				ProofAtlas supports four strategies:
			</p>

			<div class="space-y-4">
				{#each [
					{ name: 'Selection 0: Select All', desc: 'All literals are selected for inference.', note: 'Maximum inference potential, guarantees completeness but can generate many clauses.' },
					{ name: 'Selection 20: Select Maximal', desc: 'Selects all maximal literals based on Knuth-Bendix Ordering (KBO).', note: 'Focuses on "heavy" literals that might be easier to resolve.' },
					{ name: 'Selection 21: Unique Maximal', desc: 'Selects the unique maximal literal if one exists. Otherwise, selects max-weight negative literal if one exists. Otherwise, selects all maximal literals.', note: 'Good balance between restriction and completeness.' },
					{ name: 'Selection 22: Neg Max-Weight', desc: 'Selects the max-weight negative literal if one exists. Otherwise, selects all maximal literals.', note: 'Prioritizes eliminating negative literals, often leading to more efficient proof search.' },
				] as strategy}
					<div class="bg-surface-light border border-card-border rounded-lg p-4">
						<h3 class="font-semibold mb-1">{strategy.name}</h3>
						<p class="text-sm mb-1">{strategy.desc}</p>
						<p class="text-xs text-text-muted"><strong>Use case:</strong> {strategy.note}</p>
					</div>
				{/each}
			</div>

			<p class="mt-4 text-sm text-text-muted italic border-l-3 border-blue-400 pl-3">
				<strong>Note:</strong> No single strategy is universally best. Performance varies significantly
				depending on problem structure.
			</p>
		</section>

		<section id="simplification">
			<h2 class="font-display text-2xl font-bold mb-4 text-text">Simplification Rules</h2>
			<p class="mb-4">ProofAtlas uses several simplification techniques to keep the search space manageable:</p>

			<div class="space-y-4">
				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-display font-semibold text-lg mb-2">Subsumption</h3>
					<p class="text-sm mb-2">
						A clause C <em>subsumes</em> clause D if there exists a substitution &sigma; such that C&sigma; &sube; D.
						The subsumed clause D can be removed.
					</p>
					<p class="text-sm text-text-muted mb-2"><strong>Example:</strong> <code class="bg-surface px-1.5 py-0.5 rounded text-xs">P(X)</code> subsumes <code class="bg-surface px-1.5 py-0.5 rounded text-xs">P(a) &or; Q(b)</code></p>
					<ul class="list-disc list-inside text-sm space-y-0.5">
						<li><strong>Forward subsumption:</strong> New clause subsumed by existing clause &rarr; discard</li>
						<li><strong>Backward subsumption:</strong> New clause subsumes existing clauses &rarr; remove old clauses</li>
					</ul>
				</div>

				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-display font-semibold text-lg mb-2">Demodulation</h3>
					<p class="text-sm mb-2">Rewrites clauses using unit equalities to create simpler clauses.</p>
					<p class="text-sm text-text-muted mb-2">
						<strong>Example:</strong> Using <code class="bg-surface px-1.5 py-0.5 rounded text-xs">f(X) &approx; X</code>, rewrite
						<code class="bg-surface px-1.5 py-0.5 rounded text-xs">g(f(a)) &approx; b</code> to <code class="bg-surface px-1.5 py-0.5 rounded text-xs">g(a) &approx; b</code>
					</p>
					<div class="text-xs text-text-muted">
						<strong>Requirements:</strong> Rewrite rule must be a unit equality; one-way matching only; ordering constraint l&sigma; &succ; r&sigma;.
					</div>
				</div>

				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-display font-semibold text-lg mb-2">Tautology Deletion</h3>
					<p class="text-sm">Clauses that are always true (e.g., <code class="bg-surface px-1.5 py-0.5 rounded text-xs">P(X) &or; &not;P(X)</code>) are removed.</p>
				</div>
			</div>
		</section>

		<section id="ml">
			<h2 class="font-display text-2xl font-bold mb-4 text-text">Machine Learning for Clause Selection</h2>
			<p class="mb-4">
				ProofAtlas explores using <strong>graph neural networks</strong> to learn clause selection
				strategies. Instead of hand-crafted heuristics, we train models to predict which clauses
				are likely to be useful for finding a proof.
			</p>

			<h3 class="font-display font-semibold text-lg mb-2">Training Pipeline</h3>
			<div class="bg-surface-light border border-card-border rounded-lg p-5 mb-6">
				<ol class="list-decimal list-inside space-y-1.5 text-sm">
					<li><strong>Trace Collection:</strong> Run the prover on TPTP problems, recording which clauses were used</li>
					<li><strong>Graph Construction:</strong> Convert each clause to a tree-structured graph</li>
					<li><strong>Labeling:</strong> Mark clauses in the proof DAG as positive, others as negative</li>
					<li><strong>Training:</strong> Train a GNN to score positive clauses higher than negatives</li>
					<li><strong>Inference:</strong> Use the trained model to guide clause selection during proof search</li>
				</ol>
			</div>

			<h3 class="font-display font-semibold text-lg mb-2">Graph Representation</h3>
			<p class="mb-3 text-sm">Each clause is converted to a tree-structured graph where nodes represent syntactic elements:</p>

			<div class="overflow-x-auto mb-6">
				<table class="w-full text-sm border-collapse">
					<thead>
						<tr class="border-b border-card-border">
							<th class="text-left py-2 pr-4 text-text-muted font-semibold">Node Type</th>
							<th class="text-left py-2 text-text-muted font-semibold">Description</th>
						</tr>
					</thead>
					<tbody>
						{#each [
							['clause', 'Root node for each clause'],
							['literal', 'Positive or negative atom'],
							['predicate', 'Predicate symbol (e.g., P, Q, =)'],
							['function', 'Function application (e.g., f(x))'],
							['variable', 'Logic variable (e.g., X, Y)'],
							['constant', 'Constant symbol (e.g., a, b, e)'],
						] as [type, desc]}
							<tr class="border-b border-card-border/50">
								<td class="py-1.5 pr-4"><code class="bg-surface px-1.5 py-0.5 rounded text-xs">{type}</code></td>
								<td class="py-1.5 text-text-muted">{desc}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<h3 class="font-display font-semibold text-lg mb-2">Example Graph</h3>
			<div class="bg-surface-light border border-card-border rounded-lg p-4 mb-6">
				<p class="text-sm font-semibold mb-2">Graph for <code class="bg-surface px-1.5 py-0.5 rounded text-xs">P(X, a) &or; &not;Q(f(Y))</code></p>
				<pre class="font-mono text-sm leading-relaxed text-text-muted">clause
├── literal (polarity=+)
│   └── predicate "P"
│       ├── variable "X"
│       └── constant "a"
└── literal (polarity=-)
    └── predicate "Q"
        └── function "f"
            └── variable "Y"</pre>
			</div>

			<h3 class="font-display font-semibold text-lg mb-2">Model Architectures</h3>
			<div class="space-y-4 mb-6">
				<div class="bg-surface-light border border-card-border rounded-lg p-4">
					<h4 class="font-semibold mb-2">Graph Convolutional Network (GCN)</h4>
					<pre class="font-mono text-xs bg-surface rounded p-3 mb-2 text-text-muted">h' = LayerNorm(ReLU(A · h · W))   # Per layer
clause_emb = pool(h, clause_mask)  # Pool node embeddings
score = MLP(clause_emb)            # Score each clause</pre>
					<p class="text-xs text-text-muted">hidden_dim=64, num_layers=3, dropout=0.1</p>
				</div>

				<div class="bg-surface-light border border-card-border rounded-lg p-4">
					<h4 class="font-semibold mb-2">Multi-Layer Perceptron (MLP)</h4>
					<pre class="font-mono text-xs bg-surface rounded p-3 mb-2 text-text-muted">clause_emb = pool(node_features, clause_mask)
score = MLP(clause_emb)</pre>
					<p class="text-xs text-text-muted">Faster inference, useful for understanding the value of graph structure.</p>
				</div>

				<div class="bg-surface-light border border-card-border rounded-lg p-4">
					<h4 class="font-semibold mb-2">Graph Attention Network (GAT)</h4>
					<pre class="font-mono text-xs bg-surface rounded p-3 mb-2 text-text-muted">alpha_ij = softmax(LeakyReLU(a · [Wh_i || Wh_j]))
h'_i = sum_j alpha_ij · Wh_j</pre>
					<p class="text-xs text-text-muted">hidden_dim=64, num_layers=3, num_heads=4</p>
				</div>
			</div>

			<h3 class="font-display font-semibold text-lg mb-2">Quick Start</h3>
			<pre class="font-mono text-sm bg-gray-800 text-green-300 rounded-lg p-4 overflow-x-auto"># 1. Train a GCN model (collects traces if needed)
proofatlas-bench --config gcn_mlp --retrain

# 2. Evaluate the trained model
proofatlas-bench --config gcn_mlp

# 3. View training metrics
# Open training page in browser</pre>
		</section>

		<section id="examples">
			<h2 class="font-display text-2xl font-bold mb-4 text-text">Examples</h2>

			<div class="space-y-6">
				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-semibold mb-2">Example 1: Simple Propositional Logic</h3>
					<pre class="font-mono text-sm bg-surface rounded p-3 mb-2 overflow-x-auto text-text-muted">% If P implies Q, and Q implies R, then P implies R
fof(axiom1, axiom, p => q).
fof(axiom2, axiom, q => r).
fof(goal, conjecture, p => r).</pre>
					<p class="text-sm text-green-400"><strong>Result:</strong> Proof found using binary resolution</p>
				</div>

				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-semibold mb-2">Example 2: Equality Reasoning</h3>
					<pre class="font-mono text-sm bg-surface rounded p-3 mb-2 overflow-x-auto text-text-muted">% Symmetry and transitivity
fof(symmetry, axiom, ![X,Y]: (X = Y => Y = X)).
fof(transitivity, axiom, ![X,Y,Z]: ((X = Y & Y = Z) => X = Z)).
fof(goal, conjecture, ![A,B,C]: ((A = B & C = B) => A = C)).</pre>
					<p class="text-sm text-green-400"><strong>Result:</strong> Requires superposition for equality reasoning</p>
				</div>

				<div class="bg-surface-light border border-card-border rounded-lg p-5">
					<h3 class="font-semibold mb-2">Example 3: Group Theory</h3>
					<pre class="font-mono text-sm bg-surface rounded p-3 mb-2 overflow-x-auto text-text-muted">fof(left_identity, axiom, ![X]: mult(e, X) = X).
fof(left_inverse, axiom, ![X]: mult(inv(X), X) = e).
fof(associativity, axiom, ![X,Y,Z]:
    mult(X, mult(Y, Z)) = mult(mult(X, Y), Z)).
fof(right_identity, conjecture, ![X]: mult(X, e) = X).</pre>
					<p class="text-sm text-green-400"><strong>Result:</strong> Classic group theory proof using superposition and demodulation</p>
				</div>
			</div>
		</section>

		<section id="references">
			<h2 class="font-display text-2xl font-bold mb-4 text-text">References</h2>
			<ul class="list-disc list-inside space-y-2 text-sm text-text-muted">
				<li>Bachmair, L., & Ganzinger, H. (2001). <em>Resolution Theorem Proving</em>. Handbook of Automated Reasoning.</li>
				<li>Nieuwenhuis, R., & Rubio, A. (2001). <em>Paramodulation-Based Theorem Proving</em>. Handbook of Automated Reasoning.</li>
				<li>Schulz, S. (2013). <em>System Description: E 1.8</em>. LPAR 2013.</li>
			</ul>
		</section>
	</div>
</div>
