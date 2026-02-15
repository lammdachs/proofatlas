<script lang="ts">
	import { base } from '$app/paths';

	interface RunData {
		prover: string;
		preset: string;
		proof_rate: number;
		completed: number;
		total: number;
		stats: Record<string, number>;
	}

	let allRuns: Record<string, RunData> = $state({});
	let loading = $state(true);
	let noData = $state(false);
	let filterProvers = $state(new Set(['proofatlas', 'vampire', 'spass']));
	let filterPresets: Set<string> = $state(new Set());
	let viewMode: 'comparison' | 'table' = $state('comparison');

	$effect(() => {
		loadData();
	});

	async function loadData() {
		try {
			const resp = await fetch(`${base}/data/benchmarks/index.json`);
			if (!resp.ok) { noData = true; loading = false; return; }
			const index = await resp.json();
			const runNames: string[] = index.runs || [];
			if (runNames.length === 0) { noData = true; loading = false; return; }

			const loaded: Record<string, RunData> = {};
			for (const name of runNames) {
				try {
					const r = await fetch(`${base}/data/benchmarks/${name}.json`);
					if (r.ok) loaded[name] = await r.json();
				} catch { /* skip */ }
			}
			if (Object.keys(loaded).length === 0) { noData = true; loading = false; return; }

			allRuns = loaded;
			filterPresets = new Set(Object.values(loaded).map(r => r.preset));
			loading = false;
		} catch {
			noData = true; loading = false;
		}
	}

	let filteredRuns = $derived(
		Object.values(allRuns).filter(r =>
			filterProvers.has(r.prover) && filterPresets.has(r.preset)
		)
	);

	let allPresets = $derived([...new Set(Object.values(allRuns).map(r => r.preset))].sort());
	let totalRuns = $derived(Object.values(allRuns).length);
	let totalProofs = $derived(Object.values(allRuns).reduce((s, r) => s + (r.stats?.proof || 0), 0));
	let bestRun = $derived(
		Object.values(allRuns).reduce<RunData | null>(
			(a, b) => !a || (b.proof_rate || 0) > (a.proof_rate || 0) ? b : a,
			null
		)
	);

	// Comparison view helpers
	let comparisonProvers = $derived(
		['proofatlas', 'vampire', 'spass'].filter(p =>
			filteredRuns.some(r => r.prover === p)
		)
	);

	let presetGroups = $derived(() => {
		const groups: Record<string, Record<string, RunData>> = {};
		for (const r of filteredRuns) {
			if (!groups[r.preset]) groups[r.preset] = {};
			groups[r.preset][r.prover] = r;
		}
		return groups;
	});

	function toggleProver(p: string) {
		const next = new Set(filterProvers);
		if (next.has(p)) next.delete(p); else next.add(p);
		filterProvers = next;
	}

	function togglePreset(p: string) {
		const next = new Set(filterPresets);
		if (next.has(p)) next.delete(p); else next.add(p);
		filterPresets = next;
	}

	function proofRateClass(rate: number): string {
		if (rate >= 70) return 'text-green-400';
		if (rate >= 40) return 'text-honey-400';
		return 'text-berry-400';
	}

	function proverActiveClass(p: string): string {
		switch (p) {
			case 'proofatlas': return 'bg-berry-400 border-berry-400 text-gray-900';
			case 'vampire': return 'bg-terracotta-400 border-terracotta-400 text-gray-900';
			case 'spass': return 'bg-green-400 border-green-400 text-gray-900';
			default: return 'bg-gray-400 border-gray-400 text-gray-900';
		}
	}

	function proverTextClass(p: string): string {
		switch (p) {
			case 'proofatlas': return 'text-berry-400';
			case 'vampire': return 'text-terracotta-400';
			case 'spass': return 'text-green-400';
			default: return 'text-gray-400';
		}
	}

	function proverBadgeClass(p: string): string {
		switch (p) {
			case 'proofatlas': return 'bg-berry-400 text-gray-900';
			case 'vampire': return 'bg-terracotta-400 text-gray-900';
			case 'spass': return 'bg-green-400 text-gray-900';
			default: return 'bg-gray-400 text-gray-900';
		}
	}
</script>

<svelte:head>
	<title>ProofAtlas - Results</title>
</svelte:head>

<div class="space-y-6">
	<h2 class="font-display text-2xl font-bold text-text">Results</h2>
	<p class="text-text-muted text-sm">Comparing theorem provers across TPTP problems</p>

	{#if loading}
		<p class="text-center py-10 text-text-muted">Loading benchmark data...</p>
	{:else if noData}
		<div class="text-center py-10 text-text-muted">
			<h3 class="font-semibold text-lg mb-2">No Benchmark Data Available</h3>
			<p class="mb-3">Run benchmarks and export results:</p>
			<pre class="inline-block text-left font-mono text-sm bg-surface-light p-4 rounded">python scripts/bench.py --prover proofatlas vampire spass
python scripts/export.py --benchmarks</pre>
		</div>
	{:else}
		<!-- Stats -->
		<div class="grid grid-cols-3 gap-4">
			<div class="p-4 bg-surface-light rounded border border-card-border">
				<div class="text-2xl font-bold text-text">{totalRuns}</div>
				<div class="text-xs text-text-muted uppercase">Configurations</div>
			</div>
			<div class="p-4 bg-surface-light rounded border border-card-border">
				<div class="text-2xl font-bold text-text">{totalProofs.toLocaleString()}</div>
				<div class="text-xs text-text-muted uppercase">Total Proofs</div>
			</div>
			<div class="p-4 bg-surface-light rounded border border-card-border">
				<div class="text-2xl font-bold text-text">{bestRun?.proof_rate?.toFixed(1) || 0}%</div>
				<div class="text-xs text-text-muted uppercase">Best: {bestRun?.prover}/{bestRun?.preset}</div>
			</div>
		</div>

		<!-- Filters -->
		<div class="bg-surface-light rounded border border-card-border p-4 space-y-3">
			<div class="flex flex-wrap gap-3 items-center">
				<span class="text-sm font-semibold text-text-muted">Prover:</span>
				{#each ['proofatlas', 'vampire', 'spass'] as p}
					<button
						class="px-3 py-1 rounded text-xs font-semibold border-2 transition-colors {filterProvers.has(p) ? proverActiveClass(p) : 'border-card-border text-text-muted'}"
						onclick={() => toggleProver(p)}
					>{p}</button>
				{/each}
			</div>
			<div class="flex flex-wrap gap-3 items-center">
				<span class="text-sm font-semibold text-text-muted">Preset:</span>
				{#each allPresets as p}
					<button
						class="px-3 py-1 rounded text-xs font-semibold border-2 transition-colors {filterPresets.has(p) ? 'bg-blue-400 border-blue-400 text-gray-900' : 'border-card-border text-text-muted'}"
						onclick={() => togglePreset(p)}
					>{p}</button>
				{/each}
			</div>
			<div class="flex flex-wrap gap-3 items-center">
				<span class="text-sm font-semibold text-text-muted">View:</span>
				<button
					class="px-3 py-1 rounded text-xs font-semibold border-2 transition-colors {viewMode === 'comparison' ? 'bg-blue-400 border-blue-400 text-gray-900' : 'border-card-border text-text-muted'}"
					onclick={() => viewMode = 'comparison'}
				>Comparison</button>
				<button
					class="px-3 py-1 rounded text-xs font-semibold border-2 transition-colors {viewMode === 'table' ? 'bg-blue-400 border-blue-400 text-gray-900' : 'border-card-border text-text-muted'}"
					onclick={() => viewMode = 'table'}
				>Table</button>
			</div>
		</div>

		{#if viewMode === 'comparison'}
			{@const groups = presetGroups()}
			{@const presets = Object.keys(groups).sort()}
			{#if filteredRuns.length === 0}
				<p class="text-center py-6 text-text-muted">No runs match the current filters.</p>
			{:else}
				<div class="overflow-x-auto">
					<table class="w-full text-sm border-collapse">
						<thead>
							<tr class="border-b border-card-border">
								<th class="text-left py-2 px-3 text-text-muted font-semibold">Preset</th>
								{#each comparisonProvers as p}
									<th class="text-center py-2 px-3 font-semibold {proverTextClass(p)}">{p}</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each presets as preset}
								{@const group = groups[preset]}
								{@const rates = comparisonProvers.map(p => group[p]?.proof_rate || 0)}
								{@const maxRate = Math.max(...rates)}
								<tr class="border-b border-card-border/50 hover:bg-surface-light/50">
									<td class="py-2 px-3 font-semibold text-text">{preset}</td>
									{#each comparisonProvers as p}
										{@const run = group[p]}
										{#if run}
											{@const rate = run.proof_rate || 0}
											<td class="py-2 px-3 text-center font-mono {rate === maxRate && rate > 0 ? 'font-bold text-green-400' : 'text-text-muted'}" title="{run.stats?.proof || 0}/{run.completed || 0} proofs">
												{rate.toFixed(1)}%
											</td>
										{:else}
											<td class="py-2 px-3 text-center text-text-muted italic">-</td>
										{/if}
									{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full text-sm border-collapse">
					<thead>
						<tr class="border-b border-card-border">
							<th class="text-left py-2 px-3 text-text-muted font-semibold">Prover</th>
							<th class="text-left py-2 px-3 text-text-muted font-semibold">Preset</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Progress</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Proofs</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Saturated</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Timeout</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Proof Rate</th>
						</tr>
					</thead>
					<tbody>
						{#each [...filteredRuns].sort((a, b) => (b.proof_rate || 0) - (a.proof_rate || 0)) as run}
							<tr class="border-b border-card-border/50 hover:bg-surface-light/50">
								<td class="py-2 px-3"><span class="px-2 py-0.5 rounded text-xs font-semibold {proverBadgeClass(run.prover)}">{run.prover}</span></td>
								<td class="py-2 px-3 text-text">{run.preset}</td>
								<td class="py-2 px-3 text-right font-mono text-text-muted">{run.completed}/{run.total}</td>
								<td class="py-2 px-3 text-right font-mono text-text">{run.stats?.proof || 0}</td>
								<td class="py-2 px-3 text-right font-mono text-text-muted">{run.stats?.saturated || 0}</td>
								<td class="py-2 px-3 text-right font-mono text-text-muted">{run.stats?.timeout || 0}</td>
								<td class="py-2 px-3 text-right font-mono font-semibold {proofRateClass(run.proof_rate || 0)}">{(run.proof_rate || 0).toFixed(1)}%</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	{/if}
</div>
