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
	let filterPresets: Set<string> = $state(new Set());

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
		Object.values(allRuns).filter(r => filterPresets.has(r.preset))
	);

	let allPresets = $derived([...new Set(Object.values(allRuns).map(r => r.preset))].sort());
	let totalRuns = $derived(Object.values(allRuns).length);
	let totalProblems = $derived(
		Object.values(allRuns).reduce((m, r) => Math.max(m, r.total || 0), 0)
	);
	let bestRun = $derived(
		Object.values(allRuns).reduce<RunData | null>(
			(a, b) => !a || (b.proof_rate || 0) > (a.proof_rate || 0) ? b : a,
			null
		)
	);

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
</script>

<svelte:head>
	<title>ProofAtlas - Results</title>
</svelte:head>

<div class="space-y-6">
	<h2 class="font-display text-2xl font-bold text-text">Results</h2>
	<p class="text-text-muted text-sm">Per-configuration benchmark results across the TPTP slice</p>

	{#if loading}
		<p class="text-center py-10 text-text-muted">Loading benchmark data...</p>
	{:else if noData}
		<div class="text-center py-10 text-text-muted">
			<h3 class="font-semibold text-lg mb-2">No Benchmark Data Available</h3>
			<p class="mb-3">Run benchmarks and export results:</p>
			<pre class="inline-block text-left font-mono text-sm bg-surface-light p-4 rounded">proofatlas-bench --config &lt;name&gt;</pre>
		</div>
	{:else}
		<!-- Stats -->
		<div class="grid grid-cols-3 gap-4">
			<div class="p-4 bg-surface-light rounded border border-card-border">
				<div class="text-2xl font-bold text-text">{totalRuns}</div>
				<div class="text-xs text-text-muted uppercase">Configurations</div>
			</div>
			<div class="p-4 bg-surface-light rounded border border-card-border">
				<div class="text-2xl font-bold text-text">{totalProblems.toLocaleString()}</div>
				<div class="text-xs text-text-muted uppercase">Problems</div>
			</div>
			<div class="p-4 bg-surface-light rounded border border-card-border">
				<div class="text-2xl font-bold text-text">{bestRun?.proof_rate?.toFixed(1) || 0}%</div>
				<div class="text-xs text-text-muted uppercase">Best: {bestRun?.preset ?? '-'}</div>
			</div>
		</div>

		<!-- Preset filter -->
		<div class="bg-surface-light rounded border border-card-border p-4">
			<div class="flex flex-wrap gap-2 items-center">
				<span class="text-sm font-semibold text-text-muted">Preset:</span>
				{#each allPresets as p}
					<button
						class="px-3 py-1 rounded text-xs font-semibold border-2 transition-colors {filterPresets.has(p) ? 'bg-blue-400 border-blue-400 text-gray-900' : 'border-card-border text-text-muted'}"
						onclick={() => togglePreset(p)}
					>{p}</button>
				{/each}
			</div>
		</div>

		{#if filteredRuns.length === 0}
			<p class="text-center py-6 text-text-muted">No runs match the current filters.</p>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full text-sm border-collapse">
					<thead>
						<tr class="border-b border-card-border">
							<th class="text-left py-2 px-3 text-text-muted font-semibold">Preset</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Progress</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Proofs</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Saturated</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Resource limit</th>
							<th class="text-right py-2 px-3 text-text-muted font-semibold">Proof Rate</th>
						</tr>
					</thead>
					<tbody>
						{#each [...filteredRuns].sort((a, b) => (b.proof_rate || 0) - (a.proof_rate || 0)) as run}
							<tr class="border-b border-card-border/50 hover:bg-surface-light/50">
								<td class="py-2 px-3 text-text">{run.preset}</td>
								<td class="py-2 px-3 text-right font-mono text-text-muted">{run.completed}/{run.total}</td>
								<td class="py-2 px-3 text-right font-mono text-text">{run.stats?.proof || 0}</td>
								<td class="py-2 px-3 text-right font-mono text-text-muted">{run.stats?.saturated || 0}</td>
								<td class="py-2 px-3 text-right font-mono text-text-muted">{run.stats?.resource_limit || 0}</td>
								<td class="py-2 px-3 text-right font-mono font-semibold {proofRateClass(run.proof_rate || 0)}">{(run.proof_rate || 0).toFixed(1)}%</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	{/if}
</div>
