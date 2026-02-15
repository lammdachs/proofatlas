<script lang="ts">
	interface RunResult {
		problem: string;
		status: string;
		time_s: number | null;
	}

	interface RunConfig {
		results: RunResult[];
	}

	let runIndex: string[] = $state([]);
	let currentConfig: string | null = $state(null);
	let currentData: RunConfig | null = $state(null);
	let loading = $state(true);
	let noData = $state(false);
	let statusFilters: Set<string> = $state(new Set());
	let sortCol: 'problem' | 'status' | 'time_s' = $state('problem');
	let sortAsc = $state(true);

	$effect(() => {
		loadIndex();
	});

	async function loadIndex() {
		try {
			const resp = await fetch('/data/runs/index.json');
			if (!resp.ok) { noData = true; loading = false; return; }
			const index = await resp.json();
			runIndex = index.runs || [];
			if (runIndex.length === 0) { noData = true; loading = false; return; }

			const hash = location.hash.slice(1);
			const initial = runIndex.includes(hash) ? hash : runIndex[0];
			await loadConfig(initial);
		} catch {
			noData = true; loading = false;
		}
	}

	async function loadConfig(name: string) {
		currentConfig = name;
		location.hash = name;

		try {
			const resp = await fetch(`/data/runs/${name}.json`);
			if (!resp.ok) return;
			currentData = await resp.json();

			const statuses = [...new Set(currentData!.results.map(r => r.status))].sort();
			statusFilters = new Set(statuses);
			loading = false;
		} catch { /* skip */ }
	}

	let statusCounts = $derived(() => {
		if (!currentData) return {};
		const counts: Record<string, number> = {};
		for (const r of currentData.results) {
			counts[r.status] = (counts[r.status] || 0) + 1;
		}
		return counts;
	});

	let allStatuses = $derived(currentData ? [...new Set(currentData.results.map(r => r.status))].sort() : []);

	let filteredRows = $derived(() => {
		if (!currentData) return [];
		let rows = currentData.results.filter(r => statusFilters.has(r.status));
		rows.sort((a, b) => {
			if (sortCol === 'time_s') {
				const va = a.time_s ?? 0, vb = b.time_s ?? 0;
				return sortAsc ? va - vb : vb - va;
			}
			const va = String((a as any)[sortCol] || '');
			const vb = String((b as any)[sortCol] || '');
			return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
		});
		return rows;
	});

	function toggleStatus(s: string) {
		const next = new Set(statusFilters);
		if (next.has(s)) next.delete(s); else next.add(s);
		statusFilters = next;
	}

	function handleSort(col: typeof sortCol) {
		if (sortCol === col) { sortAsc = !sortAsc; }
		else { sortCol = col; sortAsc = true; }
	}

	function statusActiveClass(s: string): string {
		switch (s) {
			case 'proof': return 'bg-green-400 border-green-400 text-gray-900';
			case 'saturated': return 'bg-blue-400 border-blue-400 text-gray-900';
			case 'resource_limit': return 'bg-honey-400 border-honey-400 text-gray-900';
			case 'error': return 'bg-berry-400 border-berry-400 text-gray-900';
			default: return 'bg-gray-400 border-gray-400 text-gray-900';
		}
	}

	function statusBadgeClass(s: string): string {
		switch (s) {
			case 'proof': return 'bg-green-800/30 text-green-300';
			case 'saturated': return 'bg-blue-800/30 text-blue-300';
			case 'resource_limit': return 'bg-honey-800/30 text-honey-300';
			case 'error': return 'bg-berry-800/30 text-berry-300';
			default: return 'bg-gray-800/30 text-gray-300';
		}
	}
</script>

<svelte:head>
	<title>ProofAtlas - Runs</title>
</svelte:head>

<div class="space-y-6">
	<h2 class="font-display text-2xl font-bold text-text">Runs</h2>
	<p class="text-text-muted text-sm">Per-problem benchmark results</p>

	{#if loading}
		<p class="text-center py-10 text-text-muted">Loading run data...</p>
	{:else if noData}
		<div class="text-center py-10 text-text-muted">
			<h3 class="font-semibold text-lg mb-2">No Run Data Available</h3>
			<p class="mb-3">Run benchmarks and sync results:</p>
			<pre class="inline-block text-left font-mono text-sm bg-surface-light p-4 rounded">proofatlas-bench --config age_weight</pre>
		</div>
	{:else}
		<!-- Stats -->
		{@const counts = statusCounts()}
		<div class="grid grid-cols-2 sm:grid-cols-5 gap-3">
			<div class="p-3 bg-surface-light rounded border border-card-border">
				<div class="text-xl font-bold text-text">{currentData?.results.length.toLocaleString()}</div>
				<div class="text-xs text-text-muted uppercase">Attempted</div>
			</div>
			<div class="p-3 bg-surface-light rounded border border-card-border">
				<div class="text-xl font-bold text-green-400">{(counts.proof || 0).toLocaleString()}</div>
				<div class="text-xs text-text-muted uppercase">Proof</div>
			</div>
			<div class="p-3 bg-surface-light rounded border border-card-border">
				<div class="text-xl font-bold text-blue-400">{(counts.saturated || 0).toLocaleString()}</div>
				<div class="text-xs text-text-muted uppercase">Saturated</div>
			</div>
			<div class="p-3 bg-surface-light rounded border border-card-border">
				<div class="text-xl font-bold text-honey-400">{(counts.resource_limit || 0).toLocaleString()}</div>
				<div class="text-xs text-text-muted uppercase">Resource Limit</div>
			</div>
			<div class="p-3 bg-surface-light rounded border border-card-border">
				<div class="text-xl font-bold text-berry-400">{(counts.error || 0).toLocaleString()}</div>
				<div class="text-xs text-text-muted uppercase">Error</div>
			</div>
		</div>

		<!-- Config + Status Filters -->
		<div class="bg-surface-light rounded border border-card-border p-4 space-y-3">
			<div class="flex flex-wrap gap-2 items-center">
				<span class="text-sm font-semibold text-text-muted">Config:</span>
				{#each runIndex as name}
					<button
						class="px-3 py-1 rounded text-xs font-semibold border-2 transition-colors {currentConfig === name ? 'bg-blue-400 border-blue-400 text-gray-900' : 'border-card-border text-text-muted'}"
						onclick={() => loadConfig(name)}
					>{name}</button>
				{/each}
			</div>
			<div class="flex flex-wrap gap-2 items-center">
				<span class="text-sm font-semibold text-text-muted">Status:</span>
				{#each allStatuses as s}
					<button
						class="px-3 py-1 rounded text-xs font-semibold border-2 transition-colors {statusFilters.has(s) ? statusActiveClass(s) : 'border-card-border text-text-muted'}"
						onclick={() => toggleStatus(s)}
					>{s.replace('_', ' ')}</button>
				{/each}
			</div>
		</div>

		<!-- Table -->
		{@const rows = filteredRows()}
		<div class="text-xs text-text-muted mb-1">
			Showing {rows.length} of {currentData?.results.length} problems
		</div>
		<div class="overflow-x-auto">
			<table class="w-full text-sm border-collapse">
				<thead>
					<tr class="border-b border-card-border">
						<th class="text-left py-2 px-3 text-text-muted font-semibold cursor-pointer select-none hover:text-text" onclick={() => handleSort('problem')}>
							Problem {sortCol === 'problem' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
						</th>
						<th class="text-left py-2 px-3 text-text-muted font-semibold cursor-pointer select-none hover:text-text" onclick={() => handleSort('status')}>
							Status {sortCol === 'status' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
						</th>
						<th class="text-right py-2 px-3 text-text-muted font-semibold cursor-pointer select-none hover:text-text" onclick={() => handleSort('time_s')}>
							Time {sortCol === 'time_s' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
						</th>
					</tr>
				</thead>
				<tbody>
					{#each rows as r}
						<tr class="border-b border-card-border/50 hover:bg-surface-light/50">
							<td class="py-1.5 px-3 text-text">{r.problem}</td>
							<td class="py-1.5 px-3">
								<span class="inline-block px-2 py-0.5 rounded text-xs font-semibold {statusBadgeClass(r.status)}">
									{r.status.replace('_', ' ')}
								</span>
							</td>
							<td class="py-1.5 px-3 text-right font-mono text-text-muted">
								{r.time_s != null ? r.time_s.toFixed(3) + 's' : '-'}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
