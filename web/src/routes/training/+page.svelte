<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';

	interface EpochData {
		epoch: number;
		train_loss: number;
		val_loss: number;
		val_acc: number;
		learning_rate: number;
	}

	interface EvalData {
		epoch: number;
		success_rate: number;
		num_success: number;
		num_problems: number;
		avg_time: number | null;
	}

	interface RunData {
		name: string;
		model: Record<string, unknown>;
		training: Record<string, unknown>;
		epochs: EpochData[];
		evaluations: EvalData[];
		best_epoch: number | null;
		best_val_loss: number | null;
		total_time_seconds: number | null;
		termination_reason: string | null;
	}

	let allRuns: Record<string, RunData> = $state({});
	let selectedRun = $state('');
	let loading = $state(true);
	let noData = $state(false);
	let showComparison = $state(false);

	let Chart: any = $state(null);
	let charts: Record<string, any> = {};

	onMount(async () => {
		const mod = await import('chart.js/auto');
		Chart = mod.default;
		await loadRuns();
	});

	async function loadRuns() {
		try {
			const cb = `?t=${Date.now()}`;
			const resp = await fetch(`${base}/data/training/index.json${cb}`);
			if (!resp.ok) { noData = true; loading = false; return; }
			const index = await resp.json();
			const names: string[] = index.runs || [];
			if (names.length === 0) { noData = true; loading = false; return; }

			const loaded: Record<string, RunData> = {};
			for (const name of names) {
				try {
					const r = await fetch(`${base}/data/training/${name}.json${cb}`);
					if (r.ok) loaded[name] = await r.json();
				} catch { /* skip */ }
			}
			if (Object.keys(loaded).length === 0) { noData = true; loading = false; return; }

			allRuns = loaded;
			const first = Object.keys(loaded).sort().reverse()[0];
			selectedRun = first;
			loading = false;

			// Wait for DOM
			await new Promise(r => requestAnimationFrame(r));
			renderCharts();
		} catch {
			noData = true; loading = false;
		}
	}

	function renderCharts() {
		if (!Chart || !selectedRun || !allRuns[selectedRun]) return;
		const run = allRuns[selectedRun];
		if (!run.epochs) return;

		const labels = run.epochs.map(e => e.epoch);
		const trainLoss = run.epochs.map(e => e.train_loss);
		const valLoss = run.epochs.map(e => e.val_loss);
		const valAcc = run.epochs.map(e => e.val_acc);
		const lr = run.epochs.map(e => e.learning_rate);

		const commonOpts = {
			responsive: true,
			maintainAspectRatio: false,
			plugins: { legend: { position: 'top' as const } },
		};

		// lammdachs colors
		const blue = '#456878';
		const berry = '#9E2D39';
		const green = '#4A6444';
		const honey = '#B58C18';

		// Destroy old charts
		Object.values(charts).forEach((c: any) => c?.destroy?.());
		charts = {};

		const lossEl = document.getElementById('loss-chart') as HTMLCanvasElement;
		if (lossEl) {
			charts.loss = new Chart(lossEl, {
				type: 'line',
				data: {
					labels,
					datasets: [
						{ label: 'Train Loss', data: trainLoss, borderColor: blue, backgroundColor: blue + '1a', fill: true, tension: 0.1 },
						{ label: 'Val Loss', data: valLoss, borderColor: berry, backgroundColor: berry + '1a', fill: true, tension: 0.1 },
					]
				},
				options: { ...commonOpts, scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Loss' } } } }
			});
		}

		const accEl = document.getElementById('acc-chart') as HTMLCanvasElement;
		if (accEl) {
			charts.acc = new Chart(accEl, {
				type: 'line',
				data: {
					labels,
					datasets: [
						{ label: 'Validation Accuracy', data: valAcc, borderColor: green, backgroundColor: green + '1a', fill: true, tension: 0.1 },
					]
				},
				options: { ...commonOpts, scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Accuracy' }, min: 0, max: 1 } } }
			});
		}

		const lrEl = document.getElementById('lr-chart') as HTMLCanvasElement;
		if (lrEl) {
			charts.lr = new Chart(lrEl, {
				type: 'line',
				data: {
					labels,
					datasets: [
						{ label: 'Learning Rate', data: lr, borderColor: honey, backgroundColor: honey + '1a', fill: true, tension: 0.1 },
					]
				},
				options: { ...commonOpts, scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'LR' }, type: 'logarithmic' } } }
			});
		}

		const evals = run.evaluations || [];
		const proofEl = document.getElementById('proof-chart') as HTMLCanvasElement;
		if (proofEl && evals.length > 0) {
			charts.proof = new Chart(proofEl, {
				type: 'line',
				data: {
					labels: evals.map(e => e.epoch),
					datasets: [
						{ label: 'Proof Success Rate', data: evals.map(e => e.success_rate), borderColor: honey, backgroundColor: honey + '1a', fill: true, tension: 0.1 },
					]
				},
				options: { ...commonOpts, scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Success Rate' }, min: 0, max: 1 } } }
			});
		}
	}

	function handleRunChange(e: Event) {
		selectedRun = (e.target as HTMLSelectElement).value;
		requestAnimationFrame(() => renderCharts());
	}

	function formatTime(seconds: number | null | undefined): string {
		if (!seconds) return 'N/A';
		if (seconds < 60) return `${seconds.toFixed(0)}s`;
		if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
		return `${(seconds / 3600).toFixed(1)}h`;
	}

	function compareRuns() {
		if (Object.keys(allRuns).length < 2) return;
		showComparison = true;
		requestAnimationFrame(() => {
			const el = document.getElementById('compare-chart') as HTMLCanvasElement;
			if (!el || !Chart) return;
			charts.compare?.destroy?.();

			const colors = ['#456878', '#9E2D39', '#4A6444', '#B58C18', '#A76647', '#4F443F'];
			const datasets = Object.entries(allRuns).map(([name, run], i) => {
				const label = run.model?.type
					? `${run.model.type} (${run.model.hidden_dim}h x ${run.model.num_layers}L)`
					: name;
				return {
					label,
					data: (run.epochs || []).map(e => ({ x: e.epoch, y: e.val_loss })),
					borderColor: colors[i % colors.length],
					fill: false,
					tension: 0.1,
				};
			});

			charts.compare = new Chart(el, {
				type: 'line',
				data: { datasets },
				options: {
					responsive: true,
					maintainAspectRatio: false,
					plugins: { legend: { position: 'top' as const }, title: { display: true, text: 'Validation Loss Comparison' } },
					scales: { x: { type: 'linear', title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Val Loss' } } }
				}
			});
		});
	}

	let currentRunData = $derived(allRuns[selectedRun] || null);
</script>

<svelte:head>
	<title>ProofAtlas - Training</title>
</svelte:head>

<div class="space-y-6">
	<h2 class="font-display text-2xl font-bold text-text">Training</h2>
	<p class="text-text-muted text-sm">Interactive visualization of model training runs</p>

	{#if loading}
		<p class="text-center py-10 text-text-muted">Loading training data...</p>
	{:else if noData}
		<div class="text-center py-10 text-text-muted">
			<h3 class="font-semibold text-lg mb-2">No Training Data Available</h3>
			<p>Training results will appear here after running experiments.</p>
		</div>
	{:else}
		<!-- Run selector -->
		<div class="flex flex-wrap gap-3 items-center">
			<label class="text-sm font-semibold text-text-muted" for="run-select">Select Run:</label>
			<select
				id="run-select"
				class="px-3 py-2 bg-surface-light border border-card-border rounded text-sm text-text min-w-64 focus:outline-none focus:border-accent"
				value={selectedRun}
				onchange={handleRunChange}
			>
				{#each Object.keys(allRuns).sort().reverse() as name}
					<option value={name}>{name}</option>
				{/each}
			</select>
			<button
				class="px-3 py-2 bg-blue-400 hover:bg-blue-300 text-gray-900 rounded text-sm font-semibold transition-colors"
				onclick={compareRuns}
			>Compare Runs</button>
		</div>

		{#if currentRunData}
			<!-- Run info -->
			<div class="bg-surface-light rounded border border-card-border p-5">
				<h3 class="font-display font-semibold text-lg text-text mb-3">{currentRunData.name || selectedRun}</h3>
				<div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
					<div class="p-3 bg-surface rounded border border-card-border">
						<div class="text-xs text-text-muted uppercase">Model Type</div>
						<div class="text-sm font-semibold text-text">{currentRunData.model?.type || 'N/A'}</div>
					</div>
					<div class="p-3 bg-surface rounded border border-card-border">
						<div class="text-xs text-text-muted uppercase">Architecture</div>
						<div class="text-sm font-semibold text-text">{currentRunData.model?.hidden_dim || '?'}h x {currentRunData.model?.num_layers || '?'}L</div>
					</div>
					<div class="p-3 bg-surface rounded border border-card-border">
						<div class="text-xs text-text-muted uppercase">Best Epoch</div>
						<div class="text-sm font-semibold text-text">{currentRunData.best_epoch ?? 'N/A'}</div>
					</div>
					<div class="p-3 bg-surface rounded border border-card-border">
						<div class="text-xs text-text-muted uppercase">Best Val Loss</div>
						<div class="text-sm font-semibold text-text">{currentRunData.best_val_loss?.toFixed(4) ?? 'N/A'}</div>
					</div>
					<div class="p-3 bg-surface rounded border border-card-border">
						<div class="text-xs text-text-muted uppercase">Training Time</div>
						<div class="text-sm font-semibold text-text">{formatTime(currentRunData.total_time_seconds)}</div>
					</div>
					<div class="p-3 bg-surface rounded border border-card-border">
						<div class="text-xs text-text-muted uppercase">Epochs</div>
						<div class="text-sm font-semibold text-text">{currentRunData.epochs?.length ?? 0}</div>
					</div>
					<div class="p-3 bg-surface rounded border border-card-border">
						<div class="text-xs text-text-muted uppercase">Termination</div>
						<div class="text-sm font-semibold text-text">{currentRunData.termination_reason || 'N/A'}</div>
					</div>
				</div>

				<details class="mt-3">
					<summary class="cursor-pointer text-xs text-text-muted hover:text-text select-none">Configuration</summary>
					<div class="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-2 text-xs">
						{#each [...Object.entries(currentRunData.model || {}), ...Object.entries(currentRunData.training || {})] as [key, val]}
							{#if val != null}
								<div><span class="text-text-muted">{key}:</span> <span class="font-semibold text-text">{val}</span></div>
							{/if}
						{/each}
					</div>
				</details>
			</div>

			<!-- Charts -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
				<div class="bg-surface-light rounded border border-card-border p-4">
					<h3 class="font-semibold text-sm text-text mb-2">Training & Validation Loss</h3>
					<div class="relative h-72"><canvas id="loss-chart"></canvas></div>
				</div>
				<div class="bg-surface-light rounded border border-card-border p-4">
					<h3 class="font-semibold text-sm text-text mb-2">Validation Accuracy</h3>
					<div class="relative h-72"><canvas id="acc-chart"></canvas></div>
				</div>
				<div class="bg-surface-light rounded border border-card-border p-4">
					<h3 class="font-semibold text-sm text-text mb-2">Learning Rate</h3>
					<div class="relative h-72"><canvas id="lr-chart"></canvas></div>
				</div>
				{#if (currentRunData.evaluations || []).length > 0}
					<div class="bg-surface-light rounded border border-card-border p-4">
						<h3 class="font-semibold text-sm text-text mb-2">Proof Success Rate</h3>
						<div class="relative h-72"><canvas id="proof-chart"></canvas></div>
					</div>
				{/if}
			</div>

			<!-- Evaluation details table -->
			{#if (currentRunData.evaluations || []).length > 0}
				<div class="bg-surface-light rounded border border-card-border p-4">
					<h3 class="font-semibold text-sm text-text mb-3">Evaluation Details</h3>
					<table class="w-full text-sm border-collapse">
						<thead>
							<tr class="border-b border-card-border">
								<th class="text-left py-2 px-3 text-text-muted font-semibold">Epoch</th>
								<th class="text-left py-2 px-3 text-text-muted font-semibold">Success Rate</th>
								<th class="text-left py-2 px-3 text-text-muted font-semibold">Problems Solved</th>
								<th class="text-left py-2 px-3 text-text-muted font-semibold">Avg Time</th>
							</tr>
						</thead>
						<tbody>
							{#each currentRunData.evaluations as ev}
								<tr class="border-b border-card-border/50">
									<td class="py-1.5 px-3 text-text">{ev.epoch}</td>
									<td class="py-1.5 px-3 text-text">{(ev.success_rate * 100).toFixed(1)}%</td>
									<td class="py-1.5 px-3 text-text-muted">{ev.num_success} / {ev.num_problems}</td>
									<td class="py-1.5 px-3 text-text-muted font-mono">{ev.avg_time?.toFixed(2) ?? 'N/A'}s</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		{/if}

		<!-- Comparison -->
		{#if showComparison}
			<div class="bg-surface-light rounded border border-card-border p-4">
				<h3 class="font-semibold text-sm text-text mb-2">Run Comparison</h3>
				<div class="relative h-96"><canvas id="compare-chart"></canvas></div>
			</div>
		{/if}
	{/if}
</div>
