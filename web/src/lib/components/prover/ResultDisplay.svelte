<script lang="ts">
	import type { ProveResult, ProofStep } from '$lib/prover';
	import ProofInspector from './ProofInspector.svelte';

	let { result } = $props<{ result: ProveResult | null }>();

	let viewMode: 'proof' | 'inspector' = $state('proof');

	function statusColor(result: ProveResult): string {
		if (result.success) return 'var(--color-status-success)';
		if (result.status === 'timeout' || result.status === 'resource_limit') return 'var(--color-status-timeout)';
		return 'var(--color-status-error)';
	}

	function formatDuration(secs: number | undefined | null): string {
		if (secs === undefined || secs === null) return '-';
		if (secs < 0.001) return `${(secs * 1000000).toFixed(0)}us`;
		if (secs < 1) return `${(secs * 1000).toFixed(1)}ms`;
		return `${secs.toFixed(3)}s`;
	}

	function formatNum(n: number | undefined | null): string {
		return (n !== undefined && n !== null) ? n.toLocaleString() : '-';
	}

	type ProfileEntry = [string, string | number | undefined | null];

	function profileTable(rows: ProfileEntry[]): ProfileEntry[] {
		return rows.filter(([, v]) => v !== undefined && v !== null && v !== '-' && v !== '0' && v !== 0);
	}
</script>

{#if result}
	<div class="space-y-6">
		<!-- Status -->
		<div class="p-5 border-b-2" style="border-color: {statusColor(result)}; background: color-mix(in srgb, {statusColor(result)} 8%, transparent);">
			<p class="font-semibold" style="color: {statusColor(result)}">{result.message}</p>
		</div>

		<!-- Profile (expandable) -->
		{#if result.profile}
			{@const profile = result.profile as Record<string, any>}
			<details>
				<summary class="py-2 cursor-pointer text-xs uppercase tracking-wide text-green font-mono hover:text-text select-none transition-colors">
					// profiling details
				</summary>
				<div class="pt-2">
					<!-- Phase Timings -->
					<div>
						<h4 class="text-xs uppercase tracking-wide text-green font-mono" style="margin-top: 0.75rem; margin-bottom: 0.25rem;">// phase timings</h4>
						<table class="w-full text-sm">
							<tbody>
								{#each profileTable([
									['Total', formatDuration(profile.total_time)],
									['Init', formatDuration(profile.init_time)],
									['Process new', formatDuration(profile.process_new_time)],
									['Forward simplification', formatDuration(profile.forward_simplify_time)],
									['Backward simplification', formatDuration(profile.backward_simplify_time)],
									['Clause selection', formatDuration(profile.select_given_time)],
									['Inference generation', formatDuration(profile.generate_inferences_time)],
									['Inference addition', formatDuration(profile.add_inferences_time)],
								]) as [label, value]}
									<tr class="">
										<td class="py-1 text-text-muted">{label}</td>
										<td class="py-1 text-right font-mono text-text">{value}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>

					<!-- Generating Rules -->
					{#if profile.generating_rules}
						<div>
							<h4 class="text-xs uppercase tracking-wide text-green font-mono" style="margin-top: 0.75rem; margin-bottom: 0.25rem;">// generating inferences</h4>
							<table class="w-full text-sm">
								<tbody>
									{#each Object.entries(profile.generating_rules as Record<string, any>) as [name, stats]}
										<tr class="">
											<td class="py-1 text-text-muted">{name}</td>
											<td class="py-1 text-right font-mono text-text">{formatNum(stats.count)} in {formatDuration(stats.time)}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{/if}

					<!-- Clause Lifecycle -->
					<div>
						<h4 class="text-xs uppercase tracking-wide text-green font-mono" style="margin-top: 0.75rem; margin-bottom: 0.25rem;">// clause lifecycle</h4>
						<table class="w-full text-sm">
							<tbody>
								{#each profileTable([
									['Iterations', formatNum(profile.iterations)],
									['Clauses generated', formatNum(profile.clauses_generated)],
									['Clauses added', formatNum(profile.clauses_added)],
									['Max unprocessed', formatNum(profile.max_unprocessed_size)],
									['Max processed', formatNum(profile.max_processed_size)],
								]) as [label, value]}
									<tr class="">
										<td class="py-1 text-text-muted">{label}</td>
										<td class="py-1 text-right font-mono text-text">{value}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>

					<!-- Selector Stats -->
					{#if profile.selector_name}
						<div>
							<h4 class="text-xs uppercase tracking-wide text-green font-mono" style="margin-top: 0.75rem; margin-bottom: 0.25rem;">// selector</h4>
							<table class="w-full text-sm">
								<tbody>
									{#each profileTable([
										['Name', profile.selector_name],
										['Cache hits', formatNum(profile.selector_cache_hits)],
										['Cache misses', formatNum(profile.selector_cache_misses)],
										['Embed time', formatDuration(profile.selector_embed_time)],
										['Score time', formatDuration(profile.selector_score_time)],
									]) as [label, value]}
										<tr class="">
											<td class="py-1 text-text-muted">{label}</td>
											<td class="py-1 text-right font-mono text-text">{value}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{/if}
				</div>
			</details>
		{/if}

		<!-- Clauses / Inspector -->
		{#if (result.proof && result.proof.length > 0) || result.trace}
			<div>
				<!-- View toggle -->
				<div class="flex items-center border-b border-surface-lighter/50 mb-4">
					<button
						class="link-underline flex-1 py-4 text-base text-center transition-all cursor-pointer {viewMode === 'proof'
							? 'font-semibold'
							: 'text-text-muted hover:text-text'}"
						style={viewMode === 'proof' ? 'color: var(--color-green);' : ''}
						onclick={() => viewMode = 'proof'}
					>
						Proof
					</button>
					<div class="w-px h-6 bg-surface-lighter/50"></div>
					<button
						class="link-underline flex-1 py-4 text-base text-center transition-all cursor-pointer {viewMode === 'inspector'
							? 'font-semibold'
							: 'text-text-muted hover:text-text'}"
						style={viewMode === 'inspector' ? 'color: var(--color-green);' : ''}
						onclick={() => viewMode = 'inspector'}
					>
						Inspector
					</button>
				</div>

				{#if viewMode === 'proof'}
					{#if result.proof && result.proof.length > 0}
						<div class="overflow-hidden border border-surface-lighter/50 pb-4" style="background: color-mix(in srgb, var(--color-surface-light) 50%, transparent);">
							<table class="w-full">
								<thead>
									<tr class="border-b border-surface-lighter/50">
										<th class="px-4 py-3 text-left text-xs uppercase tracking-wide text-text-muted font-mono font-medium w-12">#</th>
										<th class="px-4 py-3 text-left text-xs uppercase tracking-wide text-text-muted font-mono font-medium">Clause</th>
										<th class="px-4 py-3 text-right text-xs uppercase tracking-wide text-text-muted font-mono font-medium">Rule</th>
									</tr>
								</thead>
								<tbody>
									{#each result.proof as step, i}
										<tr class="border-b border-surface-lighter/20 hover:bg-surface-lighter/30 transition-colors">
											<td class="px-4 py-2.5 font-mono text-xs text-text-muted">{step.id}</td>
											<td class="px-4 py-2.5 font-mono text-sm text-text">{step.clause}</td>
											<td class="px-4 py-2.5 text-right font-mono text-xs text-text-muted whitespace-nowrap">
												{step.rule}{#if step.parents.length > 0} ({step.parents.join(', ')}){/if}
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{:else}
						<p class="text-text-muted text-sm italic">No proof steps to display</p>
					{/if}
				{:else if result.trace}
					<ProofInspector trace={result.trace} />
				{/if}
			</div>
		{/if}
	</div>
{/if}
