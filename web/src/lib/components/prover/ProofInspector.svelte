<script lang="ts">
	import type { Trace, Iteration, TraceEvent } from '$lib/prover';
	import { onMount, onDestroy } from 'svelte';

	let { trace } = $props<{ trace: Trace }>();

	// State
	let currentStep = $state(0);
	let currentEventIdx = $state(0);
	let mobileClauseView: 'N' | 'U' | 'P' = $state('N');

	// Clause sets
	let newClauses = $state(new Set<number>());
	let unprocessedClauses = $state(new Set<number>());
	let processedClauses = $state(new Set<number>());

	// Build clause lookup
	let clauseMap = $derived.by(() => {
		const map = new Map<number, { id: number; clause: string; rule: string; parents: number[] }>();
		if (trace?.initial_clauses) {
			for (const c of trace.initial_clauses) {
				map.set(c.id, { id: c.id, clause: c.clause, rule: 'Input', parents: [] });
			}
		}
		if (trace?.iterations) {
			for (const iter of trace.iterations) {
				for (const ev of iter.simplification) {
					if (!map.has(ev.clause_idx)) {
						map.set(ev.clause_idx, { id: ev.clause_idx, clause: ev.clause, rule: ev.rule, parents: ev.premises || [] });
					}
				}
				if (iter.selection && !map.has(iter.selection.clause_idx)) {
					map.set(iter.selection.clause_idx, { id: iter.selection.clause_idx, clause: iter.selection.clause, rule: iter.selection.rule, parents: [] });
				}
				for (const ev of iter.generation) {
					if (!map.has(ev.clause_idx)) {
						map.set(ev.clause_idx, { id: ev.clause_idx, clause: ev.clause, rule: ev.rule, parents: ev.premises || [] });
					}
				}
			}
		}
		return map;
	});

	let iterations = $derived(trace?.iterations || []);
	let maxStep = $derived(iterations.length - 1);
	let currentIter = $derived(iterations[currentStep]);

	// Set of clause IDs involved in the current event (for highlighting in clause sets)
	let highlightedClauseIds = $derived.by(() => {
		const ids = new Set<number>();
		if (currentEvent) {
			ids.add(currentEvent.clause_idx);
			if (currentEvent.premises) {
				for (const p of currentEvent.premises) ids.add(p);
			}
		}
		return ids;
	});

	function flattenEvents(iter: Iteration): TraceEvent[] {
		const events: TraceEvent[] = [];
		for (const ev of iter.simplification) events.push(ev);
		if (iter.selection) events.push(iter.selection);
		for (const ev of iter.generation) events.push(ev);
		return events;
	}

	let currentFlatEvents = $derived(iterations.length > 0 ? flattenEvents(iterations[currentStep]) : []);
	let currentEvent = $derived(currentEventIdx >= 0 && currentEventIdx < currentFlatEvents.length ? currentFlatEvents[currentEventIdx] : null);

	function replayEvent(ev: TraceEvent, n: Set<number>, u: Set<number>, p: Set<number>) {
		const rule = ev.rule;
		if (['TautologyDeletion', 'SubsumptionDeletion', 'DemodulationDeletion',
			'ForwardSubsumptionDeletion', 'BackwardSubsumptionDeletion', 'ForwardDemodulation'].includes(rule)) {
			n.delete(ev.clause_idx);
			u.delete(ev.clause_idx);
			p.delete(ev.clause_idx);
		} else if (rule === 'Transfer') {
			n.delete(ev.clause_idx);
			u.add(ev.clause_idx);
		} else if (rule === 'Demodulation') {
			n.add(ev.clause_idx);
		} else if (rule === 'GivenClauseSelection') {
			u.delete(ev.clause_idx);
			p.add(ev.clause_idx);
		} else {
			n.add(ev.clause_idx);
		}
	}

	function goToStep(iterNum: number, eventIdx = 0) {
		if (iterations.length === 0) return;
		if (iterNum < 0 || iterNum > maxStep) return;

		const n = new Set<number>();
		const u = new Set<number>();
		const p = new Set<number>();

		for (let i = 0; i < iterNum; i++) {
			for (const ev of flattenEvents(iterations[i])) {
				replayEvent(ev, n, u, p);
			}
		}

		if (eventIdx > 0) {
			const events = flattenEvents(iterations[iterNum]);
			for (let i = 0; i < eventIdx && i < events.length; i++) {
				replayEvent(events[i], n, u, p);
			}
		}

		newClauses = n;
		unprocessedClauses = u;
		processedClauses = p;
		currentStep = iterNum;
		currentEventIdx = eventIdx;
	}

	function stepForward() {
		if (iterations.length === 0) return;
		const lastIdx = currentFlatEvents.length - 1;
		if (currentEventIdx < lastIdx) {
			goToStep(currentStep, currentEventIdx + 1);
		} else if (currentStep < maxStep) {
			goToStep(currentStep + 1, 0);
		}
	}

	function stepBackward() {
		if (iterations.length === 0) return;
		if (currentEventIdx > 0) {
			goToStep(currentStep, currentEventIdx - 1);
		} else if (currentStep > 0) {
			const prevEvents = flattenEvents(iterations[currentStep - 1]);
			goToStep(currentStep - 1, prevEvents.length - 1);
		}
	}

	function goFirst() { goToStep(0, 0); }
	function goLast() {
		if (iterations.length === 0) return;
		const lastEvents = flattenEvents(iterations[maxStep]);
		goToStep(maxStep, Math.max(0, lastEvents.length - 1));
	}

	let isAtStart = $derived(currentStep <= 0 && currentEventIdx <= 0);
	let isAtEnd = $derived.by(() => {
		if (iterations.length === 0) return true;
		const lastEvents = flattenEvents(iterations[maxStep]);
		return currentStep >= maxStep && currentEventIdx >= lastEvents.length - 1;
	});

	function eventLabel(ev: TraceEvent): string {
		switch (ev.rule) {
			case 'TautologyDeletion': return 'Tautology';
			case 'SubsumptionDeletion':
			case 'ForwardSubsumptionDeletion': return 'Fwd Subsumption';
			case 'BackwardSubsumptionDeletion': return 'Bwd Subsumption';
			case 'DemodulationDeletion':
			case 'ForwardDemodulation': return 'Fwd Demodulation';
			case 'Transfer': return 'Transfer';
			case 'Input': return 'Input';
			case 'Demodulation': return 'Bwd Demodulation';
			case 'GivenClauseSelection': return 'Selection';
			default: return ev.rule;
		}
	}

	function eventCategory(rule: string): 'deletion' | 'transfer' | 'selection' | 'simplify' | 'generation' {
		if (['TautologyDeletion', 'SubsumptionDeletion',
			'ForwardSubsumptionDeletion', 'BackwardSubsumptionDeletion'].includes(rule)) {
			return 'deletion';
		}
		if (rule === 'Transfer' || rule === 'Input') return 'transfer';
		if (rule === 'GivenClauseSelection') return 'selection';
		if (rule === 'Demodulation' || rule === 'DemodulationDeletion') return 'simplify';
		return 'generation';
	}

	function categoryColor(cat: string): string {
		switch (cat) {
			case 'deletion': return 'var(--color-event-deletion)';
			case 'transfer': return 'var(--color-event-generation)';
			case 'selection': return 'var(--color-event-generation)';
			case 'simplify': return 'var(--color-event-simplify)';
			default: return 'var(--color-event-transfer)';
		}
	}

	function formatClauseRef(id: number): string {
		const c = clauseMap.get(id);
		const text = c ? escapeHtml(c.clause) : '?';
		return `<span class="text-text-muted">[${id}]</span> ${text}`;
	}

	function escapeHtml(text: string): string {
		return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
	}

	// Keyboard/mouse handlers
	function handleKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
		if (e.key === 'ArrowLeft') { e.preventDefault(); if (currentStep > 0) goToStep(currentStep - 1, 0); }
		else if (e.key === 'ArrowRight') { e.preventDefault(); if (currentStep < maxStep) goToStep(currentStep + 1, 0); }
		else if (e.key === 'ArrowUp') { e.preventDefault(); stepBackward(); }
		else if (e.key === 'ArrowDown') { e.preventDefault(); stepForward(); }
	}

	function handleMouseUp(e: MouseEvent) {
		if (e.button === 3) { e.preventDefault(); stepBackward(); }
		else if (e.button === 4) { e.preventDefault(); stepForward(); }
	}

	function handleMouseDown(e: MouseEvent) {
		if (e.button === 3 || e.button === 4) e.preventDefault();
	}

	onMount(() => {
		document.addEventListener('keydown', handleKeydown);
		document.addEventListener('mouseup', handleMouseUp);
		document.addEventListener('mousedown', handleMouseDown);
		if (iterations.length > 0) goToStep(0, 0);
	});

	onDestroy(() => {
		document.removeEventListener('keydown', handleKeydown);
		document.removeEventListener('mouseup', handleMouseUp);
		document.removeEventListener('mousedown', handleMouseDown);
	});

	let clauseSets = $derived([
		{ key: 'N' as const, title: 'N', full: 'New', set: newClauses },
		{ key: 'U' as const, title: 'U', full: 'Unprocessed', set: unprocessedClauses },
		{ key: 'P' as const, title: 'P', full: 'Processed', set: processedClauses },
	]);

	let mobileActiveSet = $derived(clauseSets.find(s => s.key === mobileClauseView)!);

	// Visible event window: always 5 rows (dots count as rows)
	let eventWindow = $derived.by(() => {
		const total = currentFlatEvents.length;
		if (total === 0) return { start: 0, end: -1, topDots: false, bottomDots: false };
		if (total <= 5) return { start: 0, end: total - 1, topDots: false, bottomDots: false };

		let start = currentEventIdx - 1;
		let end = currentEventIdx + 1;

		// At the start: no top dots, show 4 events instead of 3
		if (start <= 0) { start = 0; end = 3; }
		// At the end: no bottom dots, show 4 events instead of 3
		if (end >= total - 1) { end = total - 1; start = total - 4; }

		return {
			start,
			end,
			topDots: start > 0,
			bottomDots: end < total - 1,
		};
	});

	function renderClauseSet(clauseIds: Set<number>): { id: number; text: string }[] {
		return Array.from(clauseIds).sort((a, b) => a - b).map(id => {
			const c = clauseMap.get(id);
			return { id, text: c ? c.clause : '(not available)' };
		});
	}
</script>

{#if iterations.length > 0}
	<div class="space-y-6">
		<!-- Events Timeline + Event Detail -->
		<div>
			<div class="flex items-center justify-between mb-2">
				<h4 class="text-xs uppercase tracking-wide text-green font-mono">// iteration {currentStep + 1}/{maxStep + 1} &middot; event {currentEventIdx + 1}/{currentFlatEvents.length}</h4>
				<span class="text-text-muted text-xs font-mono">
					<kbd class="px-1.5 py-0.5 border border-surface-lighter/50">&#8592;&#8594;</kbd> iterations
					<kbd class="ml-2 px-1.5 py-0.5 border border-surface-lighter/50">&#8593;&#8595;</kbd> events
				</span>
			</div>
			<div class="border border-surface-lighter/50 flex flex-col md:flex-row">
				<!-- Left: event window -->
				<div class="md:w-80 shrink-0 border-b md:border-b-0 md:border-r border-surface-lighter/50" style="min-height: 8.75rem;">
					{#if currentFlatEvents.length > 0}
						{#if eventWindow.topDots}
							<div class="px-3 py-1.5 text-center font-mono text-text-muted text-xs tracking-widest">···</div>
						{/if}
						{#each currentFlatEvents.slice(eventWindow.start, eventWindow.end + 1) as ev, j}
							{@const i = eventWindow.start + j}
							{@const cat = eventCategory(ev.rule)}
							{@const active = i === currentEventIdx}
							<button
								class="w-full text-left px-3 py-1.5 font-mono text-sm cursor-pointer transition-all flex items-center gap-2 hover:bg-surface-lighter/30 {active ? 'text-text font-semibold' : 'text-text-muted'}"
								style={active ? `border-left: 2px solid ${categoryColor(cat)}; background: color-mix(in srgb, ${categoryColor(cat)} 8%, transparent);` : 'border-left: 2px solid transparent;'}
								onclick={() => goToStep(currentStep, i)}
							>
								<span class="w-1.5 h-1.5 rounded-full shrink-0" style="background: {categoryColor(cat)}"></span>
								{eventLabel(ev)}
								<span class="opacity-50">[{ev.clause_idx}]</span>
							</button>
						{/each}
						{#if eventWindow.bottomDots}
							<div class="px-3 py-1.5 text-center font-mono text-text-muted text-xs tracking-widest">···</div>
						{/if}
					{:else}
						<div class="px-3 py-6 text-center text-xs text-text-muted italic">No events</div>
					{/if}
				</div>

				<!-- Right: event detail -->
				<div class="flex-1 p-4">
					{#if currentEvent}
						{@const ev = currentEvent}
						{@const cat = eventCategory(ev.rule)}
						{@const isDeletion = cat === 'deletion'}
						{@const isDemodDeletion = ev.rule === 'DemodulationDeletion' && ev.premises?.length >= 2}
						{@const isTransfer = ev.rule === 'Transfer'}
						{@const isSelection = ev.rule === 'GivenClauseSelection'}
						{@const isDemodAdd = ev.rule === 'Demodulation'}
						<table class="w-full text-sm">
							<tbody>
								{#if isDemodDeletion}
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Rewritten</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.clause_idx)}</td>
									</tr>
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Using</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.premises[1])}</td>
									</tr>
									{#if ev.replacement_idx != null}
										<tr class="border-b border-surface-lighter/20 last:border-0">
											<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Result</td>
											<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.replacement_idx)}</td>
										</tr>
									{/if}
								{:else if isDeletion}
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Deleted</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.clause_idx)}</td>
									</tr>
									{#if ev.premises.length > 0}
										<tr class="border-b border-surface-lighter/20 last:border-0">
											<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">By</td>
											<td class="py-2 font-mono text-sm text-text leading-relaxed">{#each ev.premises as p}{@html formatClauseRef(p)}<br/>{/each}</td>
										</tr>
									{/if}
								{:else if isTransfer}
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Clause</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.clause_idx)}</td>
									</tr>
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Move</td>
										<td class="py-2 text-sm text-text">N &rarr; U</td>
									</tr>
								{:else if isSelection}
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Given</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.clause_idx)}</td>
									</tr>
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Move</td>
										<td class="py-2 text-sm text-text">U &rarr; P</td>
									</tr>
								{:else if isDemodAdd && ev.premises.length >= 2}
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Rewritten</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.premises[0])}</td>
									</tr>
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Using</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.premises[1])}</td>
									</tr>
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Result</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.clause_idx)}</td>
									</tr>
								{:else}
									{#if ev.premises.length > 0}
										<tr class="border-b border-surface-lighter/20 last:border-0">
											<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Premises</td>
											<td class="py-2 font-mono text-sm text-text leading-relaxed">{#each ev.premises as p}{@html formatClauseRef(p)}<br/>{/each}</td>
										</tr>
									{/if}
									<tr class="border-b border-surface-lighter/20 last:border-0">
										<td class="w-24 py-2 text-xs uppercase tracking-wide text-text-muted font-mono align-top">Result</td>
										<td class="py-2 font-mono text-sm text-text leading-relaxed">{@html formatClauseRef(ev.clause_idx)}</td>
									</tr>
								{/if}
							</tbody>
						</table>
					{:else}
						<div class="py-8 text-center text-xs text-text-muted italic">Select an event</div>
					{/if}
				</div>
			</div>
		</div>

		<!-- 3. Clause Sets -->
		<div>
			<h4 class="text-xs uppercase tracking-wide text-green font-mono mb-2">// clause sets</h4>
			<div class="border border-surface-lighter/50">
				<!-- Header row -->
				<div class="flex items-center border-b border-surface-lighter/50">
					<!-- Desktop: three column headers -->
					<div class="hidden md:flex w-full">
						{#each clauseSets as col, i}
							<div class="flex-1 px-4 py-2.5 flex items-center gap-2 {i < 2 ? 'border-r border-surface-lighter/50' : ''}">
								<span class="text-xs font-mono text-text uppercase tracking-wide">{col.full}</span>
								<span class="text-xs font-mono text-text-muted tabular-nums">{col.set.size}</span>
							</div>
						{/each}
					</div>
					<!-- Mobile: tab selector -->
					<div class="flex md:hidden w-full">
						{#each clauseSets as col}
							<button
								class="flex-1 px-4 py-2.5 text-center font-mono text-xs uppercase tracking-wide cursor-pointer transition-colors {mobileClauseView === col.key ? 'text-text border-b-2 border-text' : 'text-text-muted hover:text-text'}"
								onclick={() => mobileClauseView = col.key}
							>
								{col.title} <span class="opacity-60">{col.set.size}</span>
							</button>
						{/each}
					</div>
				</div>

				<!-- Desktop: three columns -->
				<div class="hidden md:flex">
					{#each clauseSets as col, i}
						<div class="flex-1 max-h-80 overflow-y-auto {i < 2 ? 'border-r border-surface-lighter/50' : ''}">
							{#each renderClauseSet(col.set) as clause}
								<div
									class="px-3 py-1 font-mono text-sm leading-relaxed truncate hover:whitespace-normal hover:break-all transition-all {highlightedClauseIds.has(clause.id) ? 'text-text' : 'text-text opacity-60'}"
									style={highlightedClauseIds.has(clause.id) && currentEvent
										? `background: color-mix(in srgb, ${categoryColor(eventCategory(currentEvent.rule))} 12%, transparent);`
										: ''}
									title="[{clause.id}] {clause.text}"
								>
									<span class="opacity-50">[{clause.id}]</span> {clause.text}
								</div>
							{/each}
							{#if col.set.size === 0}
								<div class="px-3 py-6 text-center text-xs text-text-muted italic">Empty</div>
							{/if}
						</div>
					{/each}
				</div>

				<!-- Mobile: single active set -->
				<div class="md:hidden max-h-80 overflow-y-auto">
					{#each renderClauseSet(mobileActiveSet.set) as clause}
						<div
							class="px-3 py-1 font-mono text-sm leading-relaxed truncate hover:whitespace-normal hover:break-all transition-all {highlightedClauseIds.has(clause.id) ? 'text-text' : 'text-text opacity-60'}"
							style={highlightedClauseIds.has(clause.id) && currentEvent
								? `background: color-mix(in srgb, ${categoryColor(eventCategory(currentEvent.rule))} 12%, transparent);`
								: ''}
							title="[{clause.id}] {clause.text}"
						>
							<span class="opacity-50">[{clause.id}]</span> {clause.text}
						</div>
					{/each}
					{#if mobileActiveSet.set.size === 0}
						<div class="px-3 py-6 text-center text-xs text-text-muted italic">Empty</div>
					{/if}
				</div>
			</div>
		</div>
	</div>
{:else}
	<p class="text-text-muted text-sm italic">No trace data available</p>
{/if}
