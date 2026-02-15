<script lang="ts">
	import { base } from '$app/paths';

	let {
		tptpInput = $bindable(''),
		serverAvailable,
		onprove,
	} = $props<{
		tptpInput: string;
		serverAvailable: boolean;
		onprove: () => void;
	}>();

	interface Example {
		id: string;
		title: string;
		description: string;
		file: string;
	}

	let examples: Example[] = $state([]);
	let exampleContents: Record<string, string> = $state({});
	let selectedExample = $state('');
	let tptpName = $state('');
	let loadingProblem = $state(false);

	$effect(() => {
		loadExamples();
	});

	async function loadExamples() {
		try {
			const response = await fetch(`${base}/examples/examples.json`);
			if (!response.ok) return;
			const data = await response.json();
			examples = data.examples;

			// Preload all example files
			for (const ex of examples) {
				try {
					const fileResp = await fetch(`${base}/examples/${ex.file}`);
					if (fileResp.ok) exampleContents[ex.id] = await fileResp.text();
				} catch { /* skip */ }
			}
		} catch {
			// Examples unavailable
		}
	}

	function handleExampleChange(e: Event) {
		const id = (e.target as HTMLSelectElement).value;
		selectedExample = id;
		if (id && exampleContents[id]) {
			tptpInput = exampleContents[id];
		}
	}

	async function loadProblem() {
		if (!tptpName.trim()) return;
		loadingProblem = true;
		try {
			const response = await fetch(`${base}/api/tptp/${encodeURIComponent(tptpName.trim())}`);
			if (!response.ok) {
				const err = await response.json().catch(() => ({}));
				throw new Error(err.error || `Server error: ${response.status}`);
			}
			const data = await response.json();
			tptpInput = data.content;
			selectedExample = '';
			tptpName = '';
		} catch (error) {
			alert(`Error loading problem: ${error}`);
		} finally {
			loadingProblem = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.ctrlKey && e.key === 'Enter') {
			onprove();
		}
	}

	function handleTptpNameKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') loadProblem();
	}
</script>

<div class="space-y-4">
	<!-- Toolbar -->
	<div class="flex flex-wrap items-center gap-3">
		<select
			class="input flex-1 min-w-48"
			value={selectedExample}
			onchange={handleExampleChange}
		>
			<option value="">Select an example...</option>
			{#each examples as ex}
				<option value={ex.id} title={ex.description}>{ex.title}</option>
			{/each}
		</select>

		<button
			class="btn btn-secondary"
			onclick={() => { tptpInput = ''; selectedExample = ''; }}
		>
			Clear
		</button>

		<input
			type="text"
			class="input flex-1 min-w-40"
			placeholder={serverAvailable ? 'Problem name (e.g., GRP001-1)' : 'Install locally for TPTP loading'}
			disabled={!serverAvailable}
			bind:value={tptpName}
			onkeydown={handleTptpNameKeydown}
		/>

		<button
			class="btn btn-secondary"
			disabled={!serverAvailable || loadingProblem}
			onclick={loadProblem}
		>
			{loadingProblem ? 'Loading...' : 'Load'}
		</button>
	</div>

	<!-- TPTP Input -->
	<textarea
		class="input w-full h-72 p-4 font-mono text-[15px] leading-relaxed resize-y"
		style="border-radius: 1rem;"
		placeholder="Enter TPTP problem here..."
		spellcheck="false"
		bind:value={tptpInput}
		onkeydown={handleKeydown}
	></textarea>
</div>
