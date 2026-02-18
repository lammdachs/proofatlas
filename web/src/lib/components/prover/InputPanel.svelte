<script lang="ts">
	import { base } from '$app/paths';

	const TPTP_MIRROR = 'https://raw.githubusercontent.com/lammdachs/proofatlas-tptp-subset/main';

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

	/** Map a TPTP problem name (e.g. "GRP001-1") to its domain directory. */
	function tptpDomain(name: string): string {
		return name.replace(/[0-9].*$/, '');
	}

	/** Fetch a file from the GitHub mirror, returning its text. */
	async function fetchMirror(path: string): Promise<string> {
		const response = await fetch(`${TPTP_MIRROR}/${path}`);
		if (!response.ok) throw new Error(`Not found: ${path}`);
		return response.text();
	}

	/** Replace include('Axioms/...') directives with the actual axiom file contents. */
	async function resolveIncludes(content: string): Promise<string> {
		const includeRe = /^include\('([^']+)'\s*(?:,\s*\[[^\]]*\])?\s*\)\.\s*$/gm;
		const matches = [...content.matchAll(includeRe)];
		if (matches.length === 0) return content;

		const fetches = matches.map(m => fetchMirror(m[1]).catch(() => `% Failed to load ${m[1]}\n`));
		const axiomContents = await Promise.all(fetches);

		let result = content;
		for (let i = matches.length - 1; i >= 0; i--) {
			const m = matches[i];
			result = result.slice(0, m.index!) + axiomContents[i] + result.slice(m.index! + m[0].length);
		}
		return result;
	}

	async function loadProblem() {
		if (!tptpName.trim()) return;
		loadingProblem = true;
		try {
			const name = tptpName.trim();
			const file = name.endsWith('.p') ? name : `${name}.p`;
			let content: string | null = null;

			// Try local server first (server resolves includes via local TPTP)
			if (serverAvailable) {
				try {
					const response = await fetch(`${base}/api/tptp/${encodeURIComponent(name)}`);
					if (response.ok) {
						const data = await response.json();
						content = data.content;
					}
				} catch { /* fall through to mirror */ }
			}

			// Fall back to GitHub mirror — fetch problem and inline axiom includes
			if (content === null) {
				const domain = tptpDomain(file.replace('.p', ''));
				const raw = await fetchMirror(`Problems/${domain}/${file}`);
				content = await resolveIncludes(raw);
			}

			tptpInput = content;
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
			placeholder="Problem name (e.g., GRP001-1)"
			bind:value={tptpName}
			onkeydown={handleTptpNameKeydown}
		/>

		<button
			class="btn btn-secondary"
			disabled={loadingProblem}
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
