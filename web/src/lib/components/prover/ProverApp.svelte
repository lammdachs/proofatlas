<script lang="ts">
	import {
		detectServer,
		initWasm,
		proveViaServer,
		proveViaWasm,
		stopWasmProver,
		validateConfig,
		type ProveResult,
		type ServerStatus,
	} from '$lib/prover';
	import ConfigPanel from './ConfigPanel.svelte';
	import InputPanel from './InputPanel.svelte';
	import ResultDisplay from './ResultDisplay.svelte';

	let serverAvailable = $state(false);
	let mlAvailable = $state(false);
	let wasmReady = $state(false);
	let proving = $state(false);
	let cancelled = $state(false);
	let result: ProveResult | null = $state(null);
	let error: string | null = $state(null);
	let abortController: AbortController | null = null;

	let tptpInput = $state('');
	let configJson = $state('');

	// Detect server + init WASM on mount
	$effect(() => {
		init();
	});

	async function init() {
		const status = await detectServer();
		serverAvailable = status.available;
		mlAvailable = status.mlAvailable;

		wasmReady = await initWasm();
	}

	function stop() {
		cancelled = true;
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		stopWasmProver();
		proving = false;
	}

	async function prove() {
		const input = tptpInput.trim();
		if (!input) {
			error = 'Please enter a TPTP problem';
			return;
		}

		let config: Record<string, unknown>;
		try {
			config = JSON.parse(configJson);
		} catch {
			error = 'Invalid config JSON';
			return;
		}

		const configError = validateConfig(config);
		if (configError) {
			error = configError;
			return;
		}

		error = null;
		result = null;
		proving = true;
		cancelled = false;
		abortController = new AbortController();

		try {
			let proveResult: ProveResult;
			if (serverAvailable) {
				proveResult = await proveViaServer(input, config, abortController.signal);
			} else if (wasmReady) {
				proveResult = await proveViaWasm(input, config);
			} else {
				error = 'No prover available (server offline, WASM not loaded)';
				return;
			}
			if (!cancelled) {
				result = proveResult;
			}
		} catch (e) {
			if (!cancelled) {
				const msg = e instanceof Error ? e.message : String(e);
				if (msg !== 'The operation was aborted.' && msg !== 'AbortError') {
					error = msg;
				}
			}
		} finally {
			proving = false;
			abortController = null;
		}
	}
</script>

<div class="space-y-6 py-6 stagger-children">
	<!-- Input -->
	<InputPanel bind:tptpInput {serverAvailable} onprove={prove} />

	<!-- Config + Prove -->
	<ConfigPanel bind:configJson {serverAvailable} {mlAvailable}>
		{#if proving}
			<button
				class="btn btn-warning"
				onclick={stop}
			>
				Stop
			</button>
		{:else}
			<button
				class="btn btn-primary"
				disabled={!serverAvailable && !wasmReady}
				onclick={prove}
			>
				Prove
			</button>
		{/if}
	</ConfigPanel>

	<!-- Error -->
	{#if error}
		<div class="p-5 border-b-2" style="border-color: var(--color-status-error); background: color-mix(in srgb, var(--color-status-error) 8%, transparent);">
			<p class="text-sm font-semibold" style="color: var(--color-status-error)">{error}</p>
		</div>
	{/if}

	<!-- Working indicator -->
	{#if proving}
		<div class="p-5 flex items-center gap-3 border-b-2" style="border-color: var(--color-status-working); background: color-mix(in srgb, var(--color-status-working) 8%, transparent);">
			<span class="spinner"></span>
			<p class="text-sm font-semibold" style="color: var(--color-status-working)">Solving...</p>
		</div>
	{/if}

	<!-- Result -->
	<ResultDisplay {result} />
</div>
