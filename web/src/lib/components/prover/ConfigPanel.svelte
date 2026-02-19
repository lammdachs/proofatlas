<script lang="ts">
	import { BUILTIN_PRESETS } from '$lib/prover';
	import { base } from '$app/paths';
	import type { Snippet } from 'svelte';

	let {
		serverAvailable,
		mlAvailable = false,
		configJson = $bindable(''),
		children,
	} = $props<{
		serverAvailable: boolean;
		mlAvailable?: boolean;
		configJson: string;
		children?: Snippet;
	}>();

	let presets: Record<string, Record<string, unknown>> = $state({});
	let selectedPreset = $state('time');
	let configOpen = $state(false);

	$effect(() => {
		loadPresets();
	});

	async function loadPresets() {
		let loaded: Record<string, Record<string, unknown>> | null = null;

		try {
			let response;
			if (serverAvailable) {
				response = await fetch(`${base}/configs/proofatlas.json`);
			}
			if (!response || !response.ok) {
				response = await fetch(`${base}/configs/proofatlas.json`);
			}
			if (response?.ok) {
				const data = await response.json();
				loaded = data.presets;
			}
		} catch {
			// Fall back to built-in
		}

		if (!loaded) {
			loaded = BUILTIN_PRESETS;
		}

		// Without server, filter to heuristic-only presets
		if (!serverAvailable) {
			const filtered: typeof loaded = {};
			for (const [name, preset] of Object.entries(loaded)) {
				if (!preset.encoder) filtered[name] = preset;
			}
			loaded = filtered;
		}

		presets = loaded;

		// Apply default preset
		if (presets['time']) {
			selectedPreset = 'time';
			applyPreset('time');
		}
	}

	function applyPreset(name: string) {
		const preset = presets[name];
		if (!preset) return;
		selectedPreset = name;
		configJson = JSON.stringify(preset, null, 2);
	}

	function handlePresetChange(e: Event) {
		const value = (e.target as HTMLSelectElement).value;
		if (!value) {
			selectedPreset = '';
			return;
		}
		applyPreset(value);
	}

	function handleJsonInput() {
		// Manual edit resets preset to Custom
		selectedPreset = '';
	}
</script>

<div class="space-y-3">
	<div class="flex w-full items-center justify-between">
		<div class="flex items-center gap-3">
			<label class="flex items-center gap-2 text-sm text-text-muted">
				Config:
				<select
					class="input min-w-48"
					value={selectedPreset}
					onchange={handlePresetChange}
				>
					<option value="">Custom</option>
					{#each Object.entries(presets) as [name, preset]}
						<option value={name} title={preset.description as string || ''}>{name}</option>
					{/each}
				</select>
			</label>
			{#if !serverAvailable}
				<span class="text-xs text-text-muted">ML configs require a server</span>
			{:else if !mlAvailable}
				<span class="text-xs text-text-muted">ML configs require model weights</span>
			{/if}
		</div>
		{#if children}
			{@render children()}
		{/if}
	</div>

	<details bind:open={configOpen}>
		<summary class="py-2 cursor-pointer text-xs uppercase tracking-wide text-green font-mono hover:text-text select-none transition-colors">
			// config json
		</summary>
		<div class="pt-2">
			<textarea
				class="input w-full mt-4 p-4 font-mono text-sm resize-none"
				style="field-sizing: content; tab-size: 2; border-radius: 0;"
				spellcheck="false"
				bind:value={configJson}
				oninput={handleJsonInput}
			></textarea>
		</div>
	</details>
</div>
