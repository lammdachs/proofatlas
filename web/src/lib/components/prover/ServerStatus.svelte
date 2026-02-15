<script lang="ts">
	let { serverAvailable, mlAvailable } = $props<{ serverAvailable: boolean; mlAvailable: boolean }>();
</script>

<span class="inline-flex items-center gap-1.5 text-xs text-text-muted font-mono">
	<span class="w-1.5 h-1.5 rounded-full {serverAvailable ? (mlAvailable ? 'bg-green-400' : 'bg-honey-400') : 'bg-blue-400'}"></span>
	{#if serverAvailable}
		{mlAvailable ? 'server + ml' : 'server'}
	{:else}
		browser
	{/if}
</span>
