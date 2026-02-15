<script lang="ts">
	import '../styles/global.css';
	import ThemeToggle from '$lib/components/ThemeToggle.svelte';
	import { page } from '$app/state';

	let { children } = $props();
	let mobileMenuOpen = $state(false);

	const navLinks = [
		{ href: '/', label: 'Prover' },
		{ href: '/docs', label: 'Docs' },
		{ href: '/results', label: 'Results' },
		{ href: '/runs', label: 'Runs' },
		{ href: '/training', label: 'Training' },
	];

	function isActive(href: string): boolean {
		if (href === '/') return page.url.pathname === '/';
		return page.url.pathname.startsWith(href);
	}
</script>

<!-- Background blobs -->
<div class="fixed inset-0 overflow-hidden pointer-events-none -z-10">
	<div class="absolute top-20 left-10 w-72 h-72 bg-green/30 rounded-full blur-3xl animate-blob"></div>
	<div class="absolute top-40 right-20 w-96 h-96 bg-terracotta/25 rounded-full blur-3xl animate-blob" style="animation-delay: 2s;"></div>
	<div class="absolute bottom-20 left-1/3 w-80 h-80 bg-accent/30 rounded-full blur-3xl animate-blob" style="animation-delay: 4s;"></div>
</div>

<!-- Navigation -->
<nav class="fixed top-0 left-0 right-0 z-50 px-6 py-4 backdrop-blur-md bg-surface/80 border-b border-surface-lighter/50">
	<div class="max-w-6xl mx-auto flex items-center justify-between">
		<a href="/" class="text-2xl font-bold font-display bg-gradient-to-r from-green to-terracotta bg-clip-text text-transparent hover:opacity-80 transition-opacity">
			ProofAtlas
		</a>

		<!-- Desktop nav -->
		<div class="hidden md:flex items-center gap-8">
			{#each navLinks as link}
				<a
					href={link.href}
					class="link-underline transition-colors {isActive(link.href) ? 'text-green font-semibold' : 'text-text-muted hover:text-text'}"
				>
					{link.label}
				</a>
			{/each}
			<a
				href="https://github.com/lexpk/proofatlas"
				target="_blank"
				rel="noopener noreferrer"
				class="text-text-muted hover:text-text transition-colors"
				aria-label="GitHub"
			>
				<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
					<path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
				</svg>
			</a>
			<ThemeToggle />
		</div>

		<!-- Mobile controls -->
		<div class="flex items-center gap-2 md:hidden">
			<ThemeToggle />
			<button
				class="p-2 text-text-muted hover:text-text transition-colors"
				aria-label="Toggle menu"
				onclick={() => mobileMenuOpen = !mobileMenuOpen}
			>
				<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
				</svg>
			</button>
		</div>
	</div>

	<!-- Mobile menu -->
	{#if mobileMenuOpen}
		<div class="md:hidden mt-4 pb-4 border-t border-surface-lighter/50 pt-4">
			<div class="flex flex-col gap-4">
				{#each navLinks as link}
					<a
						href={link.href}
						class="transition-colors {isActive(link.href) ? 'text-green font-semibold' : 'text-text-muted hover:text-text'}"
						onclick={() => mobileMenuOpen = false}
					>
						{link.label}
					</a>
				{/each}
				<a
					href="https://github.com/lexpk/proofatlas"
					target="_blank"
					rel="noopener noreferrer"
					class="text-text-muted hover:text-text transition-colors"
				>
					GitHub
				</a>
			</div>
		</div>
	{/if}
</nav>

<main class="pt-24 pb-16 px-6 min-h-screen">
	<div class="max-w-6xl mx-auto">
		{@render children()}
	</div>
</main>

<footer class="border-t border-surface-lighter/50 bg-surface/50 backdrop-blur-sm">
	<div class="max-w-6xl mx-auto px-6 py-8">
		<div class="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-text-muted">
			<span>ProofAtlas by <a href="https://lammdachs.io" class="text-green hover:text-text transition-colors">lammdachs</a></span>
			<a
				href="https://github.com/lexpk/proofatlas"
				target="_blank"
				rel="noopener noreferrer"
				class="hover:text-text transition-colors"
			>
				GitHub
			</a>
		</div>
	</div>
</footer>
