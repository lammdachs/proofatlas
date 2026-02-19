/** Non-UI proving logic: server detection, WASM init, prove functions. */
import { browser } from '$app/environment';
import { base } from '$app/paths';

export interface ProveResult {
	success: boolean;
	status: string;
	message: string;
	proof: ProofStep[] | null;
	all_clauses: ProofStep[] | null;
	statistics: Statistics;
	trace: Trace | null;
	profile: Record<string, unknown> | null;
}

export interface ProofStep {
	id: number;
	clause: string;
	rule: string;
	parents: number[];
}

export interface Statistics {
	initial_clauses: number;
	generated_clauses: number;
	final_clauses: number;
	time_ms: number;
}

export interface Trace {
	initial_clauses: { id: number; clause: string }[];
	iterations: Iteration[];
}

export interface Iteration {
	simplification: TraceEvent[];
	selection: TraceEvent | null;
	generation: TraceEvent[];
}

export interface TraceEvent {
	clause_idx: number;
	clause: string;
	rule: string;
	premises: number[];
	replacement_idx?: number;
}

export interface ServerStatus {
	available: boolean;
	mlAvailable: boolean;
}

export async function detectServer(): Promise<ServerStatus> {
	try {
		const response = await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(2000) });
		if (response.ok) {
			const data = await response.json();
			return { available: true, mlAvailable: data.ml_available || false };
		}
	} catch {
		// Network error or timeout
	}
	return { available: false, mlAvailable: false };
}

export async function initWasm(): Promise<boolean> {
	if (!browser) return false;
	try {
		// Check that the WASM JS file is fetchable
		const wasmPath = `${base}/pkg/proofatlas_wasm_bg.wasm`;
		const response = await fetch(wasmPath, { method: 'HEAD', signal: AbortSignal.timeout(2000) });
		return response.ok;
	} catch {
		return false;
	}
}

export async function proveViaServer(input: string, config: Record<string, unknown>, signal?: AbortSignal): Promise<ProveResult> {
	const body = { input, ...config };
	const response = await fetch(`${base}/api/prove`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
		signal,
	});

	if (!response.ok) {
		const err = await response.json().catch(() => ({}));
		throw new Error(err.error || `Server error: ${response.status}`);
	}

	return await response.json();
}

export function configToWasmOptions(config: Record<string, unknown>): Record<string, unknown> {
	const opts: Record<string, unknown> = { selector_type: 'age_weight' };
	if (config.timeout !== undefined) opts.timeout_ms = (config.timeout as number) * 1000;
	if (config.literal_selection !== undefined) opts.literal_selection = String(config.literal_selection);
	if (config.age_weight_ratio !== undefined) opts.age_weight_ratio = config.age_weight_ratio;
	if (config.max_iterations !== undefined) opts.max_iterations = config.max_iterations;
	return opts;
}

let activeWorker: Worker | null = null;

export function stopWasmProver(): void {
	if (activeWorker) {
		activeWorker.terminate();
		activeWorker = null;
	}
}

export async function proveViaWasm(input: string, config: Record<string, unknown>): Promise<ProveResult> {
	stopWasmProver();

	const options = configToWasmOptions(config);
	const workerPath = `${base}/prover-worker.js`;

	return new Promise((resolve, reject) => {
		const worker = new Worker(workerPath, { type: 'module' });
		activeWorker = worker;
		worker.postMessage({ type: 'init', basePath: base });

		worker.onmessage = (e) => {
			activeWorker = null;
			if (e.data.type === 'result') {
				resolve(e.data.result);
			} else if (e.data.type === 'error') {
				reject(new Error(e.data.message));
			}
		};

		worker.onerror = (e) => {
			activeWorker = null;
			reject(new Error(e.message || 'Worker error'));
		};

		worker.postMessage({ type: 'prove', input, options });
	});
}

export const KNOWN_CONFIG_KEYS = new Set([
	'description', 'timeout', 'literal_selection', 'age_weight_ratio',
	'max_iterations', 'memory_limit', 'encoder', 'scorer', 'traces',
]);

export function validateConfig(config: Record<string, unknown>): string | null {
	const unknown = Object.keys(config).filter(k => !KNOWN_CONFIG_KEYS.has(k));
	if (unknown.length > 0) {
		return `Unknown config key${unknown.length > 1 ? 's' : ''}: ${unknown.join(', ')}`;
	}
	if (config.encoder && !config.scorer) {
		return 'Missing required key: scorer (required when encoder is set)';
	}
	return null;
}

export const BUILTIN_PRESETS: Record<string, Record<string, unknown>> = {
	time: {
		description: '10s timeout',
		timeout: 10,
		literal_selection: 21,
		age_weight_ratio: 0.5,
	},
	age_weight: {
		description: 'Age-weight baseline (512 iterations)',
		timeout: 600,
		max_iterations: 512,
		literal_selection: 21,
		age_weight_ratio: 0.5,
	},
};
