// Web Worker that runs the WASM prover off the main thread.
// Receives { type: 'prove', input, options } messages.
// Posts back { type: 'result', result } or { type: 'error', message }.

let wasm = null;

async function initWasm() {
	if (wasm) return wasm;
	const mod = await import('/pkg/proofatlas_wasm.js');
	await mod.default();
	wasm = new mod.ProofAtlasWasm();
	return wasm;
}

self.onmessage = async (e) => {
	const { type, input, options } = e.data;
	if (type !== 'prove') return;

	try {
		const prover = await initWasm();
		const result = prover.prove_with_trace(input, options);
		self.postMessage({ type: 'result', result });
	} catch (err) {
		self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
	}
};
