import init, { ProofAtlasWasm } from './pkg/proofatlas_wasm.js';

// Examples will be loaded dynamically
let exampleMetadata = [];
let exampleContents = {}; // Store example contents by ID

let prover = null;

// Server detection state
let serverAvailable = false;
let serverMlAvailable = false;

// Config presets state
let configPresets = null; // Full config object from proofatlas.json
let activePreset = null;  // Currently selected preset config (null = Custom)

async function detectServer() {
    let detected = false;

    try {
        const response = await fetch('/api/health', { signal: AbortSignal.timeout(2000) });
        if (response.ok) {
            const data = await response.json();
            serverAvailable = true;
            serverMlAvailable = data.ml_available || false;
            detected = true;
            console.log('Server detected, ML available:', serverMlAvailable);

            // Update footer
            const footer = document.getElementById('footer-info');
            if (footer) {
                footer.textContent = 'Connected to local server' +
                    (serverMlAvailable ? ' (ML selectors available)' : ' (heuristic selectors only)') + '.';
            }

            // Update server status indicator
            const status = document.getElementById('server-status');
            if (status) {
                status.textContent = serverMlAvailable ? 'Server + ML' : 'Server (no ML)';
                status.className = 'server-status connected';
            }

            // Show ML hint if ML not available
            if (!serverMlAvailable) {
                const mlHint = document.getElementById('ml-hint');
                if (mlHint) mlHint.classList.remove('hidden');
            }
        }
    } catch {
        // Network error or timeout - no server
    }

    // If server not detected (either 404 or network error), show browser-only UI
    if (!detected) {
        serverAvailable = false;
        serverMlAvailable = false;
        console.log('No server detected, running in browser-only mode');

        const status = document.getElementById('server-status');
        if (status) {
            status.textContent = 'Browser only';
            status.className = 'server-status browser-only';
        }

        // Show ML hint
        const mlHint = document.getElementById('ml-hint');
        if (mlHint) mlHint.classList.remove('hidden');

        // Disable TPTP problem loading
        const loadBtn = document.getElementById('load-url-btn');
        const urlInput = document.getElementById('tptp-url');
        if (loadBtn) {
            loadBtn.disabled = true;
        }
        if (urlInput) {
            urlInput.disabled = true;
            urlInput.placeholder = 'Install locally for TPTP problem loading';
        }
    }
}

// Hardcoded heuristic presets for browser-only mode (no server / config fetch fails)
const BUILTIN_PRESETS = {
    "time": {
        "description": "10s timeout",
        "timeout": 10, "literal_selection": 21, "age_weight_ratio": 0.5
    },
    "age_weight": {
        "description": "Age-weight baseline (512 iterations)",
        "timeout": 600, "max_iterations": 512, "literal_selection": 21, "age_weight_ratio": 0.5
    },
};

async function loadConfigs() {
    let presets = null;

    try {
        // Try fetching from server /configs/ endpoint first, then fall back to relative path
        let response;
        if (serverAvailable) {
            response = await fetch('/configs/proofatlas.json');
        }
        if (!response || !response.ok) {
            response = await fetch('configs/proofatlas.json');
        }
        if (response.ok) {
            configPresets = await response.json();
            presets = configPresets.presets;
        }
    } catch (error) {
        console.log('Failed to fetch configs:', error.message);
    }

    // Fall back to built-in heuristic presets
    if (!presets) {
        presets = BUILTIN_PRESETS;
        configPresets = { presets };
    }

    // Without a server, filter to heuristic-only presets
    if (!serverAvailable) {
        const filtered = {};
        for (const [name, preset] of Object.entries(presets)) {
            if (!preset.encoder) {
                filtered[name] = preset;
            }
        }
        presets = filtered;
        configPresets = { presets };
    }

    const select = document.getElementById('config-preset');
    if (!select) return;

    select.innerHTML = '<option value="">Custom</option>';

    for (const [name, preset] of Object.entries(presets)) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        opt.title = preset.description || '';
        select.appendChild(opt);
    }

    // Handle preset selection
    select.addEventListener('change', (e) => {
        const presetName = e.target.value;
        if (!presetName) {
            activePreset = null;
            return;
        }
        applyPreset(presetName);
    });

    // Default to time preset
    const defaultPreset = 'time';
    if (presets[defaultPreset]) {
        select.value = defaultPreset;
        applyPreset(defaultPreset);
    }

    console.log(`Loaded ${Object.keys(presets).length} presets`);
}

function applyPreset(name) {
    if (!configPresets || !configPresets.presets[name]) return;

    const preset = configPresets.presets[name];
    activePreset = { name, ...preset };

    // Show preset JSON in the textarea
    updateConfigJson(preset);

    console.log('Applied preset:', name, preset);
}

function autoSizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

function updateConfigJson(config) {
    const textarea = document.getElementById('config-json');
    if (textarea) {
        textarea.value = JSON.stringify(config, null, 2);
        autoSizeTextarea(textarea);
    }
}

const KNOWN_CONFIG_KEYS = new Set([
    'description', 'timeout', 'literal_selection', 'age_weight_ratio',
    'max_iterations', 'memory_limit', 'encoder', 'scorer', 'traces',
]);

function validateConfig(config) {
    const unknown = Object.keys(config).filter(k => !KNOWN_CONFIG_KEYS.has(k));
    if (unknown.length > 0) {
        return `Unknown config key${unknown.length > 1 ? 's' : ''}: ${unknown.join(', ')}`;
    }
    if (config.encoder && !config.scorer) {
        return 'Missing required key: scorer (required when encoder is set)';
    }
    return null;
}

function getConfigFromJson() {
    // Parse current JSON textarea content as the active config
    const textarea = document.getElementById('config-json');
    if (!textarea || !textarea.value.trim()) return null;
    try {
        return JSON.parse(textarea.value);
    } catch {
        return null;
    }
}

function resetPresetDropdown() {
    const select = document.getElementById('config-preset');
    if (select && select.value !== '') {
        select.value = '';
        activePreset = null;
    }
}

async function loadExamples() {
    console.log('loadExamples: Function called');
    try {
        // Use fetch API
        console.log('Fetching examples.json with fetch API');
        const response = await fetch('examples/examples.json');

        if (!response.ok) {
            throw new Error(`Failed to load examples.json: ${response.status}`);
        }

        const data = await response.json();
        exampleMetadata = data.examples;
        console.log('Loaded metadata for', exampleMetadata.length, 'examples');

        // Load all example files upfront
        console.log('Loading all example files...');
        for (const example of exampleMetadata) {
            try {
                const fileResponse = await fetch(`examples/${example.file}`);
                if (fileResponse.ok) {
                    const content = await fileResponse.text();
                    exampleContents[example.id] = content;
                }
            } catch (error) {
                console.error(`Error loading ${example.file}:`, error);
            }
        }
        console.log('All example files loaded:', Object.keys(exampleContents));

        // Populate the dropdown
        const select = document.getElementById('example-select');
        if (!select) {
            console.error('Example select element not found!');
            return;
        }

        // Clear and repopulate
        select.innerHTML = '';

        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select an example...';
        select.appendChild(defaultOption);

        // Add example options
        exampleMetadata.forEach((example) => {
            const option = document.createElement('option');
            option.value = example.id;
            option.textContent = example.title;
            option.title = example.description;
            select.appendChild(option);
        });

        // Add event listener to use preloaded examples
        select.addEventListener('change', (e) => {
            const exampleId = e.target.value;
            if (!exampleId) return;

            const content = exampleContents[exampleId];
            if (content) {
                document.getElementById('tptp-input').value = content;
            }
        });

        console.log('Example dropdown populated successfully');
    } catch (error) {
        console.error('Failed to load examples:', error);
        const select = document.getElementById('example-select');
        if (select) {
            select.innerHTML = '<option value="">Examples unavailable (error)</option>';
        }
    }
}


async function initializeWasm() {
    console.log('initializeWasm: Starting...');
    try {
        await init();
        prover = new ProofAtlasWasm();
        console.log('WASM module loaded successfully');
    } catch (error) {
        console.warn('WASM module failed to load:', error);
        if (!serverAvailable) {
            showError('Failed to initialize the browser prover. Please refresh the page.');
        }
        // When server is available, proving works without WASM — just no browser-only mode
    }

    // Load examples regardless of WASM status
    try {
        await loadExamples();
    } catch (loadError) {
        console.error('Error loading examples:', loadError);
    }
}

// Proof Inspector class — iteration-based, matching the paper's given-clause algorithm
class ProofInspector {
    constructor(trace) {
        this.trace = trace;
        this.currentStep = 0;
        this.currentEventIdx = -1; // -1 = pre-iteration state

        // Three clause sets matching the paper
        this.newClauses = new Set();
        this.unprocessedClauses = new Set();
        this.processedClauses = new Set();

        // Build clause lookup from trace events
        this.clauseMap = new Map();
        if (trace && trace.initial_clauses) {
            trace.initial_clauses.forEach(c => this.clauseMap.set(c.id, c));
        }
        if (trace && trace.iterations) {
            for (const iter of trace.iterations) {
                for (const ev of iter.simplification) {
                    if (!this.clauseMap.has(ev.clause_idx)) {
                        this.clauseMap.set(ev.clause_idx, { id: ev.clause_idx, clause: ev.clause, rule: ev.rule, parents: ev.premises });
                    }
                }
                if (iter.selection && !this.clauseMap.has(iter.selection.clause_idx)) {
                    this.clauseMap.set(iter.selection.clause_idx, { id: iter.selection.clause_idx, clause: iter.selection.clause, rule: iter.selection.rule, parents: [] });
                }
                for (const ev of iter.generation) {
                    if (!this.clauseMap.has(ev.clause_idx)) {
                        this.clauseMap.set(ev.clause_idx, { id: ev.clause_idx, clause: ev.clause, rule: ev.rule, parents: ev.premises });
                    }
                }
            }
        }

        this.setupEventHandlers();

        // Start at first iteration
        if (trace && trace.iterations && trace.iterations.length > 0) {
            this.goToStep(0);
        } else {
            this.render();
        }
    }

    setupEventHandlers() {
        // Abort previous listeners if this inspector replaces another
        if (ProofInspector._abortController) {
            ProofInspector._abortController.abort();
        }
        const ac = new AbortController();
        ProofInspector._abortController = ac;
        const opts = { signal: ac.signal };

        document.getElementById('step-first').addEventListener('click', () => this.goToStep(0, 0), opts);
        document.getElementById('step-prev').addEventListener('click', () => this.stepBackward(), opts);
        document.getElementById('step-next').addEventListener('click', () => this.stepForward(), opts);
        document.getElementById('step-last').addEventListener('click', () => {
            const iters = this.trace && this.trace.iterations ? this.trace.iterations : [];
            if (iters.length === 0) return;
            const maxStep = iters.length - 1;
            const lastIdx = this.flattenEvents(iters[maxStep]).length - 1;
            this.goToStep(maxStep, Math.max(0, lastIdx));
        }, opts);

        // Mouse back/forward buttons (buttons 3 & 4) step through events
        document.addEventListener('mouseup', (e) => {
            // Only when inspector view is visible
            const inspector = document.getElementById('proof-inspector');
            if (!inspector || inspector.classList.contains('hidden')) return;

            if (e.button === 3) { // back
                e.preventDefault();
                this.stepBackward();
            } else if (e.button === 4) { // forward
                e.preventDefault();
                this.stepForward();
            }
        }, opts);

        // Prevent default browser back/forward navigation for these buttons
        document.addEventListener('mousedown', (e) => {
            if (e.button === 3 || e.button === 4) {
                const inspector = document.getElementById('proof-inspector');
                if (inspector && !inspector.classList.contains('hidden')) {
                    e.preventDefault();
                }
            }
        }, opts);

        // Keyboard arrow keys step through events
        document.addEventListener('keydown', (e) => {
            const inspector = document.getElementById('proof-inspector');
            if (!inspector || inspector.classList.contains('hidden')) return;
            // Don't capture when typing in an input/textarea
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                this.stepBackward();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                this.stepForward();
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (this.currentStep > 0) {
                    this.goToStep(this.currentStep - 1, 0);
                }
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                const maxStep = this.trace.iterations.length - 1;
                if (this.currentStep < maxStep) {
                    this.goToStep(this.currentStep + 1, 0);
                }
            }
        }, opts);
    }

    /** Step forward one event, wrapping to next iteration. */
    stepForward() {
        if (!this.trace || !this.trace.iterations) return;
        const iter = this.trace.iterations[this.currentStep];
        const lastIdx = this.flattenEvents(iter).length - 1;

        if (this.currentEventIdx < lastIdx) {
            this.goToStep(this.currentStep, this.currentEventIdx + 1);
        } else if (this.currentStep < this.trace.iterations.length - 1) {
            this.goToStep(this.currentStep + 1, 0);
        }
    }

    /** Step backward one event, wrapping to previous iteration's last event. */
    stepBackward() {
        if (!this.trace || !this.trace.iterations) return;

        if (this.currentEventIdx > 0) {
            this.goToStep(this.currentStep, this.currentEventIdx - 1);
        } else if (this.currentStep > 0) {
            const prevIter = this.trace.iterations[this.currentStep - 1];
            const lastIdx = this.flattenEvents(prevIter).length - 1;
            this.goToStep(this.currentStep - 1, lastIdx);
        }
    }

    /** Flatten an iteration's events into a single ordered list. */
    flattenEvents(iter) {
        const events = [];
        for (const ev of iter.simplification) events.push(ev);
        if (iter.selection) events.push(iter.selection);
        for (const ev of iter.generation) events.push(ev);
        return events;
    }

    /** Replay a single event onto the clause sets. */
    replayEvent(ev) {
        switch (ev.rule) {
            case 'TautologyDeletion':
            case 'SubsumptionDeletion':
            case 'DemodulationDeletion':
            case 'ForwardSubsumptionDeletion':
            case 'BackwardSubsumptionDeletion':
            case 'ForwardDemodulation':
                this.newClauses.delete(ev.clause_idx);
                this.unprocessedClauses.delete(ev.clause_idx);
                this.processedClauses.delete(ev.clause_idx);
                break;
            case 'Transfer':
                this.newClauses.delete(ev.clause_idx);
                this.unprocessedClauses.add(ev.clause_idx);
                break;
            case 'Demodulation':
                this.newClauses.add(ev.clause_idx);
                break;
            case 'GivenClauseSelection':
                this.unprocessedClauses.delete(ev.clause_idx);
                this.processedClauses.add(ev.clause_idx);
                break;
            default:
                // Generation rules add to N
                this.newClauses.add(ev.clause_idx);
                break;
        }
    }

    /** Replay all events in an iteration. */
    replayIteration(iter) {
        for (const ev of this.flattenEvents(iter)) {
            this.replayEvent(ev);
        }
    }

    goToStep(iterNum, eventIdx = 0) {
        if (!this.trace || !this.trace.iterations) return;

        const maxStep = this.trace.iterations.length - 1;
        if (iterNum < 0 || iterNum > maxStep) return;

        // Reset and replay from scratch
        this.newClauses.clear();
        this.unprocessedClauses.clear();
        this.processedClauses.clear();

        // Replay all iterations before the target fully
        for (let i = 0; i < iterNum; i++) {
            this.replayIteration(this.trace.iterations[i]);
        }

        // Replay current iteration up to (but not including) eventIdx,
        // so clause sets show the state BEFORE the selected event.
        // eventIdx -1 = pre-iteration state (nothing replayed).
        if (eventIdx > 0) {
            const events = this.flattenEvents(this.trace.iterations[iterNum]);
            for (let i = 0; i < eventIdx && i < events.length; i++) {
                this.replayEvent(events[i]);
            }
        }

        this.currentStep = iterNum;
        this.currentEventIdx = eventIdx;
        this.render();
    }

    render() {
        const iterations = this.trace && this.trace.iterations ? this.trace.iterations : [];
        if (iterations.length === 0) return;

        const maxStep = iterations.length - 1;
        const currentIter = iterations[this.currentStep];

        // Update counter
        document.getElementById('step-counter').textContent = `Iteration ${this.currentStep + 1} / ${maxStep + 1}`;

        // Update buttons
        const isAtStart = this.currentStep <= 0 && this.currentEventIdx <= 0;
        const lastIterEvents = this.flattenEvents(iterations[maxStep]);
        const isAtEnd = this.currentStep >= maxStep && this.currentEventIdx >= lastIterEvents.length - 1;
        document.getElementById('step-first').disabled = isAtStart;
        document.getElementById('step-prev').disabled = isAtStart;
        document.getElementById('step-next').disabled = isAtEnd;
        document.getElementById('step-last').disabled = isAtEnd;

        // Update current event detail
        this.renderCurrentEvent(currentIter);

        // Update phase events
        this.renderPhaseEvents(currentIter);

        // Update clause set displays
        this.renderClauseList('new-clauses', 'new-count', this.newClauses);
        this.renderClauseList('unprocessed-clauses', 'unprocessed-count', this.unprocessedClauses);
        this.renderClauseList('processed-clauses', 'processed-count', this.processedClauses);
    }

    /** Format a clause reference: [id] clause_text */
    formatClauseRef(id) {
        const c = this.clauseMap.get(id);
        const text = c ? escapeHtml(c.clause) : '?';
        return `<span class="clause-ref">[${id}]</span> ${text}`;
    }

    renderCurrentEvent(iter) {
        const container = document.getElementById('current-event-detail');
        if (this.currentEventIdx === -1) {
            container.innerHTML = '';
            container.classList.add('hidden');
            return;
        }

        const flatEvents = this.flattenEvents(iter);
        const ev = flatEvents[this.currentEventIdx];
        if (!ev) {
            container.innerHTML = '';
            container.classList.add('hidden');
            return;
        }

        container.classList.remove('hidden');
        let html = '';

        const rule = ev.rule;
        const isDeletion = ['TautologyDeletion', 'SubsumptionDeletion', 'DemodulationDeletion',
            'ForwardSubsumptionDeletion', 'BackwardSubsumptionDeletion', 'ForwardDemodulation'].includes(rule);
        const isTransfer = rule === 'Transfer';
        const isSelection = rule === 'GivenClauseSelection';
        const isDemodAdd = rule === 'Demodulation';

        // Rule label
        let ruleLabel = rule;
        if (isDeletion) ruleLabel = rule.replace('Deletion', '').replace('Forward', '').replace('Backward', '');
        if (isSelection) ruleLabel = 'Selection';

        html += `<div class="event-detail-rule">${escapeHtml(ruleLabel)}</div>`;
        html += '<table class="event-detail-table"><tbody>';

        if (isDeletion) {
            html += `<tr><td class="event-detail-label">Deleted</td><td>${this.formatClauseRef(ev.clause_idx)}</td></tr>`;
            if (ev.premises && ev.premises.length > 0) {
                html += `<tr><td class="event-detail-label">By</td><td>${ev.premises.map(id => this.formatClauseRef(id)).join('<br>')}</td></tr>`;
            }
        } else if (isTransfer) {
            html += `<tr><td class="event-detail-label">Clause</td><td>${this.formatClauseRef(ev.clause_idx)}</td></tr>`;
            html += `<tr><td class="event-detail-label">Move</td><td>N → U</td></tr>`;
        } else if (isSelection) {
            html += `<tr><td class="event-detail-label">Given</td><td>${this.formatClauseRef(ev.clause_idx)}</td></tr>`;
            html += `<tr><td class="event-detail-label">Move</td><td>U → P</td></tr>`;
        } else if (isDemodAdd) {
            // premises[0] = affected clause, premises[1] = unit equality
            if (ev.premises && ev.premises.length >= 2) {
                html += `<tr><td class="event-detail-label">Rewritten</td><td>${this.formatClauseRef(ev.premises[0])}</td></tr>`;
                html += `<tr><td class="event-detail-label">Using</td><td>${this.formatClauseRef(ev.premises[1])}</td></tr>`;
            }
            html += `<tr><td class="event-detail-label">Result</td><td>${this.formatClauseRef(ev.clause_idx)}</td></tr>`;
        } else {
            // Generation (Resolution, Superposition, Factoring, etc.)
            if (ev.premises && ev.premises.length > 0) {
                html += `<tr><td class="event-detail-label">Premises</td><td>${ev.premises.map(id => this.formatClauseRef(id)).join('<br>')}</td></tr>`;
            }
            html += `<tr><td class="event-detail-label">Result</td><td>${this.formatClauseRef(ev.clause_idx)}</td></tr>`;
        }

        html += '</tbody></table>';
        container.innerHTML = html;
    }

    renderPhaseEvents(iter) {
        const container = document.getElementById('phase-events');
        const summary = document.getElementById('events-summary');
        const flatEvents = this.flattenEvents(iter);

        summary.textContent = `Events (${flatEvents.length})`;

        let html = '';
        let globalIdx = 0;

        // Simplification events
        if (iter.simplification.length > 0) {
            html += '<h5 class="event-phase-label">Simplification</h5>';
            for (const ev of iter.simplification) {
                const active = this.currentEventIdx === globalIdx ? ' active' : '';
                const cssClass = this.getEventCssClass(ev.rule);
                html += `<div class="event-item${active} ${cssClass}" data-event-idx="${globalIdx}">${this.describeEvent(ev)}</div>`;
                globalIdx++;
            }
        }

        // Selection
        if (iter.selection) {
            html += '<h5 class="event-phase-label">Selection</h5>';
            const sel = iter.selection;
            const active = this.currentEventIdx === globalIdx ? ' active' : '';
            html += `<div class="event-item${active} selection" data-event-idx="${globalIdx}">Given clause [${sel.clause_idx}]</div>`;
            globalIdx++;
        }

        // Generation events
        if (iter.generation.length > 0) {
            html += '<h5 class="event-phase-label">Generation</h5>';
            for (const ev of iter.generation) {
                const active = this.currentEventIdx === globalIdx ? ' active' : '';
                const parents = ev.premises.length > 0 ? `([${ev.premises.join(', ')}])` : '';
                html += `<div class="event-item${active} generation" data-event-idx="${globalIdx}">${ev.rule}${parents} → [${ev.clause_idx}]</div>`;
                globalIdx++;
            }
        }

        if (flatEvents.length === 0) {
            html += '<em>No events in this iteration</em>';
        }

        container.innerHTML = html;

        // Attach click handlers to event items
        container.querySelectorAll('.event-item').forEach(el => {
            el.addEventListener('click', () => {
                const idx = parseInt(el.dataset.eventIdx, 10);
                this.goToStep(this.currentStep, idx);
            });
        });
    }

    describeEvent(ev) {
        switch (ev.rule) {
            case 'TautologyDeletion':
                return `Delete [${ev.clause_idx}] (tautology)`;
            case 'SubsumptionDeletion':
            case 'ForwardSubsumptionDeletion':
            case 'BackwardSubsumptionDeletion':
                return `Delete [${ev.clause_idx}] (subsumed)`;
            case 'DemodulationDeletion':
            case 'ForwardDemodulation':
                return `Delete [${ev.clause_idx}] (demodulated)`;
            case 'Transfer':
                return `Transfer [${ev.clause_idx}] N → U`;
            case 'Input':
                return `Input [${ev.clause_idx}] → N`;
            case 'Demodulation':
                return `Demodulate [${ev.premises[0] || '?'}] by [${ev.premises[1] || '?'}] → [${ev.clause_idx}]`;
            default:
                return `${ev.rule} → [${ev.clause_idx}]`;
        }
    }

    getEventCssClass(rule) {
        switch (rule) {
            case 'TautologyDeletion':
            case 'SubsumptionDeletion':
            case 'DemodulationDeletion':
            case 'ForwardSubsumptionDeletion':
            case 'BackwardSubsumptionDeletion':
            case 'ForwardDemodulation':
                return 'deletion';
            case 'Transfer':
                return 'transfer';
            case 'Input':
                return 'input';
            case 'Demodulation':
                return 'simplification';
            default:
                return '';
        }
    }

    renderClauseList(divId, countId, clauseIds) {
        const div = document.getElementById(divId);
        const count = document.getElementById(countId);

        count.textContent = clauseIds.size;

        if (clauseIds.size === 0) {
            div.innerHTML = '';
            return;
        }

        const clauseArray = Array.from(clauseIds).sort((a, b) => a - b);
        div.innerHTML = clauseArray.map(id => {
            const clause = this.clauseMap.get(id);
            if (clause) {
                return `<div class="clause-item">[${id}] ${escapeHtml(clause.clause)}</div>`;
            } else {
                return `<div class="clause-item">[${id}] (not available)</div>`;
            }
        }).join('');
    }
}

let proofInspector = null;

// Profiling renderer
function renderProfile(profile) {
    if (!profile) return '';

    const fmt = (secs) => {
        if (secs === undefined || secs === null) return '-';
        if (secs < 0.001) return `${(secs * 1000000).toFixed(0)}us`;
        if (secs < 1) return `${(secs * 1000).toFixed(1)}ms`;
        return `${secs.toFixed(3)}s`;
    };
    const num = (n) => (n !== undefined && n !== null) ? n.toLocaleString() : '-';

    const table = (rows) => {
        const filtered = rows.filter(([, v]) => v !== undefined && v !== null && v !== '-' && v !== '0' && v !== 0);
        if (filtered.length === 0) return '';
        return '<table class="profile-table">' +
            filtered.map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`).join('') +
            '</table>';
    };

    let html = '';

    // Helper to get generating rule stats
    const genRule = (name) => profile.generating_rules?.[name] || {};
    const simpRule = (name) => profile.simplification_rules?.[name] || {};

    // Phase Timings
    const phasesTracked = (profile.init_time || 0) +
                          (profile.process_new_time || 0) +
                          (profile.select_given_time || 0) +
                          (profile.generate_inferences_time || 0) +
                          (profile.add_inferences_time || 0);
    const totalOverhead = (profile.total_time || 0) - phasesTracked;

    html += '<div class="profile-group"><h4>Phase Timings</h4>';
    const phaseRows = [
        ['Total', fmt(profile.total_time)],
        ['Init (input clauses)', fmt(profile.init_time)],
        ['Process new clauses', fmt(profile.process_new_time)],
        ['  Forward simplification', fmt(profile.forward_simplify_time)],
        ['  Backward simplification', fmt(profile.backward_simplify_time)],
        ['Clause selection', fmt(profile.select_given_time)],
        ['Inference generation', fmt(profile.generate_inferences_time)],
        ['Inference addition', fmt(profile.add_inferences_time)],
    ];
    if (totalOverhead > 0.0001) {
        phaseRows.push(['Other overhead', fmt(totalOverhead)]);
    }
    html += table(phaseRows);
    html += '</div>';

    // Generating Inference Rules
    html += '<div class="profile-group"><h4>Generating Inferences</h4>';
    const genRules = profile.generating_rules || {};
    const genRows = Object.entries(genRules).map(([name, stats]) =>
        [`${name}`, `${num(stats.count)} inferences in ${fmt(stats.time)}`]
    );
    if (genRows.length > 0) {
        html += table(genRows);
    } else {
        html += '<p class="empty-note">No generating inferences recorded</p>';
    }
    html += '</div>';

    // Simplification Rules
    html += '<div class="profile-group"><h4>Simplification Rules</h4>';
    const simpRules = profile.simplification_rules || {};
    const simpRows = [];
    for (const [name, stats] of Object.entries(simpRules)) {
        // Show attempts (total checks) and successes
        if (stats.forward_attempts > 0 || stats.forward_count > 0) {
            const attempts = stats.forward_attempts || stats.forward_count || 0;
            const successes = stats.forward_count || 0;
            const time = stats.forward_attempt_time || stats.forward_time || 0;
            simpRows.push([`${name} (forward)`, `${num(successes)}/${num(attempts)} succeeded in ${fmt(time)}`]);
        }
        if (stats.backward_attempts > 0 || stats.backward_count > 0) {
            const attempts = stats.backward_attempts || stats.backward_count || 0;
            const successes = stats.backward_count || 0;
            const time = stats.backward_attempt_time || stats.backward_time || 0;
            simpRows.push([`${name} (backward)`, `${num(successes)}/${num(attempts)} succeeded in ${fmt(time)}`]);
        }
    }
    if (simpRows.length > 0) {
        html += table(simpRows);
    } else {
        html += '<p class="empty-note">No simplification rules recorded</p>';
    }
    html += '</div>';

    // Clause Lifecycle
    html += '<div class="profile-group"><h4>Clause Lifecycle</h4>';
    html += table([
        ['Iterations', num(profile.iterations)],
        ['Clauses generated', num(profile.clauses_generated)],
        ['Clauses added', num(profile.clauses_added)],
        ['Max unprocessed size', num(profile.max_unprocessed_size)],
        ['Max processed size', num(profile.max_processed_size)],
    ]);
    html += '</div>';

    // Selector Stats
    if (profile.selector_name) {
        html += '<div class="profile-group"><h4>Selector Stats</h4>';
        html += table([
            ['Selector', profile.selector_name],
            ['Cache hits', num(profile.selector_cache_hits)],
            ['Cache misses', num(profile.selector_cache_misses)],
            ['Embed time', fmt(profile.selector_embed_time)],
            ['Score time', fmt(profile.selector_score_time)],
        ]);
        html += '</div>';
    }

    return html;
}

function showResult(result) {
    const resultSection = document.getElementById('result-section');
    const resultStatus = document.getElementById('result-status');
    const resultStats = document.getElementById('result-stats');

    resultSection.classList.remove('hidden');

    // Set status
    resultStatus.className = result.success ? 'success' : (result.status === 'timeout' ? 'timeout' : 'error');
    resultStatus.textContent = result.message;

    // Set statistics
    resultStats.innerHTML = `
        <strong>Statistics:</strong>
        Initial clauses: ${result.statistics.initial_clauses} |
        Generated: ${result.statistics.generated_clauses} |
        Final: ${result.statistics.final_clauses} |
        Time: ${result.statistics.time_ms}ms
    `;

    // Show/hide profiling section
    const profileSection = document.getElementById('profile-section');
    const profileContent = document.getElementById('profile-content');
    if (result.profile) {
        profileContent.innerHTML = renderProfile(result.profile);
        profileSection.classList.remove('hidden');
    } else {
        profileSection.classList.add('hidden');
    }

    // Store result for view switching
    window.currentResult = result;

    // Show clauses container if we have any clauses or trace
    const clausesContainer = document.getElementById('clauses-container');
    const clauseList = document.getElementById('clause-list');
    const clausesTitle = document.getElementById('clauses-title');
    const proofInspectorDiv = document.getElementById('proof-inspector');

    if ((result.proof && result.proof.length > 0) || result.trace) {
        clausesContainer.classList.remove('hidden');

        // Initialize proof inspector if we have trace data
        if (result.trace) {
            proofInspector = new ProofInspector(result.trace);
        }

        // Set up radio button handlers
        const radioButtons = document.querySelectorAll('input[name="clause-view"]');
        radioButtons.forEach(radio => {
            radio.removeEventListener('change', handleClauseViewChange);
            radio.addEventListener('change', handleClauseViewChange);
        });

        // Show proof by default
        document.querySelector('input[name="clause-view"][value="proof"]').checked = true;
        clausesTitle.textContent = 'Proof';
        clauseList.classList.remove('hidden');
        proofInspectorDiv.classList.add('hidden');
        displayClauses(result.proof || [], false);
    } else {
        clausesContainer.classList.add('hidden');
    }
}

function handleClauseViewChange(e) {
    const clauseList = document.getElementById('clause-list');
    const clausesTitle = document.getElementById('clauses-title');
    const proofInspectorDiv = document.getElementById('proof-inspector');
    const result = window.currentResult;

    if (!result) return;

    if (e.target.value === 'proof') {
        clausesTitle.textContent = 'Proof';
        clauseList.classList.remove('hidden');
        proofInspectorDiv.classList.add('hidden');
        displayClauses(result.proof || [], false);
    } else if (e.target.value === 'stepper') {
        clausesTitle.textContent = 'Inspector';
        clauseList.classList.add('hidden');
        proofInspectorDiv.classList.remove('hidden');
        if (proofInspector) {
            proofInspector.render();
        }
        // Blur radio so arrow keys go to document-level handler
        e.target.blur();
    }
}

function displayClauses(clauses, showAll = false, proofClauseIds = null) {
    const clauseList = document.getElementById('clause-list');
    if (clauses.length === 0) {
        clauseList.innerHTML = '<p>No clauses to display</p>';
        return;
    }

    clauseList.innerHTML = clauses.map(step => {
        const isUnused = showAll && proofClauseIds && !proofClauseIds.has(step.id);
        const parents = step.parents.length > 0 ? step.parents.join(', ') : 'none';
        return `
            <div class="proof-step${isUnused ? ' unused' : ''}">
                <div class="proof-step-clause">${escapeHtml(step.clause)}</div>
                <div class="proof-step-info">[${step.id}] ${step.rule} (${parents})</div>
            </div>
        `;
    }).join('');
}

function showError(message) {
    const resultSection = document.getElementById('result-section');
    const resultStatus = document.getElementById('result-status');

    resultSection.classList.remove('hidden');
    resultStatus.className = 'error';
    resultStatus.textContent = message;

    document.getElementById('clauses-container').classList.add('hidden');
    document.getElementById('profile-section').classList.add('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function proveViaServer(input, config) {
    const body = { input, ...config };

    const response = await fetch('/api/prove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Server error: ${response.status}`);
    }

    return await response.json();
}

function configToWasmOptions(config) {
    const opts = { selector_type: 'age_weight' };
    if (config.timeout !== undefined) opts.timeout_ms = config.timeout * 1000;
    if (config.literal_selection !== undefined) opts.literal_selection = String(config.literal_selection);
    if (config.age_weight_ratio !== undefined) opts.age_weight_ratio = config.age_weight_ratio;
    if (config.max_iterations !== undefined) opts.max_iterations = config.max_iterations;
    return opts;
}

async function prove() {
    if (!prover && !serverAvailable) {
        showError('Prover not initialized');
        return;
    }

    const proveBtn = document.getElementById('prove-btn');
    const input = document.getElementById('tptp-input').value.trim();

    if (!input) {
        showError('Please enter a TPTP problem');
        return;
    }

    // Read config from JSON textarea
    const config = getConfigFromJson();
    if (!config) {
        showError('Invalid config JSON');
        return;
    }

    const configError = validateConfig(config);
    if (configError) {
        showError(configError);
        return;
    }

    // Disable button and show loading
    proveBtn.disabled = true;
    proveBtn.textContent = 'Proving...';
    proveBtn.classList.add('loading');

    // Show working indicator in result section
    const resultSection = document.getElementById('result-section');
    const resultStatus = document.getElementById('result-status');
    const resultStats = document.getElementById('result-stats');
    resultSection.classList.remove('hidden');
    resultStatus.className = 'working';
    resultStatus.innerHTML = '<span class="spinner"></span> Solving...';
    resultStats.innerHTML = '';
    document.getElementById('clauses-container').classList.add('hidden');
    document.getElementById('profile-section').classList.add('hidden');

    try {
        // Decide: server-side or WASM
        const useServer = serverAvailable;

        if (useServer) {
            console.log('Proving via server with config:', config);
            const result = await proveViaServer(input, config);
            showResult(result);
        } else {
            // WASM path
            if (!prover) {
                showError('WASM prover not initialized');
                return;
            }

            const options = configToWasmOptions(config);
            console.log('Proving via WASM with options:', options);

            // Allow browser to paint the loading state before blocking computation
            await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));

            // Run prover with trace for inspector
            const result = await prover.prove_with_trace(input, options);
            showResult(result);
        }
    } catch (error) {
        showError(error.toString());
    } finally {
        proveBtn.disabled = false;
        proveBtn.textContent = 'Prove';
        proveBtn.classList.remove('loading');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Detect server first, then load configs, then init WASM
    detectServer().then(() => {
        loadConfigs();
        initializeWasm();
    });

    document.getElementById('prove-btn').addEventListener('click', prove);

    document.getElementById('clear-btn').addEventListener('click', () => {
        document.getElementById('tptp-input').value = '';
        document.getElementById('example-select').value = '';
        document.getElementById('result-section').classList.add('hidden');
        document.getElementById('profile-section').classList.add('hidden');
    });

    // Allow Ctrl+Enter to prove
    document.getElementById('tptp-input').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            prove();
        }
    });

    // Reset preset dropdown when JSON textarea is manually edited, and auto-size
    const configJson = document.getElementById('config-json');
    if (configJson) {
        configJson.addEventListener('input', () => {
            resetPresetDropdown();
            autoSizeTextarea(configJson);
        });
    }

    // Re-measure textarea when details is toggled open (height is 0 while hidden)
    const configDetails = document.getElementById('config-json-details');
    if (configDetails && configJson) {
        configDetails.addEventListener('toggle', () => {
            if (configDetails.open) {
                autoSizeTextarea(configJson);
            }
        });
    }

    // Default config JSON (overridden by loadConfigs when it applies time preset)
    updateConfigJson({
        timeout: 10,
        literal_selection: 21,
        age_weight_ratio: 0.5,
    });

    // Load TPTP problem via server API
    document.getElementById('load-url-btn').addEventListener('click', async () => {
        const urlInput = document.getElementById('tptp-url');
        const input = urlInput.value.trim();
        if (!input) {
            alert('Please enter a problem name (e.g., GRP001-1)');
            return;
        }

        const loadBtn = document.getElementById('load-url-btn');
        loadBtn.disabled = true;
        loadBtn.textContent = 'Loading...';

        try {
            const response = await fetch(`/api/tptp/${encodeURIComponent(input)}`);
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || `Server error: ${response.status}`);
            }
            const data = await response.json();

            document.getElementById('tptp-input').value = data.content;
            document.getElementById('example-select').value = '';
            urlInput.value = '';
        } catch (error) {
            console.error('Error:', error);
            alert(`Error loading problem: ${error.message}`);
        } finally {
            loadBtn.disabled = false;
            loadBtn.textContent = 'Load Problem';
        }
    });

    document.getElementById('tptp-url').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            document.getElementById('load-url-btn').click();
        }
    });

});
