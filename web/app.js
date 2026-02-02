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
    try {
        const response = await fetch('/api/health', { signal: AbortSignal.timeout(2000) });
        if (response.ok) {
            const data = await response.json();
            serverAvailable = true;
            serverMlAvailable = data.ml_available || false;
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
        }
    } catch {
        serverAvailable = false;
        serverMlAvailable = false;
        console.log('No server detected, running in browser-only mode');

        const status = document.getElementById('server-status');
        if (status) {
            status.textContent = 'Browser only';
            status.className = 'server-status browser-only';
        }
    }
}

// Hardcoded heuristic presets for browser-only mode (no server / config fetch fails)
const BUILTIN_PRESETS = {
    "time_sel0": {
        "description": "10s with sel0 (all literals)",
        "timeout": 10, "literal_selection": 0, "age_weight_ratio": 0.167
    },
    "time_sel20": {
        "description": "10s with sel20 (all maximal literals)",
        "timeout": 10, "literal_selection": 20, "age_weight_ratio": 0.167
    },
    "time_sel21": {
        "description": "10s with sel21 (unique maximal, else neg max-weight, else all maximal)",
        "timeout": 10, "literal_selection": 21, "age_weight_ratio": 0.167
    },
    "age_weight_sel0": {
        "description": "Age-weight baseline (128 iterations) with sel0",
        "timeout": 600, "max_iterations": 128, "literal_selection": 0, "age_weight_ratio": 0.167
    },
    "age_weight_sel20": {
        "description": "Age-weight baseline (128 iterations) with sel20",
        "timeout": 600, "max_iterations": 128, "literal_selection": 20, "age_weight_ratio": 0.167
    },
    "age_weight_sel21": {
        "description": "Age-weight baseline (128 iterations) with sel21",
        "timeout": 600, "max_iterations": 128, "literal_selection": 21, "age_weight_ratio": 0.167
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

    // Default to time_sel21
    const defaultPreset = 'time_sel21';
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
    'max_iterations', 'encoder', 'scorer', 'traces',
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
        document.getElementById('prove-btn').disabled = false;

        // Small delay to ensure DOM is ready
        await new Promise(resolve => setTimeout(resolve, 100));

        // Load examples after WASM is ready
        try {
            await loadExamples();
        } catch (loadError) {
            console.error('Error loading examples:', loadError);
        }

        // Force a reflow of the select element
        const select = document.getElementById('example-select');
        if (select) {
            select.style.display = 'none';
            select.offsetHeight; // Force reflow
            select.style.display = '';
        }
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        showError('Failed to initialize the prover. Please refresh the page.');
    }
}

// Proof Inspector class
class ProofInspector {
    constructor(trace, allClauses) {
        this.trace = trace;
        this.allClauses = allClauses;
        this.currentStep = 0;
        this.processedClauses = new Set();
        this.unprocessedClauses = new Set();
        this.currentGiven = null;

        // Group steps by given clause
        this.givenClauseGroups = [];
        this.buildGivenClauseGroups();

        // Initialize with initial clauses
        if (trace && trace.initial_clauses) {
            trace.initial_clauses.forEach(clause => {
                this.unprocessedClauses.add(clause.id);
            });
        }

        this.setupEventHandlers();
        this.render();
    }

    buildGivenClauseGroups() {
        if (!this.trace || !this.trace.saturation_steps) return;

        let currentGroup = null;

        for (const step of this.trace.saturation_steps) {
            if (step.step_type === 'given_selection') {
                currentGroup = {
                    givenClauseId: step.clause_idx,
                    givenClause: step.clause,
                    inferences: []
                };
                this.givenClauseGroups.push(currentGroup);
            } else if (step.step_type === 'inference' && currentGroup) {
                currentGroup.inferences.push(step);
            }
        }

        console.log(`Total given clause groups: ${this.givenClauseGroups.length}`);
    }

    setupEventHandlers() {
        document.getElementById('step-first').addEventListener('click', () => this.goToStep(0));
        document.getElementById('step-prev').addEventListener('click', () => this.goToStep(this.currentStep - 1));
        document.getElementById('step-next').addEventListener('click', () => this.goToStep(this.currentStep + 1));
        document.getElementById('step-last').addEventListener('click', () => {
            const maxStep = this.givenClauseGroups ? this.givenClauseGroups.length - 1 : 0;
            this.goToStep(maxStep);
        });
    }

    goToStep(groupNum) {
        if (!this.givenClauseGroups) return;

        const maxStep = this.givenClauseGroups.length - 1;
        if (groupNum < 0 || groupNum > maxStep) return;

        // Reset state and replay up to the target group
        this.processedClauses.clear();
        this.unprocessedClauses.clear();
        this.currentGiven = null;

        // Add initial clauses to unprocessed
        if (this.trace && this.trace.initial_clauses) {
            this.trace.initial_clauses.forEach(clause => {
                this.unprocessedClauses.add(clause.id);
            });
        }

        // Process all groups up to and including the current one
        for (let i = 0; i <= groupNum; i++) {
            const group = this.givenClauseGroups[i];

            // Move previous given to processed
            if (this.currentGiven !== null) {
                this.processedClauses.add(this.currentGiven);
            }

            // Set current given clause
            this.unprocessedClauses.delete(group.givenClauseId);
            this.currentGiven = group.givenClauseId;

            // If this is the current group, add all its inferences to unprocessed
            if (i === groupNum) {
                group.inferences.forEach(inf => {
                    this.unprocessedClauses.add(inf.clause_idx);
                });
            } else {
                // For previous groups, add all inferences to unprocessed then move given to processed
                group.inferences.forEach(inf => {
                    this.unprocessedClauses.add(inf.clause_idx);
                });
                this.processedClauses.add(this.currentGiven);
                this.currentGiven = null;
            }
        }

        this.currentStep = groupNum;
        this.render();
    }

    render() {
        if (!this.givenClauseGroups || this.givenClauseGroups.length === 0) return;

        const maxStep = this.givenClauseGroups.length - 1;
        const currentGroup = this.currentStep >= 0 ? this.givenClauseGroups[this.currentStep] : null;

        // Update counter
        document.getElementById('step-counter').textContent = `Given Clause ${this.currentStep + 1} / ${maxStep + 1}`;

        // Update buttons
        document.getElementById('step-first').disabled = this.currentStep <= 0;
        document.getElementById('step-prev').disabled = this.currentStep <= 0;
        document.getElementById('step-next').disabled = this.currentStep >= maxStep;
        document.getElementById('step-last').disabled = this.currentStep >= maxStep;

        // Update step info with all inferences for this given clause
        const stepInfo = document.getElementById('step-info');
        if (currentGroup) {
            let infoHtml = '';

            if (currentGroup.inferences.length === 0) {
                infoHtml = '<em>No inferences generated from this clause</em>';
            } else {
                infoHtml = `<strong>${currentGroup.inferences.length} inference(s) generated:</strong><div class="inference-list">`;
                currentGroup.inferences.forEach(inf => {
                    const parents = inf.premises && inf.premises.length > 0 ? inf.premises.join(', ') : '';
                    if (parents) {
                        infoHtml += `<div class="inference-item">${inf.rule} with [${parents}] → Clause ${inf.clause_idx}</div>`;
                    } else {
                        infoHtml += `<div class="inference-item">${inf.rule} → Clause ${inf.clause_idx}</div>`;
                    }
                });
                infoHtml += '</div>';
            }
            stepInfo.innerHTML = infoHtml;
        }

        // Update clause lists
        this.renderClauseList('processed-clauses', 'processed-count', this.processedClauses);
        this.renderClauseList('unprocessed-clauses', 'unprocessed-count', this.unprocessedClauses);

        // Update given clause
        const givenDiv = document.getElementById('given-clause');
        if (currentGroup) {
            const clause = this.getClauseById(currentGroup.givenClauseId);
            if (clause) {
                givenDiv.innerHTML = `<strong>[${currentGroup.givenClauseId}]</strong> ${escapeHtml(clause.clause)}`;
            } else if (currentGroup.givenClause) {
                givenDiv.innerHTML = `<strong>[${currentGroup.givenClauseId}]</strong> ${escapeHtml(currentGroup.givenClause)}`;
            } else {
                givenDiv.innerHTML = `<strong>[${currentGroup.givenClauseId}]</strong> (clause details not available)`;
            }
        } else {
            givenDiv.textContent = '(none selected)';
        }
    }

    renderClauseList(divId, countId, clauseIds) {
        const div = document.getElementById(divId);
        const count = document.getElementById(countId);

        count.textContent = clauseIds.size;

        if (clauseIds.size === 0) {
            div.innerHTML = '<div class="clause-item">(empty)</div>';
            return;
        }

        const clauseArray = Array.from(clauseIds).sort((a, b) => a - b);
        div.innerHTML = clauseArray.map(id => {
            const clause = this.getClauseById(id);
            if (clause) {
                return `<div class="clause-item">[${id}] ${escapeHtml(clause.clause)}</div>`;
            } else {
                return `<div class="clause-item">[${id}] (not available)</div>`;
            }
        }).join('');
    }

    getClauseById(id) {
        if (this.allClauses) {
            return this.allClauses.find(c => c.id === id);
        }

        if (this.trace) {
            const initial = this.trace.initial_clauses?.find(c => c.id === id);
            if (initial) return initial;

            const step = this.trace.saturation_steps?.find(s => s.clause_idx === id);
            if (step) {
                return {
                    id: step.clause_idx,
                    clause: step.clause,
                    rule: step.rule,
                    parents: step.premises || []
                };
            }
        }

        return null;
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

    // Phase Timings
    html += '<div class="profile-group"><h4>Phase Timings</h4>';
    html += table([
        ['Total', fmt(profile.total_time)],
        ['Forward simplification', fmt(profile.forward_simplify_time)],
        ['  Forward demodulation', fmt(profile.forward_demod_time)],
        ['  Forward subsumption', fmt(profile.forward_subsumption_time)],
        ['  Backward subsumption', fmt(profile.backward_subsumption_time)],
        ['  Backward demodulation', fmt(profile.backward_demod_time)],
        ['Clause selection', fmt(profile.select_given_time)],
        ['Inference generation', fmt(profile.generate_inferences_time)],
        ['  Resolution', fmt(profile.resolution_time)],
        ['  Superposition', fmt(profile.superposition_time)],
        ['  Factoring', fmt(profile.factoring_time)],
        ['  Equality resolution', fmt(profile.equality_resolution_time)],
        ['  Equality factoring', fmt(profile.equality_factoring_time)],
        ['Inference addition', fmt(profile.add_inferences_time)],
    ]);
    html += '</div>';

    // Inference Counts
    html += '<div class="profile-group"><h4>Inference Counts</h4>';
    html += table([
        ['Resolution', num(profile.resolution_count)],
        ['Superposition', num(profile.superposition_count)],
        ['Factoring', num(profile.factoring_count)],
        ['Equality resolution', num(profile.equality_resolution_count)],
        ['Equality factoring', num(profile.equality_factoring_count)],
        ['Demodulation', num(profile.demodulation_count)],
    ]);
    html += '</div>';

    // Clause Lifecycle
    html += '<div class="profile-group"><h4>Clause Lifecycle</h4>';
    html += table([
        ['Iterations', num(profile.iterations)],
        ['Clauses generated', num(profile.clauses_generated)],
        ['Clauses added', num(profile.clauses_added)],
        ['Subsumed (forward)', num(profile.clauses_subsumed_forward)],
        ['Subsumed (backward)', num(profile.clauses_subsumed_backward)],
        ['Demodulated (forward)', num(profile.clauses_demodulated_forward)],
        ['Demodulated (backward)', num(profile.clauses_demodulated_backward)],
        ['Tautologies deleted', num(profile.tautologies_deleted)],
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

    if ((result.proof && result.proof.length > 0) || (result.all_clauses && result.all_clauses.length > 0) || result.trace) {
        clausesContainer.classList.remove('hidden');

        // Initialize proof inspector if we have trace data
        if (result.trace) {
            proofInspector = new ProofInspector(result.trace, result.all_clauses);
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
    } else if (e.target.value === 'all') {
        const proofClauseIds = new Set((result.proof || []).map(step => step.id));
        clausesTitle.textContent = 'All Clauses';
        clauseList.classList.remove('hidden');
        proofInspectorDiv.classList.add('hidden');
        displayClauses(result.all_clauses || [], true, proofClauseIds);
    } else if (e.target.value === 'stepper') {
        clausesTitle.textContent = 'Inspector';
        clauseList.classList.add('hidden');
        proofInspectorDiv.classList.remove('hidden');
        if (proofInspector) {
            proofInspector.render();
        }
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

    // Default config JSON (overridden by loadConfigs when it applies time_sel21)
    updateConfigJson({
        timeout: 10,
        literal_selection: 21,
        age_weight_ratio: 0.167,
    });

    // Helper function to fetch TPTP content from URL
    async function fetchTptpContent(url) {
        const corsUrl = `https://corsproxy.io/?${encodeURIComponent(url)}`;
        const response = await fetch(corsUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch: ${response.status}`);
        }
        const html = await response.text();

        // Parse HTML to extract TPTP content from <pre> tag
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const preTag = doc.querySelector('pre');

        if (!preTag) {
            throw new Error('Could not find problem content in page');
        }

        return preTag.textContent;
    }

    // Helper function to resolve include directives
    async function resolveIncludes(content) {
        const includeRegex = /include\s*\(\s*'([^']+)'\s*\)/g;
        const includes = [...content.matchAll(includeRegex)];

        if (includes.length === 0) {
            return content;
        }

        console.log(`Found ${includes.length} include directive(s), fetching axioms...`);

        let resolvedContent = content;

        for (const match of includes) {
            const includePath = match[1];
            const includeDirective = match[0];

            try {
                const parts = includePath.split('/');
                const filename = parts[parts.length - 1];

                const axiomUrl = `https://tptp.org/cgi-bin/SeeTPTP?Category=Axioms&File=${filename}`;
                console.log(`Fetching axiom: ${includePath}`);

                const axiomContent = await fetchTptpContent(axiomUrl);

                const replacement = `% BEGIN include('${includePath}')\n${axiomContent}\n% END include('${includePath}')`;
                resolvedContent = resolvedContent.replace(includeDirective, replacement);
            } catch (error) {
                console.warn(`Failed to load axiom ${includePath}:`, error);
            }
        }

        return resolvedContent;
    }

    // Helper function to convert problem name to TPTP URL
    function problemNameToUrl(input) {
        if (input.startsWith('http://') || input.startsWith('https://')) {
            return input;
        }

        let problemName = input.trim().toUpperCase();

        if (problemName.endsWith('.P')) {
            problemName = problemName.slice(0, -2);
        }

        const domainMatch = problemName.match(/^([A-Z]{3})/);
        if (!domainMatch) {
            throw new Error('Invalid problem name format. Expected format: ABC123-1 (e.g., GRP001-1)');
        }

        const domain = domainMatch[1];
        const filename = `${problemName}.p`;

        return `https://tptp.org/cgi-bin/SeeTPTP?Category=Problems&Domain=${domain}&File=${filename}`;
    }

    // Load from TPTP problem name or URL
    document.getElementById('load-url-btn').addEventListener('click', async () => {
        const urlInput = document.getElementById('tptp-url');
        const input = urlInput.value.trim();
        if (!input) {
            alert('Please enter a problem name (e.g., GRP001-1) or TPTP URL');
            return;
        }
        try {
            const url = problemNameToUrl(input);
            console.log('Loading problem from:', url);

            const content = await fetchTptpContent(url);
            const resolvedContent = await resolveIncludes(content);

            document.getElementById('tptp-input').value = resolvedContent;
            document.getElementById('example-select').value = '';
            urlInput.value = '';
        } catch (error) {
            console.error('Error:', error);
            alert(`Error loading problem: ${error.message}`);
        }
    });

    document.getElementById('tptp-url').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            document.getElementById('load-url-btn').click();
        }
    });

});
