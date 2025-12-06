import init, { ProofAtlasWasm } from './pkg/proofatlas_wasm.js';

// Examples will be loaded dynamically
let exampleMetadata = [];
let exampleContents = {}; // Store example contents by ID

let prover = null;

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
                console.log(`Loading ${example.file}...`);
                const fileResponse = await fetch(`examples/${example.file}`);
                if (fileResponse.ok) {
                    const content = await fileResponse.text();
                    exampleContents[example.id] = content;
                    console.log(`Loaded ${example.id}: ${content.length} characters`);
                } else {
                    console.error(`Failed to load ${example.file}: ${fileResponse.status}`);
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
        
        console.log('Populating dropdown with', exampleMetadata.length, 'examples');
        console.log('Select element found:', select);
        console.log('Current innerHTML before clear:', select.innerHTML);
        
        // Clear and repopulate
        select.innerHTML = '';
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select an example...';
        select.appendChild(defaultOption);
        console.log('Added default option');
        
        // Add example options
        exampleMetadata.forEach((example, index) => {
            const option = document.createElement('option');
            option.value = example.id;
            option.textContent = example.title;
            option.title = example.description;
            select.appendChild(option);
            console.log(`Added option ${index + 1}:`, example.id, example.title);
        });
        
        console.log('Final innerHTML:', select.innerHTML);
        console.log('Total options in select:', select.options.length);
        
        // Add event listener to use preloaded examples
        select.addEventListener('change', (e) => {
            const exampleId = e.target.value;
            if (!exampleId) return;
            
            console.log('Selecting example:', exampleId);
            
            // Use preloaded content
            const content = exampleContents[exampleId];
            if (content) {
                document.getElementById('tptp-input').value = content;
                console.log('Successfully set example:', exampleId);
            } else {
                console.error('Example content not found:', exampleId);
            }
        });
        
        console.log('Example dropdown populated successfully');
    } catch (error) {
        console.error('Failed to load examples:', error);
        console.error('Error type:', error.constructor.name);
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);
        // Still make the prover functional even if examples fail to load
        const select = document.getElementById('example-select');
        if (select) {
            select.innerHTML = '<option value="">Examples unavailable (error)</option>';
        } else {
            console.error('Select element not found even in error handler');
        }
    }
    console.log('loadExamples: Function completed');
}


async function initializeWasm() {
    console.log('initializeWasm: Starting...');
    try {
        console.log('initializeWasm: About to init WASM');
        await init();
        prover = new ProofAtlasWasm();
        console.log('WASM module loaded successfully');
        document.getElementById('prove-btn').disabled = false;

        // Small delay to ensure DOM is ready
        console.log('initializeWasm: Waiting for DOM...');
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Load examples after WASM is ready
        console.log('initializeWasm: About to call loadExamples');
        try {
            await loadExamples();
            console.log('initializeWasm: loadExamples completed');
        } catch (loadError) {
            console.error('initializeWasm: Error loading examples:', loadError);
            // Don't fail the entire initialization
        }
        
        // Force a reflow of the select element
        const select = document.getElementById('example-select');
        if (select) {
            select.style.display = 'none';
            select.offsetHeight; // Force reflow
            select.style.display = '';
            console.log('Forced reflow of select element');
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
        let resolutionCount = 0;
        
        for (const step of this.trace.saturation_steps) {
            if (step.step_type === 'given_selection') {
                // Start a new group for this given clause
                currentGroup = {
                    givenClauseId: step.clause_idx,
                    givenClause: step.clause,
                    inferences: []
                };
                this.givenClauseGroups.push(currentGroup);
            } else if (step.step_type === 'inference' && currentGroup) {
                // Add this inference to the current group
                currentGroup.inferences.push(step);
                
                // Debug: count resolutions
                if (step.rule === 'Resolution') {
                    resolutionCount++;
                    console.log(`Resolution #${resolutionCount}: clause ${step.clause_idx} from [${step.premises}]`);
                }
            }
        }
        
        console.log(`Total Resolution steps in trace: ${resolutionCount}`);
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
                    // The premises already include the given clause, don't add it again
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
        
        // Update given clause - show the actual clause being processed
        const givenDiv = document.getElementById('given-clause');
        if (currentGroup) {
            const clause = this.getClauseById(currentGroup.givenClauseId);
            if (clause) {
                givenDiv.innerHTML = `<strong>[${currentGroup.givenClauseId}]</strong> ${escapeHtml(clause.clause)}`;
            } else if (currentGroup.givenClause) {
                // Fallback to clause from the group if available
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
        // Look in all clauses
        if (this.allClauses) {
            return this.allClauses.find(c => c.id === id);
        }
        
        // Fallback to trace data
        if (this.trace) {
            // Check initial clauses
            const initial = this.trace.initial_clauses?.find(c => c.id === id);
            if (initial) return initial;
            
            // Check saturation steps
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
            radio.removeEventListener('change', handleClauseViewChange); // Remove old listeners
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
        } else {
            console.error('ProofInspector not initialized - no trace data available');
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
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function prove() {
    if (!prover) {
        showError('Prover not initialized');
        return;
    }

    const proveBtn = document.getElementById('prove-btn');
    const input = document.getElementById('tptp-input').value.trim();

    if (!input) {
        showError('Please enter a TPTP problem');
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

    try {
        // Get options - uses age_weight selector by default
        const options = {
            timeout_ms: parseInt(document.getElementById('timeout').value),
            max_clauses: parseInt(document.getElementById('max-clauses').value),
            literal_selection: document.getElementById('literal-selection').value,
            selector_type: 'age_weight',
            age_weight_ratio: 0.167
        };

        console.log('Using age-weight clause selection');

        // Allow browser to paint the loading state before blocking computation
        await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));

        // Run prover with trace for inspector
        const result = await prover.prove_with_trace(input, options);
        showResult(result);
        
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
    initializeWasm();
    
    document.getElementById('prove-btn').addEventListener('click', prove);
    
    document.getElementById('clear-btn').addEventListener('click', () => {
        document.getElementById('tptp-input').value = '';
        document.getElementById('example-select').value = '';
        document.getElementById('result-section').classList.add('hidden');
    });
    
    // Allow Ctrl+Enter to prove
    document.getElementById('tptp-input').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            prove();
        }
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
        // Match include directives like: include('Axioms/GRP004-0.ax')
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
                // Construct TPTP URL for the axiom file
                // Path format: 'Axioms/GRP004-0.ax' -> Category=Axioms, File=GRP004-0.ax
                const parts = includePath.split('/');
                const filename = parts[parts.length - 1];

                const axiomUrl = `https://tptp.org/cgi-bin/SeeTPTP?Category=Axioms&File=${filename}`;
                console.log(`Fetching axiom: ${includePath}`);

                const axiomContent = await fetchTptpContent(axiomUrl);

                // Replace the include directive with the actual axiom content
                // Add comments to mark the included content
                const replacement = `% BEGIN include('${includePath}')\n${axiomContent}\n% END include('${includePath}')`;
                resolvedContent = resolvedContent.replace(includeDirective, replacement);

                console.log(`✓ Loaded axiom: ${includePath}`);
            } catch (error) {
                console.warn(`Failed to load axiom ${includePath}:`, error);
                // Leave the include directive in place if we can't fetch it
            }
        }

        return resolvedContent;
    }

    // Helper function to convert problem name to TPTP URL
    function problemNameToUrl(input) {
        // If it's already a full URL, return as-is
        if (input.startsWith('http://') || input.startsWith('https://')) {
            return input;
        }

        // Extract problem name (e.g., "GRP001-1" or "grp001-1" or "GRP001-1.p")
        let problemName = input.trim().toUpperCase();

        // Remove .P extension if present
        if (problemName.endsWith('.P')) {
            problemName = problemName.slice(0, -2);
        }

        // Extract domain (first 3 letters, case-insensitive)
        const domainMatch = problemName.match(/^([A-Z]{3})/);
        if (!domainMatch) {
            throw new Error('Invalid problem name format. Expected format: ABC123-1 (e.g., GRP001-1)');
        }

        const domain = domainMatch[1];
        const filename = `${problemName}.p`;

        // Construct TPTP URL
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
            // Convert problem name to URL if needed
            const url = problemNameToUrl(input);
            console.log('Loading problem from:', url);

            // Fetch the main problem file
            const content = await fetchTptpContent(url);

            // Resolve any include directives
            const resolvedContent = await resolveIncludes(content);

            // Load into textarea
            document.getElementById('tptp-input').value = resolvedContent;
            document.getElementById('example-select').value = '';
            urlInput.value = '';

            console.log('✓ Problem loaded successfully');
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