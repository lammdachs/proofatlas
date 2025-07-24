import init, { ProofAtlasWasm } from './pkg/proofatlas_wasm.js';

// Examples will be loaded dynamically
let EXAMPLES = {};
let exampleMetadata = [];

let prover = null;

async function loadExamples() {
    try {
        // Load examples metadata
        const response = await fetch('./examples/examples.json');
        const data = await response.json();
        exampleMetadata = data.examples;
        
        // Load each example file
        for (const example of exampleMetadata) {
            const fileResponse = await fetch(`./examples/${example.file}`);
            const content = await fileResponse.text();
            EXAMPLES[example.id] = content;
        }
        
        // Populate the dropdown
        const select = document.getElementById('example-select');
        select.innerHTML = '<option value="">Select an example...</option>';
        
        exampleMetadata.forEach(example => {
            const option = document.createElement('option');
            option.value = example.id;
            option.textContent = example.title;
            option.title = example.description;
            select.appendChild(option);
        });
        
        console.log('Examples loaded successfully');
    } catch (error) {
        console.error('Failed to load examples:', error);
    }
}

async function initializeWasm() {
    try {
        await init();
        prover = new ProofAtlasWasm();
        console.log('WASM module loaded successfully');
        document.getElementById('prove-btn').disabled = false;
        
        // Load examples after WASM is ready
        await loadExamples();
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        showError('Failed to initialize the prover. Please refresh the page.');
    }
}

function showResult(result) {
    const resultSection = document.getElementById('result-section');
    const resultStatus = document.getElementById('result-status');
    const resultStats = document.getElementById('result-stats');
    const proofContainer = document.getElementById('proof-container');
    const proofSteps = document.getElementById('proof-steps');
    
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
    
    // Show clauses container if we have any clauses
    const clausesContainer = document.getElementById('clauses-container');
    const clauseList = document.getElementById('clause-list');
    const clausesTitle = document.getElementById('clauses-title');
    
    if ((result.proof && result.proof.length > 0) || (result.all_clauses && result.all_clauses.length > 0)) {
        clausesContainer.classList.remove('hidden');
        
        // Create a set of proof clause IDs for quick lookup
        const proofClauseIds = new Set((result.proof || []).map(step => step.id));
        
        // Set up radio button handlers
        const radioButtons = document.querySelectorAll('input[name="clause-view"]');
        radioButtons.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.value === 'proof') {
                    clausesTitle.textContent = 'Proof';
                    displayClauses(result.proof || [], false);
                } else {
                    clausesTitle.textContent = 'All Clauses';
                    displayClauses(result.all_clauses || [], true, proofClauseIds);
                }
            });
        });
        
        // Show proof by default
        document.querySelector('input[name="clause-view"][value="proof"]').checked = true;
        clausesTitle.textContent = 'Proof';
        displayClauses(result.proof || [], false);
    } else {
        clausesContainer.classList.add('hidden');
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
    
    try {
        // Get options
        const options = {
            timeout_ms: parseInt(document.getElementById('timeout').value),
            max_clauses: parseInt(document.getElementById('max-clauses').value),
            use_superposition: document.getElementById('superposition').checked
        };
        
        // Run prover
        const result = await prover.prove(input, options);
        showResult(result);
        
    } catch (error) {
        showError(error.toString());
    } finally {
        proveBtn.disabled = false;
        proveBtn.textContent = 'Prove';
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeWasm();
    
    document.getElementById('prove-btn').addEventListener('click', prove);
    
    document.getElementById('example-select').addEventListener('change', (e) => {
        const example = EXAMPLES[e.target.value];
        if (example) {
            document.getElementById('tptp-input').value = example;
        }
    });
    
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
});