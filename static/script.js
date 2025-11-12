const API_BASE_URL = ''; // The API is on the same domain for Vercel
let modelSchemas = {};
const processData = {};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models/info`);
        modelSchemas = await response.json();
        buildAllDynamicForms();
        showTab('order');
    } catch (error) {
        console.error("Failed to load model schemas", error);
        alert("Could not connect to the backend server. Please ensure it's running.");
    }
});

// --- DYNAMIC FORM BUILDERS ---
function buildAllDynamicForms() {
    // Only build ML-based stages dynamically
    const dynamicStages = ['knitting', 'stenter', 'compactor'];
    dynamicStages.forEach(stage => {
        const schema = modelSchemas[stage];
        const container = document.getElementById(stage);
        if (schema) {
            container.innerHTML = createDynamicFormHTML(stage, schema);
        } else {
            container.innerHTML = `<div class="form-placeholder">Model for '${stage}' is not loaded.</div>`;
        }
    });
}

function createDynamicFormHTML(stage, schema) {
    const toTitleCase = str => str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    const { inputs, outputs, categorical_options } = schema;
    const inputsHTML = inputs.map(key => {
        const label = toTitleCase(key);
        const id = `${stage}-${key.replace(/_/g, '-')}`;
        if (categorical_options && categorical_options[key]) {
            const optionsHTML = categorical_options[key].map(opt => `<option value="${opt}">${opt}</option>`).join('');
            return `<label>${label} <select id="${id}" required><option value="" disabled selected>Select...</option>${optionsHTML}</select></label>`;
        }
        return `<label>${label} <input type="number" step="any" id="${id}" required></label>`;
    }).join('');
    const outputsHTML = outputs.map(key => `<label>${toTitleCase(key)} <input type="text" id="${stage}-sugg-${key.replace(/_/g, '-')}" readonly></label>`).join('');
    return `
        <form id="${stage}-form" class="process-form">
            <div class="inputs"><h3>Inputs</h3>${inputsHTML}</div>
            <div class="suggestions"><h3>Suggestions</h3>${outputsHTML}</div>
        </form>
        <div class="button-container">
            <button class="predict-button" onclick="runPrediction('${stage}')">Get Suggestions</button>
            <button class="next-button" id="${stage}-next-btn" style="display: none;" onclick="goToNextTab('${stage}')">Next &gt;</button>
        </div>
    `;
}

// --- CORE LOGIC ---
async function runPrediction(stage) {
    const loader = document.getElementById('loader');
    loader.style.display = 'flex';
    const form = document.getElementById(`${stage}-form`);
    if (!form || !form.checkValidity()) {
        alert("Please fill in all required input fields for this stage.");
        loader.style.display = 'none';
        return;
    }
    const payload = {};
    form.querySelectorAll('input, select').forEach(el => {
        if (!el.readOnly && el.id) {
            const key = el.id.split('-').slice(1).join('_');
            payload[key] = (el.type === 'number') ? parseFloat(el.value) : el.value;
        }
    });
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict/${stage}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) { const err = await response.json(); throw new Error(err.message); }
        const result = await response.json();
        populateSuggestions(stage, result.predictions);
        chainToNextStage();
        await logCurrentState();
        const nextBtn = document.getElementById(`${stage}-next-btn`);
        if (nextBtn) nextBtn.style.display = 'inline-block';
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loader.style.display = 'none';
    }
}

function populateSuggestions(stage, predictions) {
    for (const [key, value] of Object.entries(predictions)) {
        const suggId = `${stage}-sugg-${key.replace(/_/g, '-')}`;
        const element = document.getElementById(suggId);
        if (element) element.value = value;
    }
}

function chainToNextStage() {
    const safeSetValue = (destId, sourceId) => {
        const destElement = document.getElementById(destId);
        const sourceElement = document.getElementById(sourceId);
        if (destElement && sourceElement) {
            destElement.value = sourceElement.value;
        }
    };
    
    // Order -> Knitting
    safeSetValue('knitting-target-gsm', 'order-sugg-gray-gsm');
    safeSetValue('knitting-target-dia', 'order-sugg-gray-dia');
    
    // Dyeing -> Stenter
    safeSetValue('stenter-dyed-gsm', 'dyeing-sugg-dyed-gsm');
    safeSetValue('stenter-dyed-dia', 'dyeing-sugg-dyed-dia');
    
    // Carry over final targets & batch number
    safeSetValue('stenter-target-gsm', 'order-req-gsm');
    safeSetValue('stenter-target-dia', 'order-req-dia');
    safeSetValue('compactor-target-gsm', 'order-req-gsm');
    safeSetValue('compactor-target-dia', 'order-req-dia');
    safeSetValue('feedback-batch-no', 'order-batch-no');
}

function goToNextTab(currentStage) {
    const pipeline = ['order', 'knitting', 'dyeing', 'stenter', 'compactor', 'feedback'];
    const currentIndex = pipeline.indexOf(currentStage);
    if (currentIndex < pipeline.length - 1) {
        showTab(pipeline[currentIndex + 1]);
    }
}

// --- DATA MANAGEMENT & UI HELPERS ---
async function submitFeedback() {
    await logCurrentState(true);
    alert(`Feedback for Batch No. ${document.getElementById('order-batch-no')?.value} saved!`);
    showTab('history');
}
async function logCurrentState(isFinal = false) {
    const logPayload = {};
    document.querySelectorAll('form input, form select').forEach(el => {
        if (el.id) {
            const key = el.id.replace(/-/g, '_');
            logPayload[key] = el.value;
        }
    });
    const finalPayload = { batch_no: logPayload.order_batch_no };
    for (const [key, value] of Object.entries(logPayload)) {
        const cleanKey = key.split('_').slice(1).join('_');
        finalPayload[cleanKey] = value;
    }
    await fetch(`${API_BASE_URL}/api/log`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(finalPayload) });
}
async function loadHistory() {
    const response = await fetch(`${API_BASE_URL}/api/history`); const data = await response.json();
    const tableHead = document.querySelector('#history-table thead'); const tableBody = document.querySelector('#history-table tbody');
    tableHead.innerHTML = ''; tableBody.innerHTML = '';
    if (data.length === 0) { tableBody.innerHTML = '<tr><td colspan="100%">No history.</td></tr>'; return; }
    const headers = Object.keys(data[0]); const headerRow = document.createElement('tr');
    headers.forEach(h => { const th = document.createElement('th'); th.textContent = h.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()); headerRow.appendChild(th); });
    tableHead.appendChild(headerRow);
    data.forEach(row => { const tr = document.createElement('tr'); headers.forEach(h => { const td = document.createElement('td'); td.textContent = row[h]; tr.appendChild(td); }); tableBody.appendChild(tr); });
}
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    document.querySelector(`.tab-button[onclick="showTab('${tabName}')"]`).classList.add('active');
    if (tabName === 'history') loadHistory();
}