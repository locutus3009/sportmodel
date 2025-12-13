// SportModel - Training Analytics Dashboard

const MOVEMENTS = ['squat', 'bench', 'deadlift', 'snatch', 'cj', 'bodyweight'];
const COMPOSITES = ['ipf-gl', 'sinclair'];
const BODY_COMPOSITION = ['bodyfat', 'lbm'];

// Chart instance
let chart = null;

// Data cache
const dataCache = {};

// Current tab
let currentTab = 'squat';

// Current chart data for tooltip lookup
let tooltipData = null;

// Chart range settings (defaults)
let historyYears = 2;
let predictionMonths = 12;

// Chart colors
const COLORS = {
    observation: '#4CAF50',
    prediction: '#2196F3',
    // Sigma bands: innermost (1σ) is most visible, outermost (3σ) is most transparent
    sigma1Area: 'rgba(33, 150, 243, 0.30)',  // 1σ — darkest
    sigma2Area: 'rgba(33, 150, 243, 0.18)',  // 2σ — medium
    sigma3Area: 'rgba(33, 150, 243, 0.08)',  // 3σ — lightest
    todayLine: 'rgba(255, 255, 255, 0.3)',
    gridLine: 'rgba(255, 255, 255, 0.1)',
    text: '#b0b0b0',
};

// WebSocket state
let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;
const RECONNECT_DELAY_MS = 2000;

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

function init() {
    console.log('SportModel initializing...');
    setupTabs();
    setupRangeControls();
    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);
    connectWebSocket();
    loadTdee();
}

function setupRangeControls() {
    const historyButtons = document.getElementById('history-buttons');
    const predictionButtons = document.getElementById('prediction-buttons');

    historyButtons.addEventListener('click', (e) => {
        const btn = e.target.closest('.option-btn');
        if (!btn || btn.classList.contains('active')) return;

        historyButtons.querySelector('.active').classList.remove('active');
        btn.classList.add('active');
        historyYears = parseInt(btn.dataset.value, 10);
        clearCacheForCurrentTab();
        reloadCurrentTab();
    });

    predictionButtons.addEventListener('click', (e) => {
        const btn = e.target.closest('.option-btn');
        if (!btn || btn.classList.contains('active')) return;

        predictionButtons.querySelector('.active').classList.remove('active');
        btn.classList.add('active');
        predictionMonths = parseInt(btn.dataset.value, 10);
        clearCacheForCurrentTab();
        reloadCurrentTab();
    });
}

function clearCacheForCurrentTab() {
    // Clear all cache entries for current tab (any params combination)
    for (const key in dataCache) {
        if (key.startsWith(currentTab + '_') || key.startsWith('composites_')) {
            delete dataCache[key];
        }
    }
}

// === WebSocket Connection ===

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log('Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
        showConnectionStatus('connected');
    };

    ws.onmessage = (event) => {
        console.log('WebSocket message:', event.data);

        if (event.data === 'reload') {
            showNotification('Data updated');
            clearCache();
            reloadCurrentTab();
            loadTdee();
        } else if (event.data.startsWith('error:')) {
            const errorMsg = event.data.substring(6);
            showNotification('Error: ' + errorMsg, true);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        showConnectionStatus('disconnected');
        scheduleReconnect();
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function scheduleReconnect() {
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts++;
        showConnectionStatus('reconnecting');
        const delay = RECONNECT_DELAY_MS * Math.min(reconnectAttempts, 5);
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
        setTimeout(connectWebSocket, delay);
    } else {
        console.log('Max reconnect attempts reached');
        showConnectionStatus('disconnected');
    }
}

function clearCache() {
    // Clear all cached data so next load fetches fresh data
    for (const key in dataCache) {
        delete dataCache[key];
    }
}

function reloadCurrentTab() {
    loadTabData(currentTab);
}

function showConnectionStatus(status) {
    const statusEl = document.getElementById('connectionStatus');
    const dotEl = statusEl?.querySelector('.status-dot');
    const textEl = statusEl?.querySelector('.status-text');

    if (!statusEl || !dotEl || !textEl) return;

    dotEl.classList.remove('connected', 'disconnected', 'reconnecting');

    switch (status) {
        case 'connected':
            dotEl.classList.add('connected');
            textEl.textContent = 'Live';
            break;
        case 'disconnected':
            dotEl.classList.add('disconnected');
            textEl.textContent = 'Offline';
            break;
        case 'reconnecting':
            dotEl.classList.add('reconnecting');
            textEl.textContent = 'Reconnecting...';
            break;
    }
}

function showNotification(message, isError = false) {
    // Remove existing toast if any
    const existing = document.querySelector('.toast');
    if (existing) {
        existing.remove();
    }

    const toast = document.createElement('div');
    toast.className = 'toast' + (isError ? ' error' : '');
    toast.textContent = message;
    document.body.appendChild(toast);

    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('visible');
    });

    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.remove('visible');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;
            window.location.hash = tabId;
        });
    });
}

function handleHashChange() {
    const hash = window.location.hash.slice(1) || 'squat';
    selectTab(hash);
}

function selectTab(tabId) {
    // Update active tab UI
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabId);
    });

    currentTab = tabId;
    loadTabData(tabId);
}

async function loadTabData(tabId) {
    const container = document.querySelector('.chart-container');
    container.classList.add('loading');

    try {
        if (COMPOSITES.includes(tabId)) {
            await loadCompositeData(tabId);
        } else if (BODY_COMPOSITION.includes(tabId)) {
            await loadBodyCompositionData(tabId);
        } else {
            await loadMovementData(tabId);
        }
    } catch (err) {
        console.error('Failed to load data:', err);
        showNoData();
    } finally {
        container.classList.remove('loading');
    }
}

async function loadMovementData(movementId) {
    console.log('Loading movement data for:', movementId, 'history:', historyYears, 'prediction:', predictionMonths);

    // Build cache key including current params
    const cacheKey = `${movementId}_${historyYears}_${predictionMonths}`;

    // Check cache
    if (!dataCache[cacheKey]) {
        const url = `/api/movement/${movementId}?history_years=${historyYears}&prediction_months=${predictionMonths}`;
        const response = await fetch(url);
        if (!response.ok) {
            console.error('API error:', response.status);
            throw new Error('API error');
        }
        dataCache[cacheKey] = await response.json();
        console.log('Data loaded:', dataCache[cacheKey]);
    }

    const data = dataCache[cacheKey];

    // Check for data - handle both empty arrays and undefined
    const hasObservations = data.observations && data.observations.length > 0;
    const hasPredictions = data.predictions && data.predictions.length > 0;

    console.log('Has observations:', hasObservations, 'Has predictions:', hasPredictions);

    if (!hasObservations && !hasPredictions) {
        showNoData();
        return;
    }

    renderChart(data, false);
}

async function loadCompositeData(compositeId) {
    console.log('Loading composite data for:', compositeId, 'history:', historyYears, 'prediction:', predictionMonths);

    // Build cache key including current params
    const cacheKey = `composites_${historyYears}_${predictionMonths}`;

    // Check cache
    if (!dataCache[cacheKey]) {
        const url = `/api/composites?history_years=${historyYears}&prediction_months=${predictionMonths}`;
        const response = await fetch(url);
        if (!response.ok) {
            console.error('API error:', response.status);
            throw new Error('API error');
        }
        dataCache[cacheKey] = await response.json();
        console.log('Composites data loaded:', dataCache[cacheKey]);
    }

    const composites = dataCache[cacheKey];
    const key = compositeId === 'ipf-gl' ? 'ipf_gl' : 'sinclair';
    const data = composites[key];

    if (!data || !data.predictions || data.predictions.length === 0) {
        console.log('No composite data available for:', compositeId);
        showNoData();
        return;
    }

    renderChart({
        movement: compositeId === 'ipf-gl' ? 'IPF GoodLift' : 'Sinclair',
        observations: [],
        predictions: data.predictions,
        last_observation_date: data.most_reliable_date,
        current_value: data.current_value,
    }, true);
}

async function loadBodyCompositionData(metricId) {
    console.log('Loading body composition data for:', metricId, 'history:', historyYears, 'prediction:', predictionMonths);

    // Build cache key including current params
    const cacheKey = `${metricId}_${historyYears}_${predictionMonths}`;

    // Check cache
    if (!dataCache[cacheKey]) {
        const url = `/api/${metricId}?history_years=${historyYears}&prediction_months=${predictionMonths}`;
        const response = await fetch(url);
        if (!response.ok) {
            console.error('API error:', response.status);
            if (response.status === 404) {
                showNoData();
                return;
            }
            throw new Error('API error');
        }
        dataCache[cacheKey] = await response.json();
        console.log('Body composition data loaded:', dataCache[cacheKey]);
    }

    const data = dataCache[cacheKey];

    if (!data || !data.predictions || data.predictions.length === 0) {
        console.log('No body composition data available for:', metricId);
        showNoData();
        return;
    }

    // Convert data_points to observations format for chart rendering
    const movementName = metricId === 'bodyfat' ? 'Body Fat %' : 'Lean Body Mass';

    // Find the most recent data point for "Last measurement"
    const dataPoints = data.data_points || [];
    const lastObsDate = dataPoints.length > 0
        ? dataPoints.reduce((latest, p) => p.date > latest ? p.date : latest, dataPoints[0].date)
        : null;

    renderChart({
        movement: movementName,
        observations: dataPoints,
        predictions: data.predictions,
        last_observation_date: lastObsDate,
        current_value: undefined,
    }, false);
}

function showNoData() {
    document.getElementById('chart-wrapper').classList.add('hidden');
    document.getElementById('chart-info').classList.add('hidden');
    document.getElementById('no-data').classList.remove('hidden');

    if (chart) {
        chart.destroy();
        chart = null;
    }
}

function renderChart(data, isComposite) {
    console.log('Rendering chart for:', currentTab, 'isComposite:', isComposite);

    // Store for tooltip lookup
    tooltipData = data;

    document.getElementById('chart-wrapper').classList.remove('hidden');
    document.getElementById('chart-info').classList.remove('hidden');
    document.getElementById('no-data').classList.add('hidden');

    // Prepare datasets
    const datasets = [];

    // Observations (for movements, not composites)
    if (data.observations && data.observations.length > 0) {
        console.log('Adding observations:', data.observations.length);
        datasets.push({
            label: 'Observations',
            data: data.observations.map(p => ({ x: p.date, y: p.value })),
            pointRadius: 2.5,
            pointBackgroundColor: COLORS.observation,
            pointBorderColor: COLORS.observation,
            showLine: false,
            order: 1,
        });
    }

    // Predictions with three sigma bands
    if (data.predictions && data.predictions.length > 0) {
        console.log('Adding predictions:', data.predictions.length);

        // Create sigma band datasets from outermost to innermost
        // Each upper fills to its corresponding lower

        // 3σ Upper (outermost)
        datasets.push({
            label: '3σ Upper',
            data: data.predictions.map(p => ({ x: p.date, y: p.mean + 3 * p.std_dev })),
            borderColor: 'transparent',
            backgroundColor: COLORS.sigma3Area,
            fill: '+1',
            pointRadius: 0,
            order: 8,
        });

        // 3σ Lower
        datasets.push({
            label: '3σ Lower',
            data: data.predictions.map(p => ({ x: p.date, y: p.mean - 3 * p.std_dev })),
            borderColor: 'transparent',
            fill: false,
            pointRadius: 0,
            order: 9,
        });

        // 2σ Upper
        datasets.push({
            label: '2σ Upper',
            data: data.predictions.map(p => ({ x: p.date, y: p.mean + 2 * p.std_dev })),
            borderColor: 'transparent',
            backgroundColor: COLORS.sigma2Area,
            fill: '+1',
            pointRadius: 0,
            order: 6,
        });

        // 2σ Lower
        datasets.push({
            label: '2σ Lower',
            data: data.predictions.map(p => ({ x: p.date, y: p.mean - 2 * p.std_dev })),
            borderColor: 'transparent',
            fill: false,
            pointRadius: 0,
            order: 7,
        });

        // 1σ Upper (innermost)
        datasets.push({
            label: '1σ Upper',
            data: data.predictions.map(p => ({ x: p.date, y: p.mean + p.std_dev })),
            borderColor: 'transparent',
            backgroundColor: COLORS.sigma1Area,
            fill: '+1',
            pointRadius: 0,
            order: 4,
        });

        // 1σ Lower
        datasets.push({
            label: '1σ Lower',
            data: data.predictions.map(p => ({ x: p.date, y: p.mean - p.std_dev })),
            borderColor: 'transparent',
            fill: false,
            pointRadius: 0,
            order: 5,
        });

        // Mean prediction line (on top)
        datasets.push({
            label: 'Prediction',
            data: data.predictions.map(p => ({ x: p.date, y: p.mean })),
            borderColor: COLORS.prediction,
            borderWidth: 2,
            fill: false,
            pointRadius: 0,
            tension: 0.1,
            order: 2,
        });
    }

    // Calculate date bounds based on selector values
    const today = new Date();
    const historyStart = new Date(today);
    historyStart.setFullYear(historyStart.getFullYear() - historyYears);
    const predictionEnd = new Date(today);
    predictionEnd.setMonth(predictionEnd.getMonth() + predictionMonths);

    // Y-axis label
    let yAxisLabel = isComposite ? 'Score' : 'e1RM (kg)';
    if (currentTab === 'bodyweight') {
        yAxisLabel = 'Weight (kg)';
    } else if (currentTab === 'bodyfat') {
        yAxisLabel = 'Body Fat %';
    } else if (currentTab === 'lbm') {
        yAxisLabel = 'kg';
    }

    // Destroy existing chart
    if (chart) {
        chart.destroy();
    }

    // Create chart
    const ctx = document.getElementById('main-chart').getContext('2d');

    try {
        chart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'x',
                    intersect: false,
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'month',
                            displayFormats: {
                                month: 'MMM yyyy'
                            }
                        },
                        min: historyStart,
                        max: predictionEnd,
                        grid: {
                            color: COLORS.gridLine,
                        },
                        ticks: {
                            color: COLORS.text,
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: yAxisLabel,
                            color: COLORS.text,
                        },
                        grid: {
                            color: COLORS.gridLine,
                        },
                        ticks: {
                            color: COLORS.text,
                        },
                    },
                },
                plugins: {
                    legend: {
                        display: false,
                    },
                    tooltip: {
                        backgroundColor: '#2d2d2d',
                        titleColor: '#fff',
                        bodyColor: '#b0b0b0',
                        borderColor: '#3d3d3d',
                        borderWidth: 1,
                        filter: function(tooltipItem, index, tooltipItems) {
                            const label = tooltipItem.dataset.label;

                            // Exclude all sigma bands
                            if (label.includes('σ')) {
                                return false;
                            }

                            // Find first prediction to use as reference date
                            const firstPred = tooltipItems.find(t => t.dataset.label === 'Prediction');
                            const targetDate = firstPred?.raw.x;

                            // For Prediction: only keep the first one
                            if (label === 'Prediction') {
                                return tooltipItem === firstPred;
                            }

                            // For Observations: only keep if date matches exactly
                            if (label === 'Observations') {
                                return targetDate && tooltipItem.raw.x === targetDate;
                            }

                            return false;
                        },
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label === 'Observations' ? 'Observation' : 'Prediction';
                                return `${label}: ${context.parsed.y.toFixed(1)}`;
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            todayLine: {
                                type: 'line',
                                xMin: today,
                                xMax: today,
                                borderColor: COLORS.todayLine,
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'Today',
                                    position: 'start',
                                    backgroundColor: 'rgba(45, 45, 45, 0.8)',
                                    color: '#b0b0b0',
                                    font: { size: 11 },
                                }
                            }
                        }
                    }
                }
            }
        });
        console.log('Chart created successfully');
    } catch (err) {
        console.error('Error creating chart:', err);
        showNoData();
        return;
    }

    // Update info display
    updateChartInfo(data, isComposite);
}

function updateChartInfo(data, isComposite) {
    const lastObsEl = document.getElementById('last-observation');
    const currentValEl = document.getElementById('current-value');

    // Last observation / most reliable date
    if (isComposite) {
        if (data.last_observation_date) {
            lastObsEl.textContent = `Most reliable: ${formatDate(data.last_observation_date)}`;
        } else {
            lastObsEl.textContent = '';
        }
    } else {
        if (data.last_observation_date) {
            lastObsEl.textContent = `Last measurement: ${formatDate(data.last_observation_date)}`;
        } else {
            lastObsEl.textContent = '';
        }
    }

    // Current value
    if (data.current_value !== undefined) {
        currentValEl.textContent = `Current: ${data.current_value.toFixed(1)}`;
    } else if (data.predictions && data.predictions.length > 0) {
        // Find prediction closest to today
        const today = new Date().toISOString().split('T')[0];
        const todayPred = data.predictions.find(p => p.date === today);
        if (todayPred) {
            currentValEl.textContent = `Current: ${todayPred.mean.toFixed(1)}`;
        } else {
            currentValEl.textContent = '';
        }
    } else {
        currentValEl.textContent = '';
    }
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// === TDEE Display ===

async function loadTdee() {
    const valueEl = document.getElementById('tdee-value');
    if (!valueEl) return;

    try {
        const response = await fetch('/api/tdee');
        if (!response.ok) {
            throw new Error('API error');
        }

        const data = await response.json();
        console.log('TDEE data:', data);

        if (data.average_tdee !== undefined) {
            // Success response
            valueEl.textContent = `${Math.round(data.average_tdee)} kcal`;
            valueEl.classList.remove('error');
            valueEl.title = `Today TDEE: ${Math.round(data.tdee)} | Avg intake: ${Math.round(data.avg_calories)} kcal | Weight change: ${data.weight_change_kg.toFixed(2)} kg | Pairs: ${data.pairs_used}`;
        } else if (data.error) {
            // Error response
            valueEl.textContent = data.message;
            valueEl.classList.add('error');
            valueEl.title = '';
        } else {
            valueEl.textContent = '--';
            valueEl.classList.add('error');
            valueEl.title = '';
        }
    } catch (err) {
        console.error('Failed to load TDEE:', err);
        valueEl.textContent = '--';
        valueEl.classList.add('error');
        valueEl.title = '';
    }
}
