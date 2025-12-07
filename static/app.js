// SportModel - Training Analytics Dashboard

const MOVEMENTS = ['squat', 'bench', 'deadlift', 'snatch', 'cj', 'bodyweight'];
const COMPOSITES = ['ipf-gl', 'sinclair'];

// Chart instance
let chart = null;

// Data cache
const dataCache = {};

// Current tab
let currentTab = 'squat';

// Chart colors
const COLORS = {
    observation: '#4CAF50',
    prediction: '#2196F3',
    ciArea: 'rgba(33, 150, 243, 0.15)',
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
    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);
    connectWebSocket();
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
    console.log('Loading movement data for:', movementId);

    // Check cache
    if (!dataCache[movementId]) {
        const response = await fetch(`/api/movement/${movementId}`);
        if (!response.ok) {
            console.error('API error:', response.status);
            throw new Error('API error');
        }
        dataCache[movementId] = await response.json();
        console.log('Data loaded:', dataCache[movementId]);
    }

    const data = dataCache[movementId];

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
    console.log('Loading composite data for:', compositeId);

    // Check cache
    if (!dataCache.composites) {
        const response = await fetch('/api/composites');
        if (!response.ok) {
            console.error('API error:', response.status);
            throw new Error('API error');
        }
        dataCache.composites = await response.json();
        console.log('Composites data loaded:', dataCache.composites);
    }

    const composites = dataCache.composites;
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
            pointRadius: 5,
            pointBackgroundColor: COLORS.observation,
            pointBorderColor: COLORS.observation,
            showLine: false,
            order: 1,
        });
    }

    // Predictions
    if (data.predictions && data.predictions.length > 0) {
        console.log('Adding predictions:', data.predictions.length);

        // CI Upper (for fill)
        datasets.push({
            label: '95% CI Upper',
            data: data.predictions.map(p => ({ x: p.date, y: p.ci_upper })),
            borderColor: 'transparent',
            backgroundColor: COLORS.ciArea,
            fill: '+1',
            pointRadius: 0,
            order: 3,
        });

        // CI Lower
        datasets.push({
            label: '95% CI Lower',
            data: data.predictions.map(p => ({ x: p.date, y: p.ci_lower })),
            borderColor: 'transparent',
            fill: false,
            pointRadius: 0,
            order: 4,
        });

        // Mean prediction line
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

    // Calculate date bounds
    const today = new Date();
    const oneYearAgo = new Date(today);
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
    const sixMonthsFuture = new Date(today);
    sixMonthsFuture.setMonth(sixMonthsFuture.getMonth() + 6);

    // Y-axis label
    let yAxisLabel = isComposite ? 'Score' : 'e1RM (kg)';
    if (currentTab === 'bodyweight') {
        yAxisLabel = 'Weight (kg)';
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
                    mode: 'index',
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
                        min: oneYearAgo,
                        max: sixMonthsFuture,
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
                        callbacks: {
                            label: function(context) {
                                if (context.dataset.label === '95% CI Upper' ||
                                    context.dataset.label === '95% CI Lower') {
                                    return null;
                                }
                                const value = context.parsed.y.toFixed(1);
                                return `${context.dataset.label}: ${value}`;
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
