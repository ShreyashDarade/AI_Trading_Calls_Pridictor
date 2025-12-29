/**
 * Indian AI Trader - Main Application JavaScript
 * Handles SSE streaming, API calls, and UI updates
 */

// ============================================
// CONFIGURATION
// ============================================

const API_BASE_URL = '/api';
const SSE_URL = '/api/stream';

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Format number as Indian currency
 */
function formatCurrency(amount) {
    const isNegative = amount < 0;
    amount = Math.abs(amount);
    
    const formatted = new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(amount);
    
    return isNegative ? '-' + formatted : formatted;
}

/**
 * Format percentage
 */
function formatPercent(value, decimals = 2) {
    const prefix = value >= 0 ? '+' : '';
    return prefix + value.toFixed(decimals) + '%';
}

/**
 * Get CSS class based on value (positive/negative)
 */
function getChangeClass(value) {
    return value >= 0 ? 'positive' : 'negative';
}

/**
 * Get icon for price change
 */
function getChangeIcon(value) {
    return value >= 0 ? 'fa-caret-up' : 'fa-caret-down';
}

// ============================================
// API FUNCTIONS
// ============================================

/**
 * Fetch data from API
 */
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(API_BASE_URL + endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API fetch error:', error);
        throw error;
    }
}

/**
 * Get market status
 */
async function getMarketStatus() {
    return fetchAPI('/market/status');
}

/**
 * Get watchlist
 */
async function getWatchlist() {
    return fetchAPI('/watchlist');
}

/**
 * Get quotes for symbols
 */
async function getQuotes(symbols) {
    return fetchAPI(`/quotes?symbols=${symbols.join(',')}`);
}

/**
 * Get latest signals
 */
async function getSignals(minConfidence = 0.5, limit = 10) {
    return fetchAPI(`/signals/latest?min_confidence=${minConfidence}&limit=${limit}`);
}

/**
 * Get portfolio
 */
async function getPortfolio() {
    return fetchAPI('/portfolio');
}

// ============================================
// SSE (Server-Sent Events) for Live Updates
// ============================================

let eventSource = null;

/**
 * Start receiving live updates via SSE
 */
function startLiveUpdates() {
    // Close existing connection if any
    if (eventSource) {
        eventSource.close();
    }
    
    console.log('Starting SSE connection...');
    eventSource = new EventSource(SSE_URL);
    
    eventSource.onopen = function() {
        console.log('SSE connection established');
    };
    
    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleLiveUpdate(data);
        } catch (error) {
            console.error('Error parsing SSE data:', error);
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('SSE error:', error);
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
            if (eventSource.readyState === EventSource.CLOSED) {
                startLiveUpdates();
            }
        }, 5000);
    };
}

/**
 * Handle incoming live update
 */
function handleLiveUpdate(data) {
    if (data.type === 'price') {
        updateStockPrice(data.data);
    } else if (data.type === 'signal') {
        displayNewSignal(data.data);
    } else if (data.type === 'portfolio') {
        updatePortfolioDisplay(data.data);
    }
}

/**
 * Stop live updates
 */
function stopLiveUpdates() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
        console.log('SSE connection closed');
    }
}

// ============================================
// UI UPDATE FUNCTIONS
// ============================================

/**
 * Update stock price in the UI
 */
function updateStockPrice(stock) {
    const symbol = stock.symbol;
    
    // Update watchlist tile
    const tiles = document.querySelectorAll('.stock-tile');
    tiles.forEach(tile => {
        const symbolEl = tile.querySelector('.stock-symbol');
        if (symbolEl && symbolEl.textContent === symbol) {
            const ltpEl = tile.querySelector('.stock-ltp');
            const changeEl = tile.querySelector('.stock-change');
            
            if (ltpEl) {
                ltpEl.textContent = formatCurrency(stock.ltp);
                // Add flash effect
                ltpEl.classList.add('animate-pulse');
                setTimeout(() => ltpEl.classList.remove('animate-pulse'), 1000);
            }
            
            if (changeEl) {
                const changeClass = getChangeClass(stock.change_percent);
                const iconClass = getChangeIcon(stock.change_percent);
                
                changeEl.className = 'stock-change ' + changeClass;
                changeEl.innerHTML = `
                    <i class="fas ${iconClass}"></i>
                    <span>${stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)} (${formatPercent(stock.change_percent)})</span>
                `;
            }
        }
    });
    
    // Update positions table
    const ltpCell = document.getElementById(`ltp-${symbol.toLowerCase()}`);
    if (ltpCell) {
        ltpCell.textContent = formatCurrency(stock.ltp);
    }
}

/**
 * Display new signal
 */
function displayNewSignal(signal) {
    const container = document.getElementById('signals-container');
    if (!container) return;
    
    const actionClass = signal.action.toLowerCase().replace('_', '-');
    const actionIcon = signal.action === 'LONG' ? 'fa-arrow-up' : 
                       signal.action === 'SHORT' ? 'fa-arrow-down' : 'fa-minus';
    
    const signalHTML = `
        <div class="signal-card ${actionClass} mb-md animate-slideUp">
            <div class="signal-header">
                <span class="signal-symbol">${signal.symbol}</span>
                <span class="signal-action ${actionClass}">
                    <i class="fas ${actionIcon}"></i> ${signal.action.replace('_', ' ')}
                </span>
            </div>
            <div class="signal-confidence">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${signal.confidence * 100}%;"></div>
                </div>
                <span class="confidence-value">${Math.round(signal.confidence * 100)}%</span>
            </div>
            ${signal.entry_price ? `
            <div class="signal-levels">
                <div class="signal-level">
                    <div class="signal-level-label">Entry</div>
                    <div class="signal-level-value">${formatCurrency(signal.entry_price)}</div>
                </div>
                <div class="signal-level">
                    <div class="signal-level-label">Stop Loss</div>
                    <div class="signal-level-value text-danger">${formatCurrency(signal.stop_loss)}</div>
                </div>
                <div class="signal-level">
                    <div class="signal-level-label">Target</div>
                    <div class="signal-level-value text-success">${formatCurrency(signal.target_1)}</div>
                </div>
            </div>
            ` : ''}
            <div class="signal-reasons">
                ${signal.reason_codes.map(code => `<span class="reason-tag">${code}</span>`).join('')}
            </div>
        </div>
    `;
    
    // Insert at the beginning
    container.insertAdjacentHTML('afterbegin', signalHTML);
    
    // Remove oldest signal if more than 5
    const signals = container.querySelectorAll('.signal-card');
    if (signals.length > 5) {
        signals[signals.length - 1].remove();
    }
}

/**
 * Update portfolio display
 */
function updatePortfolioDisplay(portfolio) {
    const totalEl = document.getElementById('portfolio-total');
    const pnlEl = document.querySelector('.portfolio-pnl-value');
    const pnlPercentEl = document.querySelector('.portfolio-pnl-percent');
    
    if (totalEl) {
        totalEl.textContent = formatCurrency(portfolio.capital + portfolio.total_pnl);
    }
    
    if (pnlEl) {
        pnlEl.textContent = (portfolio.total_pnl >= 0 ? '+' : '') + formatCurrency(portfolio.total_pnl);
    }
    
    if (pnlPercentEl) {
        pnlPercentEl.textContent = formatPercent(portfolio.total_pnl_percent);
    }
}

// ============================================
// SEARCH FUNCTIONALITY
// ============================================

/**
 * Search instruments
 */
async function searchInstruments(query) {
    if (query.length < 1) return [];
    
    try {
        const data = await fetchAPI(`/instruments/search?q=${encodeURIComponent(query)}&limit=10`);
        return data.results || [];
    } catch (error) {
        console.error('Search error:', error);
        return [];
    }
}

/**
 * Setup search input with autocomplete
 */
function setupSearch(inputId, resultsId) {
    const input = document.getElementById(inputId);
    const results = document.getElementById(resultsId);
    
    if (!input) return;
    
    let debounceTimer;
    
    input.addEventListener('input', function() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            const query = this.value.trim();
            
            if (query.length < 1) {
                if (results) results.innerHTML = '';
                return;
            }
            
            const instruments = await searchInstruments(query);
            
            if (results) {
                results.innerHTML = instruments.map(inst => `
                    <div class="search-result-item" onclick="selectInstrument('${inst.symbol}')">
                        <strong>${inst.symbol}</strong>
                        <span class="text-muted">${inst.name}</span>
                    </div>
                `).join('');
            }
        }, 300);
    });
}

// ============================================
// PAGE INITIALIZATION
// ============================================

/**
 * Initialize page based on current route
 */
function initializePage() {
    const path = window.location.pathname;
    
    // Common initialization
    updateMarketStatus();
    
    // Page-specific initialization
    if (path === '/' || path === '/index.html') {
        initDashboard();
    } else if (path === '/watchlist' || path.includes('watchlist.html')) {
        initWatchlist();
    } else if (path === '/signals' || path.includes('signals.html')) {
        initSignals();
    } else if (path.startsWith('/stock/')) {
        initStockDetail(path.split('/stock/')[1]);
    }
    
    // Start live updates
    startLiveUpdates();
}

/**
 * Initialize dashboard
 */
async function initDashboard() {
    console.log('Initializing dashboard...');
    
    // Fetch initial data
    try {
        const [watchlist, signals, portfolio] = await Promise.all([
            getWatchlist().catch(() => ({ items: [] })),
            getSignals().catch(() => ({ signals: [] })),
            getPortfolio().catch(() => null)
        ]);
        
        // Update displays
        if (portfolio) {
            updatePortfolioDisplay(portfolio);
        }
        
    } catch (error) {
        console.error('Dashboard init error:', error);
    }
}

/**
 * Initialize watchlist page
 */
async function initWatchlist() {
    console.log('Initializing watchlist...');
    setupSearch('watchlist-search', 'search-results');
}

/**
 * Initialize signals page
 */
async function initSignals() {
    console.log('Initializing signals...');
    
    try {
        const data = await getSignals(0.3, 20);
        // Render signals...
    } catch (error) {
        console.error('Signals init error:', error);
    }
}

/**
 * Initialize stock detail page
 */
async function initStockDetail(symbol) {
    console.log('Initializing stock detail for:', symbol);
}

/**
 * Update market status display
 */
async function updateMarketStatus() {
    try {
        const status = await getMarketStatus();
        
        const dot = document.getElementById('market-dot');
        const label = document.getElementById('market-label');
        
        if (dot && label) {
            if (status.status === 'OPEN') {
                dot.classList.remove('closed');
                dot.classList.add('open');
                label.textContent = 'Market Open';
            } else {
                dot.classList.remove('open');
                dot.classList.add('closed');
                label.textContent = 'Market Closed';
            }
        }
    } catch (error) {
        console.error('Market status error:', error);
    }
}

// ============================================
// EVENT LISTENERS
// ============================================

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initializePage);

// Cleanup on page unload
window.addEventListener('beforeunload', stopLiveUpdates);

// Handle visibility change (pause updates when tab is hidden)
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        stopLiveUpdates();
    } else {
        startLiveUpdates();
    }
});
