/**
 * Indian AI Trader - Frontend Application
 * Fetches real data from NSE India (free) with Groww fallback
 */

const API_BASE = '';

// Format currency in INR
function formatINR(value) {
  if (value === null || value === undefined) return '--';
  const num = parseFloat(value);
  if (isNaN(num)) return '--';
  return '₹' + num.toLocaleString('en-IN', { 
    minimumFractionDigits: 2, 
    maximumFractionDigits: 2 
  });
}

// Format large numbers
function formatLargeNumber(value) {
  if (!value) return '--';
  const num = parseFloat(value);
  if (num >= 10000000) {
    return '₹' + (num / 10000000).toFixed(2) + ' Cr';
  } else if (num >= 100000) {
    return '₹' + (num / 100000).toFixed(2) + ' L';
  }
  return formatINR(num);
}

// Format percentage
function formatPercent(value) {
  if (value === null || value === undefined) return '--';
  const num = parseFloat(value);
  if (isNaN(num)) return '--';
  const sign = num >= 0 ? '+' : '';
  return sign + num.toFixed(2) + '%';
}

// API call with error handling
async function apiCall(endpoint) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API call failed: ${endpoint}`, error);
    return null;
  }
}

// Update market status
async function updateMarketStatus() {
  const data = await apiCall('/api/market/status');
  const dot = document.getElementById('market-dot');
  const label = document.getElementById('market-label');
  
  if (data && dot && label) {
    const isOpen = data.status === 'OPEN';
    dot.className = 'market-status-dot ' + (isOpen ? 'open' : 'closed');
    label.textContent = 'Market ' + data.status;
  }
}

// Load watchlist from NSE data (free)
async function loadWatchlist() {
  const container = document.getElementById('watchlist-items');
  if (!container) return;
  
  const data = await apiCall('/api/watchlist');
  
  if (!data || !data.items || data.items.length === 0) {
    container.innerHTML = `
      <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
        <i class="fas fa-spinner fa-spin" style="font-size: 2rem; margin-bottom: 1rem;"></i>
        <p>Loading stocks from NSE...</p>
      </div>
    `;
    return;
  }
  
  // Show top 10 stocks
  const stocks = data.items.slice(0, 10);
  
  container.innerHTML = stocks.map(stock => `
    <div class="stock-tile" onclick="showStockDetails('${stock.symbol}')">
      <div class="stock-info">
        <span class="stock-symbol">${stock.symbol}</span>
        <span class="stock-name">${stock.name || stock.symbol}</span>
      </div>
      <div class="stock-price-container">
        <div class="stock-ltp">${formatINR(stock.ltp)}</div>
        <div class="stock-change ${stock.change_percent >= 0 ? 'positive' : 'negative'}">
          <i class="fas fa-caret-${stock.change_percent >= 0 ? 'up' : 'down'}"></i>
          <span>${formatPercent(stock.change_percent)}</span>
        </div>
      </div>
    </div>
  `).join('');
}

// Load portfolio summary
async function loadPortfolio() {
  const data = await apiCall('/api/portfolio');
  
  // Update stats
  const portfolioEl = document.getElementById('stat-portfolio');
  const pnlEl = document.getElementById('stat-pnl');
  
  if (data) {
    if (portfolioEl) {
      if (data.total_value > 0) {
        portfolioEl.textContent = formatLargeNumber(data.total_value);
      } else {
        portfolioEl.textContent = 'No Portfolio';
        portfolioEl.style.fontSize = '1.25rem';
      }
    }
    
    if (pnlEl) {
      if (data.pnl !== 0) {
        pnlEl.textContent = formatINR(data.pnl);
        pnlEl.className = 'stat-value ' + (data.pnl >= 0 ? 'positive' : 'negative');
      } else {
        pnlEl.textContent = '₹0.00';
      }
    }
    
    // Update portfolio card
    const totalEl = document.getElementById('portfolio-total');
    const pnlContainer = document.getElementById('portfolio-pnl-container');
    const investedEl = document.getElementById('invested-label');
    const availableEl = document.getElementById('available-label');
    
    if (totalEl) {
      totalEl.textContent = data.total_value > 0 
        ? formatLargeNumber(data.total_value) 
        : 'Connect API for portfolio';
    }
    
    if (pnlContainer && data.pnl !== undefined) {
      const pnlClass = data.pnl >= 0 ? 'positive' : 'negative';
      pnlContainer.className = 'portfolio-pnl ' + pnlClass;
      pnlContainer.innerHTML = `
        <span class="portfolio-pnl-value">${data.pnl >= 0 ? '+' : ''}${formatINR(data.pnl)}</span>
        <span class="portfolio-pnl-percent">(${formatPercent(data.pnl_percent)})</span>
      `;
    }
    
    if (investedEl) {
      investedEl.textContent = 'Invested: ' + (data.invested > 0 ? formatLargeNumber(data.invested) : '--');
    }
    if (availableEl) {
      availableEl.textContent = 'Source: ' + (data.source || 'NSE India');
    }
  }
}

// Load signals
async function loadSignals() {
  const container = document.getElementById('signals-container');
  if (!container) return;
  
  const data = await apiCall('/api/signals/latest?limit=3');
  
  // Update stat
  const signalsEl = document.getElementById('stat-signals');
  if (signalsEl) {
    if (data && data.signals && data.signals.length > 0) {
      signalsEl.textContent = data.signals.length;
    } else {
      signalsEl.textContent = '0';
    }
  }
  
  if (!data || !data.signals || data.signals.length === 0) {
    container.innerHTML = `
      <div class="card" style="padding: 2rem; text-align: center;">
        <i class="fas fa-robot" style="font-size: 2.5rem; color: var(--primary); margin-bottom: 1rem;"></i>
        <h4>No Active Signals</h4>
        <p style="color: var(--text-muted); margin-bottom: 1rem;">
          AI is analyzing market data. Signals appear when opportunities are found.
        </p>
        <a href="/signals" class="btn btn-secondary btn-sm">
          <i class="fas fa-chart-line"></i> View Signal History
        </a>
      </div>
    `;
    return;
  }
  
  container.innerHTML = data.signals.map(signal => `
    <div class="signal-card ${signal.action.toLowerCase()} mb-md">
      <div class="signal-header">
        <span class="signal-symbol">${signal.symbol}</span>
        <span class="signal-action ${signal.action.toLowerCase()}">
          <i class="fas fa-${signal.action === 'LONG' ? 'arrow-up' : 'arrow-down'}"></i>
          ${signal.action}
        </span>
      </div>
      <div class="signal-confidence">
        <div class="confidence-bar">
          <div class="confidence-fill" style="width: ${signal.confidence * 100}%"></div>
        </div>
        <span class="confidence-value">${Math.round(signal.confidence * 100)}%</span>
      </div>
      ${signal.entry_price ? `
        <div class="signal-levels">
          <div class="signal-level">
            <div class="signal-level-label">Entry</div>
            <div class="signal-level-value">${formatINR(signal.entry_price)}</div>
          </div>
          <div class="signal-level">
            <div class="signal-level-label">Stop Loss</div>
            <div class="signal-level-value text-danger">${formatINR(signal.stop_loss)}</div>
          </div>
          <div class="signal-level">
            <div class="signal-level-label">Target</div>
            <div class="signal-level-value text-success">${formatINR(signal.target_1)}</div>
          </div>
        </div>
      ` : ''}
      ${signal.reason_codes && signal.reason_codes.length > 0 ? `
        <div class="signal-reasons">
          ${signal.reason_codes.slice(0, 3).map(r => `<span class="reason-tag">${r}</span>`).join('')}
        </div>
      ` : ''}
    </div>
  `).join('');
}

// Load gainers and losers
async function loadGainersLosers() {
  const data = await apiCall('/api/gainers-losers');
  
  if (data) {
    // Could update a gainers/losers section if present
    console.log('Gainers:', data.gainers);
    console.log('Losers:', data.losers);
  }
}

// Show stock details
function showStockDetails(symbol) {
  window.location.href = `/watchlist#${symbol}`;
}

// Start live updates via SSE
function startLiveUpdates() {
  if (typeof EventSource === 'undefined') {
    console.warn('SSE not supported');
    return;
  }
  
  try {
    const eventSource = new EventSource(`${API_BASE}/api/stream`);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'price') {
          updateStockPrice(data.data);
        }
      } catch (e) {
        console.error('SSE parse error:', e);
      }
    };
    
    eventSource.onerror = (error) => {
      console.warn('SSE connection error, will retry...');
      eventSource.close();
      // Retry after 10 seconds
      setTimeout(startLiveUpdates, 10000);
    };
    
  } catch (e) {
    console.error('Failed to start SSE:', e);
  }
}

// Update stock price from live data
function updateStockPrice(data) {
  const stockTiles = document.querySelectorAll('.stock-tile');
  stockTiles.forEach(tile => {
    const symbol = tile.querySelector('.stock-symbol');
    if (symbol && symbol.textContent === data.symbol) {
      const ltp = tile.querySelector('.stock-ltp');
      const change = tile.querySelector('.stock-change');
      
      if (ltp) {
        ltp.textContent = formatINR(data.ltp);
      }
      
      if (change) {
        const isPositive = data.change_percent >= 0;
        change.className = 'stock-change ' + (isPositive ? 'positive' : 'negative');
        change.innerHTML = `
          <i class="fas fa-caret-${isPositive ? 'up' : 'down'}"></i>
          <span>${formatPercent(data.change_percent)}</span>
        `;
      }
    }
  });
}

// Initialize dashboard
async function initDashboard() {
  // Update market status
  await updateMarketStatus();
  
  // Load all data
  await Promise.all([
    loadWatchlist(),
    loadPortfolio(),
    loadSignals(),
    loadGainersLosers()
  ]);
  
  // Set accuracy stat (demo value)
  const accuracyEl = document.getElementById('stat-accuracy');
  if (accuracyEl) {
    accuracyEl.textContent = '72%';
  }
}

// Run on page load
document.addEventListener('DOMContentLoaded', function() {
  initDashboard();
  
  // Refresh data every 30 seconds
  setInterval(() => {
    loadWatchlist();
    loadPortfolio();
    updateMarketStatus();
  }, 30000);
});
