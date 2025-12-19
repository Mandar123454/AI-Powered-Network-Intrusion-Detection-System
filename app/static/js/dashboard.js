/*
 * AI-NIDS Dashboard JavaScript
 * Real-time updates and interactive features
 */

// ===== Global State =====
const NIDS = {
    charts: {},
    refreshInterval: 30000, // 30 seconds
    autoRefresh: true,
    ws: null
};

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', function() {
    initializeSidebar();
    initializeTooltips();
    initializeAutoRefresh();
    initializeSearch();
    initializeBulkActions();
});

// ===== Sidebar =====
function initializeSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.querySelector('.sidebar-toggle');
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });
    }
    
    // Close sidebar on outside click (mobile)
    document.addEventListener('click', function(e) {
        if (window.innerWidth < 992) {
            if (!sidebar.contains(e.target) && !toggleBtn?.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        }
    });
}

// ===== Tooltips =====
function initializeTooltips() {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => {
        new bootstrap.Tooltip(el);
    });
}

// ===== Auto Refresh =====
function initializeAutoRefresh() {
    const refreshToggle = document.getElementById('auto-refresh-toggle');
    if (refreshToggle) {
        refreshToggle.addEventListener('change', function() {
            NIDS.autoRefresh = this.checked;
            if (NIDS.autoRefresh) {
                startAutoRefresh();
            } else {
                stopAutoRefresh();
            }
        });
    }
    
    if (NIDS.autoRefresh) {
        startAutoRefresh();
    }
}

function startAutoRefresh() {
    NIDS.refreshTimer = setInterval(refreshDashboard, NIDS.refreshInterval);
}

function stopAutoRefresh() {
    if (NIDS.refreshTimer) {
        clearInterval(NIDS.refreshTimer);
    }
}

function refreshDashboard() {
    // Refresh stats
    fetch('/api/v1/status')
        .then(r => r.json())
        .then(data => {
            updateStats(data);
        })
        .catch(console.error);
    
    // Refresh charts if on dashboard
    if (typeof updateCharts === 'function') {
        updateCharts();
    }
    
    // Refresh alerts if on alerts page
    if (typeof refreshAlerts === 'function') {
        refreshAlerts();
    }
}

function updateStats(data) {
    // Update stat cards if they exist
    const elements = {
        'stat-total-alerts': data.statistics?.total_alerts,
        'stat-critical': data.statistics?.critical_alerts,
        'stat-flows': data.statistics?.total_flows,
        'stat-detection-rate': data.statistics?.detection_rate
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el && value !== undefined) {
            animateValue(el, parseInt(el.textContent) || 0, value, 500);
        }
    });
}

function animateValue(element, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const value = Math.floor(start + (range * easeOutQuad(progress)));
        element.textContent = value.toLocaleString();
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

function easeOutQuad(t) {
    return t * (2 - t);
}

// ===== Search =====
function initializeSearch() {
    const searchInput = document.getElementById('global-search');
    if (!searchInput) return;
    
    // Create search results dropdown
    const searchContainer = searchInput.closest('.navbar-search');
    let resultsDropdown = document.getElementById('search-results');
    
    if (!resultsDropdown) {
        resultsDropdown = document.createElement('div');
        resultsDropdown.id = 'search-results';
        resultsDropdown.className = 'search-results-dropdown';
        searchContainer.appendChild(resultsDropdown);
    }
    
    let debounceTimer;
    
    searchInput.addEventListener('input', function() {
        clearTimeout(debounceTimer);
        const query = this.value.trim();
        
        if (query.length < 2) {
            resultsDropdown.classList.remove('show');
            return;
        }
        
        debounceTimer = setTimeout(() => {
            performSearch(query, resultsDropdown);
        }, 300);
    });
    
    searchInput.addEventListener('focus', function() {
        if (this.value.length >= 2 && resultsDropdown.innerHTML) {
            resultsDropdown.classList.add('show');
        }
    });
    
    // Close on outside click
    document.addEventListener('click', function(e) {
        if (!searchContainer.contains(e.target)) {
            resultsDropdown.classList.remove('show');
        }
    });
    
    // Keyboard navigation
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            resultsDropdown.classList.remove('show');
        }
    });
}

function performSearch(query, resultsContainer) {
    resultsContainer.innerHTML = '<div class="search-loading"><i class="bi bi-arrow-repeat spin"></i> Searching...</div>';
    resultsContainer.classList.add('show');
    
    fetch(`/api/v1/search?q=${encodeURIComponent(query)}`)
        .then(r => r.json())
        .then(data => {
            displaySearchResults(data, resultsContainer);
        })
        .catch(err => {
            resultsContainer.innerHTML = '<div class="search-error"><i class="bi bi-exclamation-circle"></i> Search failed</div>';
        });
}

function displaySearchResults(results, container) {
    if (results.length === 0) {
        container.innerHTML = `
            <div class="search-empty">
                <i class="bi bi-search"></i>
                <p>No results found</p>
                <span>Try searching for an IP address, attack type, or alert</span>
            </div>
        `;
        return;
    }
    
    const severityColors = {
        'critical': '#dc3545',
        'high': '#fd7e14',
        'medium': '#ffc107',
        'low': '#198754',
        'info': '#0dcaf0'
    };
    
    container.innerHTML = `
        <div class="search-results-header">
            <span>${results.length} result${results.length > 1 ? 's' : ''} found</span>
        </div>
        ${results.map(r => `
            <a href="${r.url}" class="search-result-item">
                <div class="result-icon" style="color: ${severityColors[r.severity] || '#6c757d'}">
                    <i class="bi bi-${r.icon}"></i>
                </div>
                <div class="result-content">
                    <div class="result-title">${r.title}</div>
                    <div class="result-subtitle">${r.subtitle}</div>
                </div>
                <span class="result-type badge bg-${r.type === 'alert' ? 'danger' : r.type === 'flow' ? 'info' : 'secondary'}">${r.type}</span>
            </a>
        `).join('')}
    `;
}

// ===== Bulk Actions =====
function initializeBulkActions() {
    const selectAll = document.getElementById('select-all');
    if (selectAll) {
        selectAll.addEventListener('change', function() {
            document.querySelectorAll('.alert-checkbox').forEach(cb => {
                cb.checked = this.checked;
            });
            updateBulkActionsBar();
        });
    }
    
    document.querySelectorAll('.alert-checkbox').forEach(cb => {
        cb.addEventListener('change', updateBulkActionsBar);
    });
}

function updateBulkActionsBar() {
    const selected = document.querySelectorAll('.alert-checkbox:checked');
    const bulkBar = document.getElementById('bulk-actions');
    const count = document.getElementById('selected-count');
    
    if (bulkBar) {
        if (selected.length > 0) {
            bulkBar.classList.add('show');
            if (count) count.textContent = selected.length;
        } else {
            bulkBar.classList.remove('show');
        }
    }
}

function getSelectedIds() {
    return Array.from(document.querySelectorAll('.alert-checkbox:checked'))
        .map(cb => cb.value);
}

function bulkAcknowledge() {
    const ids = getSelectedIds();
    if (ids.length === 0) return;
    
    fetch('/alerts/bulk-acknowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids })
    })
    .then(r => r.json())
    .then(data => {
        showToast(`${data.count} alerts acknowledged`);
        refreshAlerts();
    });
}

function bulkResolve() {
    const ids = getSelectedIds();
    if (ids.length === 0) return;
    
    fetch('/alerts/bulk-resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids })
    })
    .then(r => r.json())
    .then(data => {
        showToast(`${data.count} alerts resolved`);
        refreshAlerts();
    });
}

function bulkDelete() {
    const ids = getSelectedIds();
    if (ids.length === 0) return;
    
    if (!confirm(`Delete ${ids.length} alerts? This cannot be undone.`)) return;
    
    fetch('/alerts/bulk-delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids })
    })
    .then(r => r.json())
    .then(data => {
        showToast(`${data.count} alerts deleted`);
        refreshAlerts();
    });
}

// ===== Toast Notifications =====
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <i class="bi bi-${type === 'success' ? 'check-circle' : type === 'error' ? 'x-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
    return container;
}

// ===== Utility Functions =====
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text)
        .then(() => showToast('Copied to clipboard'))
        .catch(() => showToast('Failed to copy', 'error'));
}

// ===== API Helpers =====
async function apiGet(endpoint) {
    const response = await fetch(`/api/v1${endpoint}`);
    if (!response.ok) throw new Error('API request failed');
    return response.json();
}

async function apiPost(endpoint, data) {
    const response = await fetch(`/api/v1${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error('API request failed');
    return response.json();
}

// ===== Export =====
window.NIDS = NIDS;
window.showToast = showToast;
window.formatNumber = formatNumber;
window.formatDate = formatDate;
window.copyToClipboard = copyToClipboard;
