/**
 * AI-NIDS Fuzzy Search with Fuse.js
 * ==================================
 * Provides instant fuzzy search across alerts, IPs, and attack types
 */

// Import Fuse.js from CDN (loaded in base.html)
// This file initializes and manages the search functionality

class NIDSSearch {
    constructor() {
        this.fuse = null;
        this.searchIndex = [];
        this.searchInput = null;
        this.resultsContainer = null;
        this.isInitialized = false;
        this.debounceTimer = null;
        
        // Fuse.js configuration for optimal security search
        this.fuseOptions = {
            keys: [
                { name: 'source_ip', weight: 0.3 },
                { name: 'destination_ip', weight: 0.2 },
                { name: 'attack_type', weight: 0.25 },
                { name: 'description', weight: 0.15 },
                { name: 'severity', weight: 0.1 }
            ],
            threshold: 0.4,          // Lower = more strict matching
            distance: 100,           // How close the match must be
            ignoreLocation: true,    // Don't care about match position
            includeScore: true,      // Include match score
            includeMatches: true,    // Include match indices
            minMatchCharLength: 2,   // Minimum chars to match
            useExtendedSearch: true, // Enable extended search syntax
            findAllMatches: true,
            shouldSort: true,
        };
        
        this.init();
    }
    
    async init() {
        // Wait for DOM
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }
    
    setup() {
        this.searchInput = document.querySelector('#global-search, .search-input, [data-search]');
        this.resultsContainer = document.querySelector('#search-results, .search-results');
        
        if (!this.searchInput) {
            console.log('[Search] No search input found');
            return;
        }
        
        // Create results container if not exists
        if (!this.resultsContainer) {
            this.resultsContainer = document.createElement('div');
            this.resultsContainer.id = 'search-results';
            this.resultsContainer.className = 'search-results-dropdown';
            this.searchInput.parentNode.appendChild(this.resultsContainer);
        }
        
        // Bind events
        this.searchInput.addEventListener('input', (e) => this.handleInput(e));
        this.searchInput.addEventListener('focus', () => this.showResults());
        this.searchInput.addEventListener('blur', () => {
            setTimeout(() => this.hideResults(), 200);
        });
        
        // Keyboard navigation
        this.searchInput.addEventListener('keydown', (e) => this.handleKeydown(e));
        
        // Load initial data
        this.loadSearchData();
        
        console.log('[Search] Fuzzy search initialized');
    }
    
    async loadSearchData() {
        try {
            // Fetch alerts and other searchable data
            const [alertsRes, sourcesRes] = await Promise.allSettled([
                fetch('/api/v1/alerts?limit=500'),
                fetch('/api/v1/sources/top?limit=100')
            ]);
            
            const searchData = [];
            
            // Process alerts
            if (alertsRes.status === 'fulfilled' && alertsRes.value.ok) {
                const alertsData = await alertsRes.value.json();
                const alerts = alertsData.alerts || alertsData.data || alertsData || [];
                
                alerts.forEach(alert => {
                    searchData.push({
                        type: 'alert',
                        id: alert.id,
                        source_ip: alert.source_ip,
                        destination_ip: alert.destination_ip,
                        attack_type: alert.attack_type,
                        severity: alert.severity,
                        description: alert.description || '',
                        timestamp: alert.timestamp,
                        url: `/alerts/${alert.id}`
                    });
                });
            }
            
            // Process sources
            if (sourcesRes.status === 'fulfilled' && sourcesRes.value.ok) {
                const sourcesData = await sourcesRes.value.json();
                const sources = sourcesData.sources || sourcesData || [];
                
                sources.forEach(source => {
                    searchData.push({
                        type: 'source',
                        source_ip: source.ip || source.source_ip,
                        attack_type: 'Source IP',
                        description: `${source.count || 0} alerts from this IP`,
                        url: `/alerts?source_ip=${source.ip || source.source_ip}`
                    });
                });
            }
            
            // Add common search terms
            const commonSearches = [
                { type: 'category', attack_type: 'DDoS', description: 'Distributed Denial of Service attacks', url: '/alerts?attack_type=DDoS' },
                { type: 'category', attack_type: 'SQL Injection', description: 'SQL Injection attempts', url: '/alerts?attack_type=SQL%20Injection' },
                { type: 'category', attack_type: 'Brute Force', description: 'Brute force authentication attacks', url: '/alerts?attack_type=Brute%20Force' },
                { type: 'category', attack_type: 'Port Scan', description: 'Network reconnaissance scans', url: '/alerts?attack_type=Port%20Scan' },
                { type: 'category', attack_type: 'XSS', description: 'Cross-site scripting attacks', url: '/alerts?attack_type=XSS' },
                { type: 'category', attack_type: 'Malware', description: 'Malware communication detected', url: '/alerts?attack_type=Malware' },
                { type: 'severity', severity: 'critical', description: 'All critical severity alerts', url: '/alerts?severity=critical' },
                { type: 'severity', severity: 'high', description: 'All high severity alerts', url: '/alerts?severity=high' },
            ];
            
            searchData.push(...commonSearches);
            
            this.searchIndex = searchData;
            
            // Initialize Fuse.js
            if (typeof Fuse !== 'undefined') {
                this.fuse = new Fuse(this.searchIndex, this.fuseOptions);
                this.isInitialized = true;
                console.log(`[Search] Indexed ${searchData.length} items`);
            } else {
                // Load Fuse.js dynamically
                await this.loadFuseJS();
            }
            
        } catch (error) {
            console.error('[Search] Failed to load search data:', error);
            // Use fallback basic search
            this.isInitialized = false;
        }
    }
    
    async loadFuseJS() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/fuse.js@7.0.0/dist/fuse.min.js';
            script.onload = () => {
                this.fuse = new Fuse(this.searchIndex, this.fuseOptions);
                this.isInitialized = true;
                console.log('[Search] Fuse.js loaded dynamically');
                resolve();
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }
    
    handleInput(event) {
        const query = event.target.value.trim();
        
        // Debounce search
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
            this.search(query);
        }, 150);
    }
    
    search(query) {
        if (!query || query.length < 2) {
            this.hideResults();
            return;
        }
        
        let results = [];
        
        if (this.fuse && this.isInitialized) {
            // Fuzzy search with Fuse.js
            results = this.fuse.search(query, { limit: 10 });
        } else {
            // Fallback to basic search
            results = this.basicSearch(query);
        }
        
        this.displayResults(results, query);
    }
    
    basicSearch(query) {
        const lowerQuery = query.toLowerCase();
        return this.searchIndex
            .filter(item => {
                return (
                    (item.source_ip && item.source_ip.toLowerCase().includes(lowerQuery)) ||
                    (item.destination_ip && item.destination_ip.toLowerCase().includes(lowerQuery)) ||
                    (item.attack_type && item.attack_type.toLowerCase().includes(lowerQuery)) ||
                    (item.description && item.description.toLowerCase().includes(lowerQuery)) ||
                    (item.severity && item.severity.toLowerCase().includes(lowerQuery))
                );
            })
            .slice(0, 10)
            .map(item => ({ item, score: 0.5 }));
    }
    
    displayResults(results, query) {
        if (!results.length) {
            this.resultsContainer.innerHTML = `
                <div class="search-no-results">
                    <i class="bi bi-search"></i>
                    <p>No results for "<strong>${this.escapeHtml(query)}</strong>"</p>
                    <small>Try different keywords or check spelling</small>
                </div>
            `;
            this.showResults();
            return;
        }
        
        const html = results.map((result, index) => {
            const item = result.item;
            const score = result.score ? Math.round((1 - result.score) * 100) : 100;
            
            return `
                <a href="${item.url}" class="search-result-item ${index === 0 ? 'active' : ''}" data-index="${index}">
                    <div class="result-icon ${item.type}">
                        ${this.getTypeIcon(item.type, item.severity)}
                    </div>
                    <div class="result-content">
                        <div class="result-title">
                            ${this.highlightMatch(item.attack_type || item.source_ip || 'Unknown', query)}
                        </div>
                        <div class="result-meta">
                            ${item.source_ip ? `<span class="ip">${item.source_ip}</span>` : ''}
                            ${item.severity ? `<span class="severity ${item.severity}">${item.severity}</span>` : ''}
                            ${item.description ? `<span class="desc">${this.truncate(item.description, 50)}</span>` : ''}
                        </div>
                    </div>
                    <div class="result-score">
                        <span class="score-value">${score}%</span>
                        <span class="score-label">match</span>
                    </div>
                </a>
            `;
        }).join('');
        
        this.resultsContainer.innerHTML = `
            <div class="search-results-header">
                <span>${results.length} result${results.length > 1 ? 's' : ''}</span>
                <kbd>↑↓</kbd> to navigate <kbd>↵</kbd> to select
            </div>
            <div class="search-results-list">
                ${html}
            </div>
        `;
        
        this.showResults();
        this.selectedIndex = 0;
    }
    
    getTypeIcon(type, severity) {
        const icons = {
            alert: '<i class="bi bi-exclamation-triangle-fill"></i>',
            source: '<i class="bi bi-geo-alt-fill"></i>',
            category: '<i class="bi bi-tag-fill"></i>',
            severity: '<i class="bi bi-flag-fill"></i>'
        };
        
        let icon = icons[type] || '<i class="bi bi-search"></i>';
        
        if (severity) {
            const colors = {
                critical: '#ef4444',
                high: '#f97316',
                medium: '#eab308',
                low: '#22c55e'
            };
            return `<span style="color: ${colors[severity] || '#6b7280'}">${icon}</span>`;
        }
        
        return icon;
    }
    
    highlightMatch(text, query) {
        if (!text || !query) return text;
        const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    handleKeydown(event) {
        const items = this.resultsContainer.querySelectorAll('.search-result-item');
        if (!items.length) return;
        
        switch (event.key) {
            case 'ArrowDown':
                event.preventDefault();
                this.selectedIndex = Math.min(this.selectedIndex + 1, items.length - 1);
                this.updateSelection(items);
                break;
                
            case 'ArrowUp':
                event.preventDefault();
                this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
                this.updateSelection(items);
                break;
                
            case 'Enter':
                event.preventDefault();
                const selected = items[this.selectedIndex];
                if (selected) {
                    window.location.href = selected.href;
                }
                break;
                
            case 'Escape':
                this.hideResults();
                this.searchInput.blur();
                break;
        }
    }
    
    updateSelection(items) {
        items.forEach((item, index) => {
            item.classList.toggle('active', index === this.selectedIndex);
        });
        
        // Scroll into view
        items[this.selectedIndex]?.scrollIntoView({ block: 'nearest' });
    }
    
    showResults() {
        this.resultsContainer.classList.add('show');
    }
    
    hideResults() {
        this.resultsContainer.classList.remove('show');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    truncate(text, length) {
        if (text.length <= length) return text;
        return text.substr(0, length) + '...';
    }
    
    // Public API to refresh search index
    async refresh() {
        await this.loadSearchData();
    }
}

// Initialize search when DOM is ready
const nidsSearch = new NIDSSearch();

// Export for use in other scripts
window.NIDSSearch = nidsSearch;
