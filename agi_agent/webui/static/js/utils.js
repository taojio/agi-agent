const API_BASE = window.location.origin;

const utils = {
    showToast(message, type = 'info', duration = 3000) {
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${this.getToastIcon(type)}</span>
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
        `;
        
        toastContainer.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.add('toast-fadeout');
                setTimeout(() => toast.remove(), 300);
            }
        }, duration);
    },
    
    getToastIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️',
            loading: '⏳'
        };
        return icons[type] || icons.info;
    },
    
    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container';
        document.body.appendChild(container);
        return container;
    },
    
    showSuccess(message) {
        this.showToast(message, 'success');
    },
    
    showError(message) {
        this.showToast(message, 'error');
    },
    
    showWarning(message) {
        this.showToast(message, 'warning');
    },
    
    showInfo(message) {
        this.showToast(message, 'info');
    },
    
    createSkeleton(container, type = 'card') {
        container.innerHTML = `
            <div class="skeleton-container">
                ${this.getSkeletonHTML(type)}
            </div>
        `;
    },
    
    getSkeletonHTML(type) {
        switch (type) {
            case 'card':
                return `
                    <div class="skeleton-card">
                        <div class="skeleton-avatar"></div>
                        <div class="skeleton-text-line"></div>
                        <div class="skeleton-text-line short"></div>
                        <div class="skeleton-text-line short"></div>
                    </div>
                `;
            case 'list':
                return `
                    <div class="skeleton-list">
                        <div class="skeleton-list-item">
                            <div class="skeleton-circle"></div>
                            <div class="skeleton-text-line"></div>
                            <div class="skeleton-text-line short"></div>
                        </div>
                        <div class="skeleton-list-item">
                            <div class="skeleton-circle"></div>
                            <div class="skeleton-text-line"></div>
                            <div class="skeleton-text-line short"></div>
                        </div>
                        <div class="skeleton-list-item">
                            <div class="skeleton-circle"></div>
                            <div class="skeleton-text-line"></div>
                            <div class="skeleton-text-line short"></div>
                        </div>
                    </div>
                `;
            case 'stats':
                return `
                    <div class="skeleton-stats">
                        <div class="skeleton-stat-item">
                            <div class="skeleton-icon"></div>
                            <div class="skeleton-text-line large"></div>
                            <div class="skeleton-text-line short"></div>
                        </div>
                        <div class="skeleton-stat-item">
                            <div class="skeleton-icon"></div>
                            <div class="skeleton-text-line large"></div>
                            <div class="skeleton-text-line short"></div>
                        </div>
                        <div class="skeleton-stat-item">
                            <div class="skeleton-icon"></div>
                            <div class="skeleton-text-line large"></div>
                            <div class="skeleton-text-line short"></div>
                        </div>
                    </div>
                `;
            default:
                return `<div class="skeleton-basic"><div class="skeleton-text-line"></div></div>`;
        }
    },
    
    formatTime(timestamp) {
        if (!timestamp) return '未知';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return '刚刚';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
        if (diff < 604800000) return `${Math.floor(diff / 86400000)}天前`;
        
        return date.toLocaleDateString('zh-CN');
    },
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func(...args);
                inThrottle = true;
                setTimeout(() => (inThrottle = false), limit);
            }
        };
    },
    
    async fetchAPI(endpoint, options = {}) {
        try {
            const response = await fetch(`${API_BASE}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }
            
            return response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    },
    
    escapeHTML(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    copyToClipboard(text) {
        return navigator.clipboard.writeText(text).then(() => {
            this.showSuccess('已复制到剪贴板');
        }).catch(() => {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            this.showSuccess('已复制到剪贴板');
        });
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = utils;
}