const App = window.App || {};

App.memory = {
    async loadMemories(tier = 'L1') {
        const core = App.core;
        core.state.currentMemoryTier = tier;
        try {
            const response = await fetch(`${core.API_BASE}/api/memory/tier/${tier}`);
            const data = await response.json();
            this.renderMemories(data.memories || []);
            this.updateMemoryStats(data.stats || {});
        } catch (error) {
            console.error('Failed to load memories:', error);
            core.dom.memoryList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">加载失败</div>';
        }
    },

    updateMemoryStats(stats) {
        const core = App.core;
        core.dom.memoryStats.innerHTML = `
            <span>总计: ${stats.total || 0}</span>
            <span>最近: ${stats.recent_count || 0}</span>
            <span>访问次数: ${stats.access_count || 0}</span>
        `;
    },

    renderMemories(memories) {
        const core = App.core;
        if (memories.length === 0) {
            core.dom.memoryList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">暂无记忆</div>';
            return;
        }

        core.dom.memoryList.innerHTML = memories.map(memory => `
            <div class="memory-item">
                <div class="memory-header">
                    <span class="memory-tier">${memory.tier}</span>
                    <span class="memory-time">${core.formatTime(memory.timestamp)}</span>
                    <span class="memory-confidence">置信度: ${typeof memory.confidence === 'number' ? (memory.confidence * 100).toFixed(0) : '--'}%</span>
                </div>
                <div class="memory-content">${memory.content}</div>
                ${memory.importance ? `<div class="memory-importance">重要性: ${memory.importance}</div>` : ''}
                ${memory.context ? `<div class="memory-context">上下文: ${memory.context}</div>` : ''}
            </div>
        `).join('');
    },

    async searchMemories(query) {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/memory/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, tier: core.state.currentMemoryTier })
            });
            const data = await response.json();
            this.renderMemories(data.memories || []);
        } catch (error) {
            console.error('Search failed:', error);
        }
    },

    async addMemory() {
        const core = App.core;
        const content = await core.showModalPrompt('添加记忆', '输入记忆内容');
        if (!content) return;

        try {
            const response = await fetch(`${core.API_BASE}/api/memory`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: content, tier: core.state.currentMemoryTier })
            });
            const data = await response.json();
            if (data.success) {
                this.loadMemories(core.state.currentMemoryTier);
                utils.showSuccess('记忆已添加');
            } else {
                utils.showError('添加失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('添加失败: ' + (error.message || '未知错误'));
        }
    },

    renderMemoryTimelineView(memories) {
        const core = App.core;
        core.dom.memoryList.innerHTML = memories.map(memory => `
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                    <div class="timeline-time">${new Date(memory.timestamp).toLocaleString()}</div>
                    <div>${memory.content}</div>
                </div>
            </div>
        `).join('');
    },

    renderMemoryGraphView(memories) {
        const core = App.core;
        core.dom.memoryList.innerHTML = '<div style="color: var(--text-secondary); padding: 20px;">图谱视图 - 记忆关联图</div>';
    },

    renderMemoryListView(memories) {
        const core = App.core;
        this.renderMemories(memories);
    },

    handleMemoryUpdate(data) {
        this.loadMemories(App.core.state.currentMemoryTier);
    },

    init() {
        this.loadMemories('L1');
    }
};

window.renderMemoryTimelineView = (memories) => App.memory.renderMemoryTimelineView(memories);
window.renderMemoryGraphView = (memories) => App.memory.renderMemoryGraphView(memories);
window.renderMemoryListView = (memories) => App.memory.renderMemoryListView(memories);

window.App = App;
