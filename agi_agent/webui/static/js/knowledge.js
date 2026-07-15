const App = window.App || {};

App.knowledge = {
    async loadKnowledge() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/knowledge`);
            const data = await response.json();
            this.renderKnowledge(data.knowledge || {});
        } catch (error) {
            console.error('Failed to load knowledge:', error);
        }
    },

    renderKnowledge(data) {
        const core = App.core;
        const stats = data.stats || {};
        core.dom.knowledgeStats.innerHTML = `
            <span>节点数: ${stats.nodes || 0}</span>
            <span>边数: ${stats.edges || 0}</span>
            <span>图谱大小: ${stats.size || 0} KB</span>
        `;

        if (data.graph) {
            this.renderKnowledgeGraphView(data.graph);
        }
    },

    renderKnowledgeGraphView(graph) {
        const core = App.core;
        const nodes = graph.nodes || [];
        const edges = graph.edges || [];

        core.dom.knowledgeGraph.innerHTML = `
            <div style="padding: 10px;">
                <div style="font-weight: bold; margin-bottom: 10px;">知识图谱</div>
                <div style="font-size: 12px; color: var(--text-muted);">节点: ${nodes.length} | 边: ${edges.length}</div>
                <div style="margin-top: 10px; max-height: 300px; overflow-y: auto;">
                    ${nodes.slice(0, 20).map(n => `
                        <div style="padding: 6px; border-bottom: 1px solid var(--border-color);">
                            <span style="font-weight: bold;">${n.label}</span>
                            <span style="color: var(--text-muted); margin-left: 10px;">${n.type || '实体'}</span>
                        </div>
                    `).join('')}
                    ${nodes.length > 20 ? `<div style="padding: 10px; text-align: center; color: var(--text-muted);">...还有 ${nodes.length - 20} 个节点</div>` : ''}
                </div>
            </div>
        `;
    },

    renderKnowledgeListView(graph) {
        const core = App.core;
        const nodes = graph.nodes || [];
        core.dom.knowledgeGraph.innerHTML = nodes.map(n => `
            <div style="padding: 8px; border-bottom: 1px solid var(--border-color);">
                <div style="font-weight: bold;">${n.label}</div>
                <div style="font-size: 12px; color: var(--text-muted);">${n.description || ''}</div>
            </div>
        `).join('');
    },

    handleKnowledgeUpdate(data) {
        this.loadKnowledge();
    },

    init() {
        this.loadKnowledge();
    }
};

window.renderKnowledgeGraphView = (graph) => App.knowledge.renderKnowledgeGraphView(graph);
window.renderKnowledgeListView = (graph) => App.knowledge.renderKnowledgeListView(graph);

window.App = App;
