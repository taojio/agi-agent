const App = window.App || {};

App.agents = {
    async loadAgents() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/agents`);
            const data = await response.json();
            this.renderAgents(data.agents || []);
        } catch (error) {
            console.error('Failed to load agents:', error);
            core.dom.agentList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">加载失败</div>';
        }
    },

    renderAgents(agents) {
        const core = App.core;
        if (agents.length === 0) {
            core.dom.agentList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">暂无Agent</div>';
            return;
        }

        core.dom.agentList.innerHTML = agents.map(agent => `
            <div class="agent-item" onclick="loadAgentDetail('${agent.name}')">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: bold;">${agent.name}</span>
                    <span style="padding: 2px 8px; border-radius: 10px; font-size: 11px; background: ${agent.running ? '#4CAF5020' : '#f4433620'}; color: ${agent.running ? '#4CAF50' : '#f44336'};">${agent.running ? '运行中' : '已停止'}</span>
                </div>
                <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">${agent.description || ''}</div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">类型: ${agent.type || '未知'} | 步骤: ${agent.step || 0}</div>
            </div>
        `).join('');
    },

    async loadAgentDetail(name) {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/agents/${name}`);
            const data = await response.json();
            if (data.agent) {
                core.dom.agentDetailName.textContent = data.agent.name;
                core.dom.agentDetailContent.innerHTML = `
                    <div style="padding: 10px;">
                        <div style="margin-bottom: 10px;"><strong>描述:</strong> ${data.agent.description || '无'}</div>
                        <div style="margin-bottom: 10px;"><strong>类型:</strong> ${data.agent.type || '未知'}</div>
                        <div style="margin-bottom: 10px;"><strong>状态:</strong> <span style="color: ${data.agent.running ? '#4CAF50' : '#f44336'};">${data.agent.running ? '运行中' : '已停止'}</span></div>
                        <div style="margin-bottom: 10px;"><strong>步骤:</strong> ${data.agent.step || 0}</div>
                        <div style="margin-bottom: 10px;"><strong>维度:</strong> ${data.agent.dimension || 'N/A'}</div>
                        <div style="margin-bottom: 10px;"><strong>创建时间:</strong> ${core.formatTime(data.agent.created_at)}</div>
                        <div><strong>更新时间:</strong> ${core.formatTime(data.agent.updated_at)}</div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load agent detail:', error);
            core.dom.agentDetailContent.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">加载失败</div>';
        }
    },

    async createNewAgent() {
        const core = App.core;
        const name = await core.showModalPrompt('创建Agent', '输入Agent名称');
        if (!name) return;

        try {
            const response = await fetch(`${core.API_BASE}/api/agents`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            });
            const data = await response.json();
            if (data.success) {
                this.loadAgents();
                utils.showSuccess('Agent已创建');
            } else {
                utils.showError('创建失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('创建失败: ' + (error.message || '未知错误'));
        }
    },

    init() {
        this.loadAgents();
    }
};

window.loadAgentDetail = (name) => App.agents.loadAgentDetail(name);

window.App = App;
