const App = window.App || {};

App.evolution = {
    async loadEvolution() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/evolution`);
            const data = await response.json();
            this.renderEvolution(data.evolution || {});
        } catch (error) {
            console.error('Failed to load evolution:', error);
        }
    },

    renderEvolution(data) {
        const core = App.core;
        core.dom.evolutionStats.innerHTML = `
            <span>当前代: ${data.generation || 0}</span>
            <span>适应度: ${typeof data.fitness === 'number' ? data.fitness.toFixed(4) : '--'}</span>
            <span>变异率: ${typeof data.mutation_rate === 'number' ? (data.mutation_rate * 100).toFixed(0) : '--'}%</span>
        `;

        if (data.proposals && data.proposals.length > 0) {
            core.dom.proposalList.innerHTML = data.proposals.map(p => `
                <div class="proposal-item">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: bold;">${p.title}</span>
                        <span style="padding: 2px 8px; border-radius: 10px; font-size: 11px; background: ${p.approved ? '#4CAF5020' : '#FF980020'}; color: ${p.approved ? '#4CAF50' : '#FF9800'};">${p.approved ? '已批准' : '待审核'}</span>
                    </div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">${p.description}</div>
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">${p.type || '未知类型'} | ${core.formatTime(p.timestamp)}</div>
                </div>
            `).join('');
        } else {
            core.dom.proposalList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">暂无进化提案</div>';
        }
    },

    async runEvolution() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/evolution/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                utils.showSuccess('进化已启动');
                this.loadEvolution();
            } else {
                utils.showError('启动失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('启动失败: ' + (error.message || '未知错误'));
        }
    },

    async generateSkill() {
        const core = App.core;
        const skillName = await core.showModalPrompt('生成技能', '输入技能名称');
        if (!skillName) return;

        try {
            const response = await fetch(`${core.API_BASE}/api/skills/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: skillName })
            });
            const data = await response.json();
            if (data.success) {
                utils.showSuccess(`技能 "${skillName}" 生成成功`);
                if (App.skills) {
                    App.skills.loadSkills();
                }
            } else {
                utils.showError('生成失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('生成失败: ' + (error.message || '未知错误'));
        }
    },

    init() {
        this.loadEvolution();
    }
};

window.App = App;
