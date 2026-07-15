const App = window.App || {};

App.skills = {
    async loadSkills() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/skills`);
            const data = await response.json();
            this.renderSkills(data.skills || []);
        } catch (error) {
            console.error('Failed to load skills:', error);
            core.dom.skillsList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">加载失败</div>';
        }
    },

    renderSkills(skills) {
        const core = App.core;
        if (skills.length === 0) {
            core.dom.skillsList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">暂无技能</div>';
            return;
        }

        core.dom.skillsList.innerHTML = skills.map(skill => `
            <div class="skill-item">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: bold;">${skill.name}</span>
                    <span style="padding: 2px 8px; border-radius: 10px; font-size: 11px; background: ${skill.enabled ? '#4CAF5020' : '#9E9E9E20'}; color: ${skill.enabled ? '#4CAF50' : '#9E9E9E'};">${skill.enabled ? '已启用' : '已禁用'}</span>
                </div>
                <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">${skill.description || ''}</div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">类型: ${skill.type || '未知'} | 版本: ${skill.version || '1.0'}</div>
            </div>
        `).join('');
    },

    init() {
        this.loadSkills();
    }
};

window.App = App;
