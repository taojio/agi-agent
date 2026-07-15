const App = window.App || {};

App.soul = {
    async loadSoul() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/soul`);
            const data = await response.json();
            this.renderSoul(data.soul || {});
        } catch (error) {
            console.error('Failed to load soul:', error);
        }
    },

    renderSoul(soul) {
        const core = App.core;
        if (core.dom.soulName) core.dom.soulName.textContent = soul.name || 'AGI Agent';
        if (core.dom.soulRole) core.dom.soulRole.textContent = soul.role || '未知角色';
        if (core.dom.soulRigor) core.dom.soulRigor.textContent = typeof soul.rigor === 'number' ? (soul.rigor * 100).toFixed(0) : '--';
        if (core.dom.soulCreativity) core.dom.soulCreativity.textContent = typeof soul.creativity === 'number' ? (soul.creativity * 100).toFixed(0) : '--';

        if (core.dom.soulGoalsContent) {
            const goals = soul.goals || [];
            core.dom.soulGoalsContent.innerHTML = goals.length > 0 ? goals.map(g => `<div>${g}</div>`).join('') : '<div style="color: var(--text-muted);">暂无目标</div>';
        }

        if (core.dom.soulBoundariesContent) {
            const boundaries = soul.boundaries || [];
            core.dom.soulBoundariesContent.innerHTML = boundaries.length > 0 ? boundaries.map(b => `<div>${b}</div>`).join('') : '<div style="color: var(--text-muted);">暂无边界</div>';
        }

        if (core.dom.soulPermissionsContent) {
            const permissions = soul.permissions || [];
            core.dom.soulPermissionsContent.innerHTML = permissions.length > 0 ? permissions.map(p => `<div>${p}</div>`).join('') : '<div style="color: var(--text-muted);">暂无权限</div>';
        }
    },

    async saveSoul() {
        const core = App.core;
        const soulData = {
            name: core.dom.soulName?.textContent || '',
            role: core.dom.soulRole?.textContent || '',
            rigor: parseFloat(core.dom.soulRigor?.textContent || '50') / 100,
            creativity: parseFloat(core.dom.soulCreativity?.textContent || '50') / 100
        };

        try {
            const response = await fetch(`${core.API_BASE}/api/soul`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(soulData)
            });
            const data = await response.json();
            if (data.success) {
                utils.showSuccess('SOUL已保存');
            } else {
                utils.showError('保存失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('保存失败: ' + (error.message || '未知错误'));
        }
    },

    async exportSoul() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/soul/export`);
            const data = await response.json();
            if (data.soul) {
                const blob = new Blob([JSON.stringify(data.soul, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'soul_export.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                utils.showSuccess('SOUL已导出');
            }
        } catch (error) {
            utils.showError('导出失败: ' + (error.message || '未知错误'));
        }
    },

    init() {
        this.loadSoul();
    }
};

window.App = App;
