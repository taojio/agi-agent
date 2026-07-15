const App = window.App || {};

App.sessions = {
    async loadSessions() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/sessions`);
            const data = await response.json();
            core.state.sessions = data.sessions || [];
            this.renderSessions();
        } catch (error) {
            console.error('Failed to load sessions:', error);
            core.dom.sessionList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">加载失败</div>';
        }
    },

    renderSessions() {
        const core = App.core;
        const sessions = core.state.sessions;

        if (sessions.length === 0) {
            core.dom.sessionList.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">暂无会话</div>';
            return;
        }

        core.dom.sessionList.innerHTML = sessions.map(session => `
            <div class="session-item ${session.id === core.state.currentSessionId ? 'active' : ''}" 
                 onclick="loadSession('${session.id}')">
                <div class="session-name">${session.name || '未命名会话'}</div>
                <div class="session-date">${core.formatTime(session.created_at)}</div>
                <div class="session-messages">${session.message_count || 0} 条消息</div>
            </div>
        `).join('');
    },

    async loadAllSessions() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/sessions`);
            const data = await response.json();
            this.renderSessionTable(data.sessions || []);
        } catch (error) {
            console.error('Failed to load all sessions:', error);
        }
    },

    renderSessionTable(sessions) {
        const core = App.core;
        if (sessions.length === 0) {
            core.dom.sessionsTable.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">暂无会话</td></tr>';
            return;
        }

        core.dom.sessionsTable.innerHTML = sessions.map(session => `
            <tr>
                <td>${session.id}</td>
                <td>${session.name || '未命名会话'}</td>
                <td>${session.message_count || 0}</td>
                <td>${core.formatTime(session.created_at)}</td>
                <td>${core.formatTime(session.updated_at)}</td>
            </tr>
        `).join('');
    },

    async saveAllSessions() {
        try {
            const response = await fetch(`${core.API_BASE}/api/sessions/save_all`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                utils.showSuccess('所有会话已保存');
            } else {
                utils.showError('保存失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('保存失败: ' + (error.message || '未知错误'));
        }
    },

    async exportAllSessions() {
        try {
            const response = await fetch(`${core.API_BASE}/api/sessions/export_all`);
            const data = await response.json();
            if (data.sessions) {
                const blob = new Blob([JSON.stringify(data.sessions, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `all_sessions_${Date.now()}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                utils.showSuccess('所有会话已导出');
            }
        } catch (error) {
            utils.showError('导出失败: ' + (error.message || '未知错误'));
        }
    },

    init() {
        this.loadSessions();
    }
};

window.App = App;
