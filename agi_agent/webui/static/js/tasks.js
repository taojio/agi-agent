const App = window.App || {};

App.tasks = {
    async loadTasks() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/tasks`);
            const data = await response.json();
            this.renderTasks(data.tasks || []);
            this.updateTaskStats(data.stats || {});
        } catch (error) {
            console.error('Failed to load tasks:', error);
        }
    },

    updateTaskStats(stats) {
        const core = App.core;
        if (core.dom.taskPending) core.dom.taskPending.textContent = stats.pending || 0;
        if (core.dom.taskInProgress) core.dom.taskInProgress.textContent = stats.in_progress || 0;
        if (core.dom.taskCompleted) core.dom.taskCompleted.textContent = stats.completed || 0;
    },

    renderTasks(tasks) {
        const core = App.core;
        if (tasks.length === 0) {
            core.dom.tasksStats.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">暂无任务</div>';
            return;
        }

        const pending = tasks.filter(t => t.status === 'pending');
        const inProgress = tasks.filter(t => t.status === 'in_progress');
        const completed = tasks.filter(t => t.status === 'completed');

        core.dom.tasksStats.innerHTML = `
            <div class="task-list">
                <h4>待处理 (${pending.length})</h4>
                ${pending.map(t => this.renderTaskItem(t)).join('')}
            </div>
            <div class="task-list">
                <h4>进行中 (${inProgress.length})</h4>
                ${inProgress.map(t => this.renderTaskItem(t)).join('')}
            </div>
            <div class="task-list">
                <h4>已完成 (${completed.length})</h4>
                ${completed.map(t => this.renderTaskItem(t)).join('')}
            </div>
        `;
    },

    renderTaskItem(task) {
        const statusColors = {
            pending: '#FF9800',
            in_progress: '#2196F3',
            completed: '#4CAF50'
        };
        return `
            <div class="task-item">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="width: 12px; height: 12px; border-radius: 50%; background: ${statusColors[task.status] || '#9E9E9E'};"></span>
                    <span>${task.title}</span>
                </div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">${task.description || ''}</div>
            </div>
        `;
    },

    async submitTask() {
        const core = App.core;
        const title = await core.showModalPrompt('提交任务', '输入任务标题');
        if (!title) return;

        const description = await core.showModalPrompt('任务描述', '输入任务描述');

        try {
            const response = await fetch(`${core.API_BASE}/api/tasks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: title, description: description || '' })
            });
            const data = await response.json();
            if (data.success) {
                this.loadTasks();
                utils.showSuccess('任务已提交');
            } else {
                utils.showError('提交失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('提交失败: ' + (error.message || '未知错误'));
        }
    },

    handleTasksUpdate(data) {
        this.loadTasks();
    },

    init() {
        this.loadTasks();
    }
};

window.App = App;
