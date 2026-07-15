const App = window.App || {};

App.plugins = {
    async loadPlugins() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/plugins/available`);
            const data = await response.json();
            this.renderPlugins(data.plugins || []);
        } catch (error) {
            console.error('Failed to load plugins:', error);
            core.dom.configPluginList.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 40px;">加载失败</div>';
        }
    },

    renderPlugins(plugins) {
        const core = App.core;
        if (plugins.length === 0) {
            core.dom.configPluginList.innerHTML = `
                <div style="color: var(--text-muted); text-align: center; padding: 40px;">暂无插件</div>
                <div style="display: flex; gap: 10px; justify-content: center; margin-top: 20px;">
                    <button class="btn btn-primary" onclick="loadAllPlugins()">加载所有插件</button>
                </div>
            `;
            return;
        }

        const stats = {
            total: plugins.length,
            loaded: plugins.filter(p => p.loaded).length,
            active: plugins.filter(p => p.status === 'active').length,
            unloaded: plugins.filter(p => !p.loaded).length
        };

        core.dom.configPluginList.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <div style="display: flex; gap: 20px;">
                    <span style="font-size: 13px; color: var(--text-secondary);">总计: ${stats.total}</span>
                    <span style="font-size: 13px; color: #4CAF50;">已加载: ${stats.loaded}</span>
                    <span style="font-size: 13px; color: #2196F3;">活跃: ${stats.active}</span>
                    <span style="font-size: 13px; color: var(--text-muted);">未加载: ${stats.unloaded}</span>
                </div>
                <div style="display: flex; gap: 10px;">
                    <button class="btn btn-secondary btn-sm" onclick="loadAllPlugins()">加载全部</button>
                    <button class="btn btn-secondary btn-sm" onclick="activateAllPlugins()">激活全部</button>
                    <button class="btn btn-secondary btn-sm" onclick="deactivateAllPlugins()">停用全部</button>
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                ${plugins.map(plugin => {
                    const isLoaded = plugin.loaded;
                    const isActive = plugin.status === 'active';
                    const statusText = isActive ? '活跃' : (isLoaded ? '已加载' : '未加载');
                    const statusColor = isActive ? '#4CAF50' : (isLoaded ? '#FF9800' : '#9E9E9E');

                    return `
                        <div class="plugin-item" style="padding: 16px; background: var(--bg-secondary); border-radius: var(--radius-md); border-left: 4px solid ${statusColor};">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div style="flex: 1;">
                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <h4 style="margin: 0; color: var(--text-primary);">${plugin.name || '未命名插件'}</h4>
                                        <span style="padding: 2px 8px; border-radius: 10px; font-size: 11px; background: ${statusColor}20; color: ${statusColor};">${statusText}</span>
                                    </div>
                                    <p style="margin: 8px 0; color: var(--text-secondary); font-size: 13px;">${plugin.description || '无描述'}</p>
                                    <div style="display: flex; gap: 15px; font-size: 12px; color: var(--text-muted);">
                                        <span>版本: ${plugin.version || '1.0'}</span>
                                        <span>类型: ${plugin.type || 'processor'}</span>
                                        ${plugin.priority ? `<span>优先级: ${plugin.priority}</span>` : ''}
                                        ${plugin.load_time ? `<span>加载时间: ${new Date(plugin.load_time).toLocaleTimeString()}</span>` : ''}
                                    </div>
                                    ${plugin.dependencies && plugin.dependencies.length > 0 ? `
                                        <div style="margin-top: 8px; font-size: 12px; color: #FF9800;">
                                            依赖: ${plugin.dependencies.join(', ')}
                                        </div>
                                    ` : ''}
                                </div>
                                <div style="display: flex; flex-direction: column; gap: 6px; margin-left: 20px;">
                                    ${!isLoaded ? `
                                        <button class="btn btn-primary btn-sm" onclick="loadPlugin('${plugin.name}')">加载</button>
                                    ` : `
                                        <button class="btn ${isActive ? 'btn-warning' : 'btn-success'} btn-sm" onclick="togglePlugin('${plugin.name}', ${!isActive})">
                                            ${isActive ? '停用' : '激活'}
                                        </button>
                                        <button class="btn btn-secondary btn-sm" onclick="reloadPlugin('${plugin.name}')">重载</button>
                                        <button class="btn btn-danger btn-sm" onclick="unloadPlugin('${plugin.name}')">卸载</button>
                                    `}
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    },

    async loadPlugin(pluginName) {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/plugins/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: pluginName })
            });
            const data = await response.json();
            if (data.success) {
                this.loadPlugins();
                utils.showSuccess(`插件 "${pluginName}" 加载成功`);
            } else {
                utils.showError(`加载失败: ${data.error}`);
            }
        } catch (error) {
            console.error('Plugin load failed:', error);
            utils.showError('加载失败: ' + (error.message || '未知错误'));
        }
    },

    async unloadPlugin(pluginName) {
        if (!confirm(`确定要卸载插件 "${pluginName}" 吗？`)) return;
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/plugins/${pluginName}/unload`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                this.loadPlugins();
                utils.showSuccess(`插件 "${pluginName}" 卸载成功`);
            } else {
                utils.showError(`卸载失败: ${data.error}`);
            }
        } catch (error) {
            console.error('Plugin unload failed:', error);
            utils.showError('卸载失败: ' + (error.message || '未知错误'));
        }
    },

    async togglePlugin(pluginName, activate) {
        const core = App.core;
        try {
            const endpoint = activate ? `/api/plugins/${pluginName}/activate` : `/api/plugins/${pluginName}/deactivate`;
            const response = await fetch(`${core.API_BASE}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                this.loadPlugins();
                utils.showSuccess(`插件 "${pluginName}" ${activate ? '已激活' : '已停用'}`);
            } else {
                utils.showError(`${activate ? '激活' : '停用'}失败: ${data.error}`);
            }
        } catch (error) {
            console.error('Plugin toggle failed:', error);
            utils.showError('操作失败: ' + (error.message || '未知错误'));
        }
    },

    async reloadPlugin(pluginName) {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/plugins/${pluginName}/reload`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                this.loadPlugins();
                utils.showSuccess(`插件 "${pluginName}" 重载成功`);
            } else {
                utils.showError(`重载失败: ${data.error}`);
            }
        } catch (error) {
            console.error('Plugin reload failed:', error);
            utils.showError('重载失败: ' + (error.message || '未知错误'));
        }
    },

    async loadAllPlugins() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/plugins/load_all`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            this.loadPlugins();
            utils.showSuccess('所有插件已加载');
        } catch (error) {
            console.error('Load all plugins failed:', error);
            utils.showError('加载全部失败: ' + (error.message || '未知错误'));
        }
    },

    async activateAllPlugins() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/plugins/activate_all`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            this.loadPlugins();
            utils.showSuccess('所有插件已激活');
        } catch (error) {
            console.error('Activate all plugins failed:', error);
            utils.showError('激活插件失败: ' + (error.message || '未知错误'));
        }
    },

    async deactivateAllPlugins() {
        if (!confirm('确定要停用所有插件吗？')) return;
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/plugins/deactivate_all`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            this.loadPlugins();
            utils.showSuccess('所有插件已停用');
        } catch (error) {
            console.error('Deactivate all plugins failed:', error);
            utils.showError('停用插件失败: ' + (error.message || '未知错误'));
        }
    },

    init() {}
};

window.loadAllPlugins = () => App.plugins.loadAllPlugins();
window.activateAllPlugins = () => App.plugins.activateAllPlugins();
window.deactivateAllPlugins = () => App.plugins.deactivateAllPlugins();
window.loadPlugin = (name) => App.plugins.loadPlugin(name);
window.togglePlugin = (name, activate) => App.plugins.togglePlugin(name, activate);
window.reloadPlugin = (name) => App.plugins.reloadPlugin(name);
window.unloadPlugin = (name) => App.plugins.unloadPlugin(name);

window.App = App;
