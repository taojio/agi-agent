const keyboard = {
    shortcuts: {
        'ctrl+k': { action: 'openCommandPalette', description: '打开命令面板' },
        'ctrl+n': { action: 'createNewSession', description: '新建会话' },
        'ctrl+m': { action: 'switchView', args: ['memory'], description: '切换到记忆' },
        'ctrl+t': { action: 'switchView', args: ['tasks'], description: '切换到任务' },
        'ctrl+e': { action: 'switchView', args: ['evolution'], description: '切换到进化' },
        'ctrl+s': { action: 'switchView', args: ['soul'], description: '切换到SOUL' },
        'ctrl+f': { action: 'switchView', args: ['security'], description: '切换到安全' },
        'ctrl+i': { action: 'switchView', args: ['selfimprovement'], description: '切换到自我改进' },
        'ctrl+c': { action: 'switchView', args: ['config'], description: '切换到配置' },
        'ctrl+1': { action: 'switchView', args: ['overview'], description: '切换到概览' },
        'ctrl+2': { action: 'switchView', args: ['chat'], description: '切换到聊天' },
        'ctrl+3': { action: 'switchView', args: ['memory'], description: '切换到记忆' },
        'ctrl+4': { action: 'switchView', args: ['tasks'], description: '切换到任务' },
        'ctrl+5': { action: 'switchView', args: ['evolution'], description: '切换到进化' },
        'ctrl+6': { action: 'switchView', args: ['knowledge'], description: '切换到知识图谱' },
        'ctrl+7': { action: 'switchView', args: ['synaptic'], description: '切换到突触总线' },
        'escape': { action: 'closeCommandPalette', description: '关闭命令面板/弹窗' },
        'ctrl+r': { action: 'refreshAllData', description: '刷新所有数据' },
        'ctrl+d': { action: 'runAgentStep', description: '执行单步' },
        'ctrl+enter': { action: 'sendMessage', description: '发送消息' },
        'ctrl+l': { action: 'clearChat', description: '清空聊天' },
        'ctrl+shift+c': { action: 'copyToClipboard', args: ['chat'], description: '复制聊天内容' }
    },
    
    init() {
        document.addEventListener('keydown', (e) => {
            const keyCombo = this.getKeyCombo(e);
            const shortcut = this.shortcuts[keyCombo];
            
            if (shortcut) {
                e.preventDefault();
                this.executeShortcut(shortcut);
            }
        });
    },
    
    getKeyCombo(e) {
        const parts = [];
        if (e.ctrlKey || e.metaKey) parts.push('ctrl');
        if (e.shiftKey) parts.push('shift');
        if (e.altKey) parts.push('alt');
        
        const key = e.key.toLowerCase();
        if (!['control', 'shift', 'alt', 'meta'].includes(key)) {
            parts.push(key);
        }
        
        return parts.join('+');
    },
    
    executeShortcut(shortcut) {
        const action = shortcut.action;
        const args = shortcut.args || [];
        
        if (typeof window[action] === 'function') {
            window[action](...args);
        } else {
            console.warn(`Action not found: ${action}`);
        }
    },
    
    getShortcutsList() {
        return Object.entries(this.shortcuts).map(([key, value]) => ({
            key,
            description: value.description
        }));
    },
    
    showShortcutsHelp() {
        const shortcuts = this.getShortcutsList();
        const helpHTML = shortcuts.map(s => 
            `<div style="display: flex; justify-content: space-between; padding: 8px;">
                <span>${s.description}</span>
                <kbd style="padding: 2px 6px; background: var(--bg-tertiary); border-radius: 4px; font-size: 12px;">${s.key}</kbd>
            </div>`
        ).join('');
        
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.style.display = 'flex';
        modal.innerHTML = `
            <div class="modal-dialog" style="width: 400px;">
                <div class="modal-header">
                    <span class="modal-title">键盘快捷键</span>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                </div>
                <div class="modal-body" style="max-height: 400px; overflow-y: auto;">${helpHTML}</div>
                <div class="modal-footer">
                    <button class="modal-btn modal-btn-confirm" onclick="this.closest('.modal-overlay').remove()">关闭</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = keyboard;
}