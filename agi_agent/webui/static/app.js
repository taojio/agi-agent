const API_BASE = window.location.origin;

let isAgentRunning = false;
let isSensorsEnabled = true;
let isVoiceRecording = false;
let recognition = null;

let serverAvailable = true;
let lastServerErrorTime = 0;

let currentChannel = 'general';
let channelMessages = {};

let realtimeSocket = null;
let realtimeConnected = false;
let busHasData = false;

function connectRealtimeSocket() {
    if (realtimeSocket) {
        realtimeSocket.close();
    }
    
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    realtimeSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/realtime`);
    
    realtimeSocket.onopen = () => {
        realtimeConnected = true;
        console.log('Realtime WebSocket connected');
    };
    
    realtimeSocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'realtime_update') {
                processRealtimeUpdates(data.updates);
            }
        } catch (error) {
            console.error('Failed to parse realtime update:', error);
        }
    };
    
    realtimeSocket.onclose = () => {
        realtimeConnected = false;
        console.log('Realtime WebSocket disconnected');
        setTimeout(connectRealtimeSocket, 5000);
    };
    
    realtimeSocket.onerror = (error) => {
        console.error('Realtime WebSocket error:', error);
    };
}

function processRealtimeUpdates(updates) {
    updates.forEach(update => {
        switch (update.module) {
            case 'synaptic':
                handleSynapticUpdate(update);
                break;
            case 'agent':
                handleAgentUpdate(update.data);
                break;
            case 'tasks':
                handleTasksUpdate(update.data);
                break;
            case 'memory':
                handleMemoryUpdate(update.data);
                break;
            case 'knowledge':
                handleKnowledgeUpdate(update.data);
                break;
        }
    });
}

function handleSynapticUpdate(update) {
    busHasData = update.has_data;
    updateBusStatusIndicator();
    
    if (update.has_data && update.data) {
        renderModuleActivity(update.data);
        renderSynapticStats(update.data);
    } else {
        renderEmptyBusState();
    }
}

function handleAgentUpdate(data) {
    updateAgentInfo(data);
    if (dom.statFreeEnergy) dom.statFreeEnergy.textContent = typeof data.free_energy === 'number' ? data.free_energy.toFixed(2) : '0.00';
    if (dom.statConfidence) dom.statConfidence.textContent = typeof data.confidence === 'number' ? `${(data.confidence * 100).toFixed(0)}%` : '0%';
}

function handleTasksUpdate(data) {
    if (data.board) {
        renderTasksStats(data);
    }
}

function handleMemoryUpdate(data) {
    renderMemoryStats(data);
}

function handleKnowledgeUpdate(data) {
    if (dom.statKnowledge) dom.statKnowledge.textContent = data.nodes || 0;
    if (dom.knowledgeStats) {
        dom.knowledgeStats.innerHTML = `
            <span>节点: <strong>${data.nodes || 0}</strong></span>
            <span>边: <strong>${data.edges || 0}</strong></span>
        `;
    }
}

function updateBusStatusIndicator() {
    if (!dom.gatewayStatus) return;
    
    const busIndicator = document.getElementById('busStatusIndicator');
    if (!busIndicator) {
        const indicator = document.createElement('div');
        indicator.id = 'busStatusIndicator';
        indicator.className = 'bus-status-indicator';
        indicator.style.cssText = `
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 12px;
        `;
        dom.gatewayStatus.appendChild(indicator);
    }
    
    const statusIndicator = document.getElementById('busStatusIndicator');
    if (busHasData) {
        statusIndicator.innerHTML = '<span style="color: var(--success-color);">●</span><span style="color: var(--success-color);">总线有数据</span>';
        statusIndicator.style.background = 'rgba(34, 197, 94, 0.1)';
    } else {
        statusIndicator.innerHTML = '<span style="color: var(--warning-color);">⚠</span><span style="color: var(--warning-color);">总线无数据</span>';
        statusIndicator.style.background = 'rgba(251, 191, 36, 0.1)';
    }
}

function renderEmptyBusState() {
    if (dom.synapticStats) {
        dom.synapticStats.innerHTML = `
            <div class="empty-state-warning">
                <div class="warning-icon">⚠</div>
                <div class="warning-text">总线无数据</div>
                <div class="warning-hint">系统等待数据输入...</div>
            </div>
        `;
    }
    
    if (dom.moduleActivityList) {
        dom.moduleActivityList.innerHTML = `
            <div class="empty-state-warning">
                <div class="warning-icon">🔌</div>
                <div class="warning-text">暂无模块活动</div>
                <div class="warning-hint">启动Agent或输入数据以激活总线</div>
            </div>
        `;
    }
    
    if (dom.synapticGraph) {
        dom.synapticGraph.innerHTML = `
            <div class="empty-state-warning">
                <div class="warning-icon">🕸️</div>
                <div class="warning-text">突触网络未激活</div>
            </div>
        `;
    }
    
    if (dom.signalFlowChart) {
        dom.signalFlowChart.innerHTML = `
            <div class="empty-state-warning">
                <div class="warning-icon">💤</div>
                <div class="warning-text">无信号流动</div>
            </div>
        `;
    }
    
    if (dom.oscillatorDisplay) {
        dom.oscillatorDisplay.innerHTML = `
            <div class="empty-state-warning">
                <div class="warning-icon">⏸️</div>
                <div class="warning-text">振荡器未运行</div>
            </div>
        `;
    }
}

function renderModuleActivity(data) {
    if (!dom.moduleActivityList) return;
    
    const modules = [
        { name: '感知', value: data.sensory_processing || 0 },
        { name: '推理', value: data.reasoning || 0 },
        { name: '记忆', value: data.memory_access || 0 },
        { name: '行动', value: data.action_selection || 0 },
        { name: '自我意识', value: data.self_awareness || 0 },
        { name: '情绪', value: data.emotional_state || 0 }
    ];
    
    dom.moduleActivityList.innerHTML = modules.map(m => `
        <div class="module-activity-item">
            <span class="module-name">${m.name}</span>
            <div class="module-bar-container">
                <div class="module-bar" style="width: ${Math.min(m.value * 100, 100)}%"></div>
            </div>
            <span class="module-value">${typeof m.value === 'number' ? (m.value * 100).toFixed(0) : '--'}%</span>
        </div>
    `).join('');
}

function renderSynapticStats(data) {
    if (!dom.synapticStats) return;
    
    dom.synapticStats.innerHTML = `
        <span>自由能: <strong>${typeof data.free_energy === 'number' ? data.free_energy.toFixed(4) : '--'}</strong></span>
        <span>置信度: <strong>${typeof data.confidence === 'number' ? (data.confidence * 100).toFixed(0) : '--'}%</strong></span>
        <span>新奇度: <strong>${typeof data.novelty === 'number' ? (data.novelty * 100).toFixed(0) : '--'}%</strong></span>
        <span>复杂度: <strong>${typeof data.complexity === 'number' ? (data.complexity * 100).toFixed(0) : '--'}%</strong></span>
    `;
}

function shouldLogServerError() {
    const now = Date.now();
    if (now - lastServerErrorTime > 30000) {
        lastServerErrorTime = now;
        return true;
    }
    return false;
}

const dom = {
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    chatMessages: document.getElementById('chatMessages'),
    voiceBtn: document.getElementById('voiceBtn'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    chatSessionName: document.getElementById('chatSessionName'),
    sessionList: document.getElementById('sessionList'),
    newSessionBtn: document.getElementById('newSessionBtn'),
    exportSessionBtn: document.getElementById('exportSessionBtn'),
    fastModeToggle: document.getElementById('fastModeToggle'),
    chatSearchInput: document.getElementById('chatSearchInput'),
    slashCommands: document.getElementById('slashCommands'),
    
    configTabs: document.querySelectorAll('.config-tab'),
    configSections: document.querySelectorAll('.config-section'),
    saveConfigBtn: document.getElementById('saveConfigBtn'),
    resetConfigBtn: document.getElementById('resetConfigBtn'),
    configTemperature: document.getElementById('configTemperature'),
    configTemperatureValue: document.getElementById('configTemperatureValue'),
    
    gatewayStatus: document.getElementById('gatewayStatus'),
    statusIndicator: document.querySelector('.status-indicator'),
    
    refreshBtn: document.getElementById('refreshBtn'),
    fastModeBtn: document.getElementById('fastModeBtn'),
    
    commandPaletteBtn: document.getElementById('commandPaletteBtn'),
    commandPalette: document.getElementById('commandPalette'),
    cpSearch: document.getElementById('cpSearch'),
    cpList: document.getElementById('cpList'),
    
    navItems: document.querySelectorAll('.nav-item'),
    mobileNavItems: document.querySelectorAll('.mobile-nav-item'),
    views: document.querySelectorAll('.view'),
    
    agentList: document.getElementById('agentList'),
    agentDetailName: document.getElementById('agentDetailName'),
    agentDetailContent: document.getElementById('agentDetailContent'),
    newAgentBtn: document.getElementById('newAgentBtn'),
    
    sessionsTable: document.getElementById('sessionsTable'),
    saveAllSessionsBtn: document.getElementById('saveAllSessionsBtn'),
    exportAllSessionsBtn: document.getElementById('exportAllSessionsBtn'),
    
    statActiveAgents: document.getElementById('statActiveAgents'),
    statConnectedChannels: document.getElementById('statConnectedChannels'),
    statActiveSessions: document.getElementById('statActiveSessions'),
    statTokenRate: document.getElementById('statTokenRate'),
    
    cpuStatus: document.getElementById('cpuStatus'),
    cpuUsage: document.getElementById('cpuUsage'),
    memoryStatus: document.getElementById('memoryStatus'),
    memoryUsage: document.getElementById('memoryUsage'),
    gpuStatus: document.getElementById('gpuStatus'),
    gpuUsage: document.getElementById('gpuUsage'),

    activityList: document.getElementById('activityList'),
    
    memoryTierBtns: document.querySelectorAll('.memory-tier-btn'),
    memorySearchInput: document.getElementById('memorySearchInput'),
    memoryHeaderTitle: document.getElementById('memoryHeaderTitle'),
    memoryStats: document.getElementById('memoryStats'),
    memoryList: document.getElementById('memoryList'),
    addMemoryBtn: document.getElementById('addMemoryBtn'),
    
    soulTabs: document.querySelectorAll('.soul-tab'),
    soulSections: document.querySelectorAll('.soul-section'),
    soulName: document.getElementById('soulName'),
    soulRole: document.getElementById('soulRole'),
    soulRigor: document.getElementById('soulRigor'),
    soulCreativity: document.getElementById('soulCreativity'),
    soulGoalsContent: document.getElementById('soulGoalsContent'),
    soulBoundariesContent: document.getElementById('soulBoundariesContent'),
    soulPermissionsContent: document.getElementById('soulPermissionsContent'),
    saveSoulBtn: document.getElementById('saveSoulBtn'),
    exportSoulBtn: document.getElementById('exportSoulBtn'),
    
    tasksStats: document.getElementById('tasksStats'),
    taskPending: document.getElementById('taskPending'),
    taskInProgress: document.getElementById('taskInProgress'),
    taskCompleted: document.getElementById('taskCompleted'),
    submitTaskBtn: document.getElementById('submitTaskBtn'),
    
    evolutionStats: document.getElementById('evolutionStats'),
    proposalList: document.getElementById('proposalList'),
    runEvolutionBtn: document.getElementById('runEvolutionBtn'),
    generateSkillBtn: document.getElementById('generateSkillBtn'),
    
    securityHardBoundary: document.getElementById('securityHardBoundary'),
    securityCircuitBreaker: document.getElementById('securityCircuitBreaker'),
    securityRiskClassifier: document.getElementById('securityRiskClassifier'),
    securityAudit: document.getElementById('securityAudit'),
    securityCompliance: document.getElementById('securityCompliance'),
    
    improvementPerformance: document.getElementById('improvementPerformance'),
    improvementDiagnostic: document.getElementById('improvementDiagnostic'),
    improvementProposals: document.getElementById('improvementProposals'),
    improvementSafety: document.getElementById('improvementSafety'),
    runDiagnosticBtn: document.getElementById('runDiagnosticBtn'),
    generateProposalsBtn: document.getElementById('generateProposalsBtn'),
    
    skillsList: document.getElementById('skillsList'),
    loadSkillsBtn: document.getElementById('loadSkillsBtn'),
    
    knowledgeStats: document.getElementById('knowledgeStats'),
    knowledgeGraph: document.getElementById('knowledgeGraph'),
    
    statMemoryTiers: document.getElementById('statMemoryTiers'),
    statEvolution: document.getElementById('statEvolution'),
    statFreeEnergy: document.getElementById('statFreeEnergy'),
    statConfidence: document.getElementById('statConfidence'),
    statSafety: document.getElementById('statSafety'),
    statKnowledge: document.getElementById('statKnowledge'),
    agentInfoName: document.getElementById('agentInfoName'),
    agentInfoStep: document.getElementById('agentInfoStep'),
    agentInfoStatus: document.getElementById('agentInfoStatus'),
    agentInfoDim: document.getElementById('agentInfoDim'),
    runStepBtn: document.getElementById('runStepBtn'),
    
    synapticStats: document.getElementById('synapticStats'),
    moduleActivityList: document.getElementById('moduleActivityList'),
    oscillatorDisplay: document.getElementById('oscillatorDisplay'),
    synapticGraph: document.getElementById('synapticGraph'),
    signalFlowChart: document.getElementById('signalFlowChart'),
    
    modalOverlay: document.getElementById('modalOverlay'),
    modalTitle: document.getElementById('modalTitle'),
    modalInput: document.getElementById('modalInput'),
    modalConfirm: document.getElementById('modalConfirm'),
    modalCancel: document.getElementById('modalCancel'),
    
    attachBtn: document.getElementById('attachBtn'),
    fileInput: document.getElementById('fileInput'),
    configPluginList: document.getElementById('configPluginList')
};

let modalPromptCallback = null;

function showModalPrompt(title, placeholder, defaultValue = '') {
    return new Promise((resolve) => {
        modalPromptCallback = resolve;
        dom.modalTitle.textContent = title;
        dom.modalInput.placeholder = placeholder;
        dom.modalInput.value = defaultValue;
        dom.modalOverlay.style.display = 'flex';
        setTimeout(() => dom.modalInput.focus(), 50);
    });
}

function closeModal(value = null) {
    dom.modalOverlay.style.display = 'none';
    dom.modalInput.value = '';
    if (modalPromptCallback) {
        modalPromptCallback(value);
        modalPromptCallback = null;
    }
}

dom.modalConfirm.addEventListener('click', () => {
    closeModal(dom.modalInput.value.trim() || null);
});

dom.modalInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        closeModal(dom.modalInput.value.trim() || null);
    } else if (e.key === 'Escape') {
        closeModal(null);
    }
});

dom.modalOverlay.addEventListener('click', (e) => {
    if (e.target === dom.modalOverlay) {
        closeModal(null);
    }
});

async function handleFileUpload(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    addMessage('system', `正在上传 ${files.length} 个文件...`);
    
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${API_BASE}/api/file-ingestion/upload`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                addMessage('assistant', `文件 "${data.filename}" 上传成功，ID: ${data.record_id}`);
                addMessage('assistant', `📚 正在学习文件内容... (已提取 ${data.chunks_count} 个知识块)`);
                startFileLearning(data.record_id, file.name);
            } else {
                addMessage('system', `文件 "${file.name}" 上传失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            addMessage('system', `文件 "${file.name}" 上传失败: ${error.message}`);
        }
    }
    
    e.target.value = '';
}

async function startFileLearning(recordId, filename) {
    try {
        const response = await fetch(`${API_BASE}/api/file-ingestion/record/${recordId}`);
        const data = await response.json();
        
        if (data.metadata && data.metadata.preprocessed) {
            addMessage('assistant', `✅ 文件学习完成！"${filename}" 的知识已整合到知识库中`);
            refreshOverview();
            loadMemory();
        }
    } catch (error) {
        console.error('File learning check failed:', error);
    }
}

let currentSessionId = null;
let sessions = [];
let currentMemoryTier = 'L1';

function initEventListeners() {
    dom.sendBtn.addEventListener('click', sendMessage);
    dom.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    dom.voiceBtn.addEventListener('click', toggleVoiceInput);
    dom.attachBtn.addEventListener('click', () => dom.fileInput.click());
    dom.fileInput.addEventListener('change', handleFileUpload);
    dom.clearChatBtn.addEventListener('click', clearChat);
    dom.newSessionBtn.addEventListener('click', createNewSession);
    dom.exportSessionBtn.addEventListener('click', exportSession);
    dom.fastModeToggle.addEventListener('click', toggleFastMode);
    dom.chatSearchInput.addEventListener('input', searchSessions);
    
    dom.configTabs.forEach(tab => {
        tab.addEventListener('click', () => switchConfigTab(tab.dataset.tab));
    });
    
    dom.saveConfigBtn.addEventListener('click', saveConfig);
    dom.resetConfigBtn.addEventListener('click', resetConfig);
    
    if (dom.configTemperature) {
        dom.configTemperature.addEventListener('input', (e) => {
            dom.configTemperatureValue.textContent = e.target.value;
        });
    }
    
    dom.refreshBtn.addEventListener('click', refreshAllData);
    dom.runStepBtn.addEventListener('click', runAgentStep);
    
    dom.commandPaletteBtn.addEventListener('click', openCommandPalette);
    
    dom.navItems.forEach(item => {
        item.addEventListener('click', () => switchView(item.dataset.view));
    });
    
    dom.mobileNavItems.forEach(item => {
        item.addEventListener('click', () => switchView(item.dataset.view));
    });
    
    dom.newAgentBtn.addEventListener('click', createNewAgent);
    
    dom.saveAllSessionsBtn.addEventListener('click', saveAllSessions);
    dom.exportAllSessionsBtn.addEventListener('click', exportAllSessions);
    
    dom.slashCommands.addEventListener('click', (e) => {
        const cmd = e.target.closest('.slash-command');
        if (cmd) {
            dom.chatInput.value = cmd.textContent + ' ';
            dom.chatInput.focus();
        }
    });
    
    dom.chatInput.addEventListener('keydown', (e) => {
        if (e.key === '/' && !dom.chatInput.value) {
            e.preventDefault();
            showSlashCommands();
        }
    });
    
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            openCommandPalette();
        }
        if (e.key === 'Escape') {
            closeCommandPalette();
        }
    });
    
    dom.memoryTierBtns.forEach(btn => {
        btn.addEventListener('click', () => switchMemoryTier(btn.dataset.tier));
    });
    
    dom.memorySearchInput.addEventListener('input', searchMemories);
    dom.addMemoryBtn.addEventListener('click', addMemory);
    
    dom.soulTabs.forEach(tab => {
        tab.addEventListener('click', () => switchSoulTab(tab.dataset.tab));
    });
    
    dom.saveSoulBtn.addEventListener('click', saveSoul);
    dom.exportSoulBtn.addEventListener('click', exportSoul);
    
    dom.submitTaskBtn.addEventListener('click', submitTask);
    
    dom.runEvolutionBtn.addEventListener('click', runEvolution);
    dom.generateSkillBtn.addEventListener('click', generateSkill);
    
    dom.runDiagnosticBtn.addEventListener('click', runDiagnostic);
    dom.generateProposalsBtn.addEventListener('click', generateProposals);
    
    dom.loadSkillsBtn.addEventListener('click', loadSkills);
    
    window.addEventListener('keydown', handleKeyDown);
}

function switchView(viewName) {
    dom.views.forEach(view => {
        view.style.display = view.id === `view-${viewName}` ? 'block' : 'none';
    });
    
    dom.navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });
    
    dom.mobileNavItems.forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });
    
    if (viewName === 'overview') {
        refreshOverview();
    } else if (viewName === 'chat') {
        loadSessions();
    } else if (viewName === 'memory') {
        loadMemory();
    } else if (viewName === 'soul') {
        loadSoul();
    } else if (viewName === 'tasks') {
        loadTasks();
    } else if (viewName === 'evolution') {
        loadEvolution();
    } else if (viewName === 'security') {
        loadSecurity();
    } else if (viewName === 'selfimprovement') {
        loadSelfImprovement();
    } else if (viewName === 'skills') {
        loadSkills();
    } else if (viewName === 'knowledge') {
        loadKnowledge();
    } else if (viewName === 'synaptic') {
        loadSynapticData();
    } else if (viewName === 'sessions') {
        loadSessions();
    } else if (viewName === 'agents') {
        loadAgents();
    }
}

function switchConfigTab(tabName) {
    dom.configTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    dom.configSections.forEach(section => {
        section.style.display = section.id === `config-${tabName}` ? 'block' : 'none';
    });
    
    if (tabName === 'plugins') {
        loadPlugins();
    }
}

async function loadPlugins() {
    try {
        const response = await fetch(`${API_BASE}/api/plugins/available`);
        const data = await response.json();
        renderPlugins(data.plugins || []);
    } catch (error) {
        console.error('Failed to load plugins:', error);
        dom.configPluginList.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 40px;">加载失败</div>';
    }
}

function renderPlugins(plugins) {
    if (plugins.length === 0) {
        dom.configPluginList.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 40px;">暂无插件</div>';
        return;
    }
    
    dom.configPluginList.innerHTML = plugins.map(plugin => `
        <div class="plugin-item">
            <div>
                <h4>${plugin.name || '未命名插件'}</h4>
                <p>${plugin.description || '无描述'}</p>
                <span style="font-size: 11px; color: var(--text-muted);">版本: ${plugin.version || '1.0'} | 类型: ${plugin.type || 'processor'}</span>
            </div>
            <div class="plugin-toggle ${plugin.status === 'active' ? 'on' : ''}" data-plugin="${plugin.name}" onclick="togglePlugin('${plugin.name}')"></div>
        </div>
    `).join('');
}

async function togglePlugin(pluginName) {
    const toggle = document.querySelector(`.plugin-toggle[data-plugin="${pluginName}"]`);
    const isActive = toggle.classList.contains('on');
    
    try {
        const endpoint = isActive ? `/api/plugins/${pluginName}/deactivate` : `/api/plugins/${pluginName}/activate`;
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            toggle.classList.toggle('on');
            loadPlugins();
        }
    } catch (error) {
        console.error('Plugin toggle failed:', error);
    }
}

function switchMemoryTier(tier) {
    currentMemoryTier = tier;
    dom.memoryTierBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tier === tier);
    });
    const tierNames = { L1: '情境记忆', L2: '工作记忆', L3: '中间记忆', L4: '学习记忆', L5: '永久记忆' };
    dom.memoryHeaderTitle.textContent = `${tier} ${tierNames[tier]}`;
    loadMemory();
}

function switchSoulTab(tabName) {
    dom.soulTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    dom.soulSections.forEach(section => {
        section.style.display = section.id === `soul-${tabName}` ? 'block' : 'none';
    });
}

function openCommandPalette() {
    dom.commandPalette.style.display = 'block';
    dom.cpSearch.focus();
}

function closeCommandPalette() {
    dom.commandPalette.style.display = 'none';
    dom.cpSearch.value = '';
}

function showSlashCommands() {
    const commands = [
        { cmd: '/new', desc: '新建会话' },
        { cmd: '/clear', desc: '清空聊天' },
        { cmd: '/save', desc: '保存会话' },
        { cmd: '/export', desc: '导出会话' },
        { cmd: '/config', desc: '打开配置' },
        { cmd: '/agent', desc: '切换Agent' },
        { cmd: '/memory', desc: '查看记忆' },
        { cmd: '/soul', desc: '编辑SOUL' },
        { cmd: '/tasks', desc: '任务看板' }
    ];
    
    dom.slashCommands.innerHTML = commands.map(cmd => 
        `<span class="slash-command">${cmd.cmd} <small>${cmd.desc}</small></span>`
    ).join('');
    
    setTimeout(() => {
        dom.slashCommands.innerHTML = '';
    }, 5000);
}

async function checkAgentStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/agent/status`);
        const data = await response.json();
        if (!serverAvailable) {
            serverAvailable = true;
            console.log('Server connection restored');
        }
        isAgentRunning = data.status === 'running';
        updateGatewayStatus(isAgentRunning);
    } catch (error) {
        if (serverAvailable) {
            serverAvailable = false;
            console.warn('Server unavailable:', error.message || error);
        }
        updateGatewayStatus(false);
    }
}

function updateGatewayStatus(online) {
    if (online) {
        dom.statusIndicator.classList.remove('offline');
        dom.statusIndicator.classList.add('online');
        dom.gatewayStatus.querySelector('span:last-child').textContent = 'Gateway 在线';
    } else {
        dom.statusIndicator.classList.remove('online');
        dom.statusIndicator.classList.add('offline');
        dom.gatewayStatus.querySelector('span:last-child').textContent = 'Gateway 离线';
    }
}

async function sendMessage() {
    const content = dom.chatInput.value.trim();
    if (!content) return;
    
    addMessage('user', content);
    dom.chatInput.value = '';
    
    if (content.startsWith('/')) {
        handleSlashCommand(content);
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/chat/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });
        const data = await response.json();
        addMessage('assistant', data.response);
    } catch (error) {
        console.error('Failed to send message:', error);
        addMessage('assistant', '发送消息失败，请检查网络连接');
    }
}

function handleSlashCommand(cmd) {
    switch (cmd.split(' ')[0]) {
        case '/new':
            createNewSession();
            break;
        case '/clear':
            clearChat();
            break;
        case '/save':
            saveCurrentSession();
            break;
        case '/export':
            exportSession();
            break;
        case '/config':
            switchView('config');
            break;
        case '/agent':
            switchView('agents');
            break;
        case '/memory':
            switchView('memory');
            break;
        case '/soul':
            switchView('soul');
            break;
        case '/tasks':
            switchView('tasks');
            break;
        case '/evolution':
            switchView('evolution');
            break;
        default:
            addMessage('assistant', `未知命令: ${cmd}`);
    }
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    let avatarBg = '';
    let icon = '';
    
    if (role === 'user') {
        avatarBg = 'var(--primary-color)';
        icon = '👤';
    } else if (role === 'assistant') {
        avatarBg = 'var(--secondary-color)';
        icon = '🤖';
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar" style="background: ${avatarBg}">${icon}</div>
        <div class="message-content">
            <div class="message-bubble">${content}</div>
        </div>
    `;
    
    dom.chatMessages.appendChild(messageDiv);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

function clearChat() {
    dom.chatMessages.innerHTML = `
        <div class="message system-message">
            <p>欢迎使用 OpenClaw！开始与您的智能体对话吧。</p>
        </div>
    `;
}

function toggleVoiceInput() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        addMessage('assistant', '您的浏览器不支持语音识别功能');
        return;
    }
    
    if (isVoiceRecording) {
        stopVoiceRecording();
    } else {
        startVoiceRecording();
    }
}

function startVoiceRecording() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-CN';
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onstart = () => {
        isVoiceRecording = true;
        dom.voiceBtn.style.backgroundColor = '#f85149';
        dom.voiceBtn.style.color = 'white';
        addMessage('assistant', '正在听...请说话');
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        dom.chatInput.value = transcript;
        addMessage('user', transcript);
        stopVoiceRecording();
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        addMessage('assistant', '语音识别出错，请重试');
        stopVoiceRecording();
    };
    
    recognition.onend = () => {
        stopVoiceRecording();
    };
    
    recognition.start();
}

function stopVoiceRecording() {
    if (recognition) {
        recognition.stop();
        recognition = null;
    }
    isVoiceRecording = false;
    dom.voiceBtn.style.backgroundColor = '';
    dom.voiceBtn.style.color = '';
}

function createNewSession() {
    clearChat();
    dom.chatSessionName.textContent = '新会话';
    currentSessionId = null;
}

async function saveCurrentSession() {
    const messages = Array.from(dom.chatMessages.children).map(msg => {
        const role = msg.classList.contains('user') ? 'user' : 'assistant';
        const text = msg.querySelector('.message-bubble')?.textContent || '';
        return { role, content: text };
    }).filter(m => m.content.trim());
    
    try {
        const response = await fetch(`${API_BASE}/api/sessions/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                name: dom.chatSessionName.textContent,
                messages
            })
        });
        const data = await response.json();
        if (data.status === 'success') {
            addMessage('assistant', '会话已保存');
            currentSessionId = data.session_id;
            loadSessions();
        }
    } catch (error) {
        console.error('Failed to save session:', error);
        addMessage('assistant', '保存会话失败');
    }
}

async function exportSession() {
    try {
        const response = await fetch(`${API_BASE}/api/sessions/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId })
        });
        const data = await response.json();
        if (data.status === 'success') {
            const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `session_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
            addMessage('assistant', '会话已导出');
        }
    } catch (error) {
        console.error('Failed to export session:', error);
        addMessage('assistant', '导出会话失败');
    }
}

function toggleFastMode() {
    dom.fastModeToggle.classList.toggle('active');
    dom.fastModeBtn.disabled = !dom.fastModeBtn.disabled;
}

function searchSessions(query) {
    const searchTerm = query || dom.chatSearchInput.value.toLowerCase();
    dom.sessionList.querySelectorAll('.session-item').forEach(item => {
        const name = item.querySelector('.session-name').textContent.toLowerCase();
        item.style.display = name.includes(searchTerm) ? 'flex' : 'none';
    });
}

async function loadSessions() {
    try {
        const response = await fetch(`${API_BASE}/api/sessions/list`);
        const data = await response.json();
        sessions = data.sessions || [];
        renderSessions();
    } catch (error) {
        console.error('Failed to load sessions:', error);
    }
}

function renderSessions() {
    if (sessions.length === 0) {
        dom.sessionList.innerHTML = `
            <div class="session-item active">
                <span class="session-icon">💬</span>
                <span class="session-name">新会话</span>
                <span class="session-time">刚刚</span>
            </div>
        `;
        return;
    }
    
    dom.sessionList.innerHTML = sessions.map((session, index) => `
        <div class="session-item ${index === 0 ? 'active' : ''}" onclick="loadSession('${session.id}')">
            <span class="session-icon">💬</span>
            <span class="session-name">${session.name}</span>
            <span class="session-time">${formatTime(session.last_active)}</span>
        </div>
    `).join('');
}

function loadSession(sessionId) {
    const session = sessions.find(s => s.id === sessionId);
    if (!session) return;
    
    currentSessionId = sessionId;
    dom.chatSessionName.textContent = session.name;
    clearChat();
    
    (session.messages || []).forEach(msg => {
        addMessage(msg.role, msg.content);
    });
    
    switchView('chat');
}

function formatTime(timestamp) {
    if (!timestamp) return '刚刚';
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
    return date.toLocaleDateString();
}

async function saveAllSessions() {
    try {
        const response = await fetch(`${API_BASE}/api/sessions/save_all`, { method: 'POST' });
        const data = await response.json();
        if (data.status === 'success') {
            addMessage('assistant', '所有会话已保存');
        }
    } catch (error) {
        console.error('Failed to save all sessions:', error);
    }
}

async function exportAllSessions() {
    try {
        const response = await fetch(`${API_BASE}/api/sessions/export_all`, { method: 'POST' });
        const data = await response.json();
        if (data.status === 'success') {
            const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `all_sessions_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    } catch (error) {
        console.error('Failed to export all sessions:', error);
    }
}

async function loadAgents() {
    try {
        const response = await fetch(`${API_BASE}/api/agents/list`);
        const data = await response.json();
        const agents = data.agents || [];
        renderAgents(agents);
    } catch (error) {
        console.error('Failed to load agents:', error);
        dom.agentList.innerHTML = '<div class="empty-state">无法加载Agent列表</div>';
    }
}

function renderAgents(agents) {
    if (agents.length === 0) {
        dom.agentList.innerHTML = '<div class="empty-state">暂无Agent</div>';
        return;
    }
    
    dom.agentList.innerHTML = agents.map((agent, index) => `
        <div class="agent-item ${index === 0 ? 'active' : ''}" onclick="loadAgentDetail('${agent.name}')">
            <h4>${agent.name}</h4>
            <p>${agent.description || '暂无描述'}</p>
        </div>
    `).join('');
}

function loadAgentDetail(agentName) {
    dom.agentDetailName.textContent = agentName;
    dom.agentDetailContent.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 16px;">
            <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md);">
                <h4>基本信息</h4>
                <p style="color: var(--text-secondary); margin-top: 8px;">名称: ${agentName}</p>
                <p style="color: var(--text-secondary);">状态: <span style="color: var(--success-color);">运行中</span></p>
            </div>
            <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md);">
                <h4>配置</h4>
                <p style="color: var(--text-secondary);">温度: 0.7</p>
            </div>
            <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md);">
                <h4>工具</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;">
                    <span style="padding: 4px 12px; background: var(--bg-secondary); border-radius: 12px; font-size: 12px;">文件操作</span>
                    <span style="padding: 4px 12px; background: var(--bg-secondary); border-radius: 12px; font-size: 12px;">Shell命令</span>
                    <span style="padding: 4px 12px; background: var(--bg-secondary); border-radius: 12px; font-size: 12px;">Web浏览</span>
                </div>
            </div>
        </div>
    `;
}

function createNewAgent() {
    addMessage('assistant', '新建Agent功能正在开发中...');
}

async function saveConfig() {
    const config = {
        port: parseInt(document.getElementById('configPort')?.value) || 8000,
        max_sessions: parseInt(document.getElementById('configMaxSessions')?.value) || 10,
        temperature: parseFloat(document.getElementById('configTemperature')?.value) || 0.7,
        agent_name: document.getElementById('configAgentName')?.value || 'OpenClaw Agent',
        system_prompt: document.getElementById('configSystemPrompt')?.value || ''
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/config/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await response.json();
        if (data.status === 'success') {
            addMessage('assistant', '配置已保存');
        }
    } catch (error) {
        console.error('Failed to save config:', error);
        addMessage('assistant', '保存配置失败');
    }
}

function resetConfig() {
    document.getElementById('configPort').value = 8000;
    document.getElementById('configMaxSessions').value = 10;
    document.getElementById('configTemperature').value = 0.7;
    document.getElementById('configTemperatureValue').textContent = '0.7';
    document.getElementById('configAgentName').value = 'OpenClaw Agent';
    document.getElementById('configSystemPrompt').value = '你是一个强大的AI助手...';
}

async function refreshOverview() {
    try {
        const response = await fetch(`${API_BASE}/api/system/overview`);
        const data = await response.json();
        
        if (dom.statActiveAgents) dom.statActiveAgents.textContent = data.active_agents || 0;
        if (dom.statConnectedChannels) dom.statConnectedChannels.textContent = data.connected_channels || 0;
        if (dom.statActiveSessions) dom.statActiveSessions.textContent = data.active_sessions || 0;
        if (dom.statTokenRate) dom.statTokenRate.textContent = data.token_rate || 0;

        if (dom.statMemoryTiers) dom.statMemoryTiers.textContent = data.memory_tiers || 5;
        if (dom.statEvolution) dom.statEvolution.textContent = data.evolution_count || 0;
        if (dom.statFreeEnergy) dom.statFreeEnergy.textContent = (data.free_energy || 0).toFixed(2);
        if (dom.statConfidence) dom.statConfidence.textContent = `${(data.confidence || 0) * 100}%`;
        if (dom.statSafety) dom.statSafety.textContent = data.safety_status || '安全';
        if (dom.statKnowledge) dom.statKnowledge.textContent = data.knowledge_nodes || 0;
        
        updateSystemStatus(data.system_status);
        renderActivity(data.recent_activity || []);
        
        updateAgentInfo(data.agent_info);
    } catch (error) {
        console.error('Failed to refresh overview:', error);
    }
}

function updateAgentInfo(agentInfo) {
    if (!agentInfo) return;
    dom.agentInfoName.textContent = agentInfo.name || 'AGI_Agent';
    dom.agentInfoStep.textContent = agentInfo.step || 0;
    dom.agentInfoStatus.textContent = agentInfo.status || '运行中';
    dom.agentInfoDim.textContent = agentInfo.input_dim || 16;
}

function updateSystemStatus(status) {
    if (!status) return;
    
    updateStatusItem('cpu', status.cpu_usage, status.cpu_usage > 80 ? 'warning' : status.cpu_usage > 90 ? 'error' : 'online');
    updateStatusItem('memory', status.memory_usage, status.memory_usage > 80 ? 'warning' : status.memory_usage > 90 ? 'error' : 'online');
    updateStatusItem('gpu', status.gpu_usage || '--', 'online');
}

function updateStatusItem(type, value, statusClass) {
    const dot = document.getElementById(`${type}Status`);
    const usage = document.getElementById(`${type}Usage`);
    
    if (dot) {
        dot.className = 'status-dot ' + statusClass;
    }
    if (usage) {
        usage.textContent = typeof value === 'number' ? value + '%' : value;
    }
}

function renderActivity(activity) {
    dom.activityList.innerHTML = activity.map(item => `
        <div class="activity-item">
            <span class="activity-title">${item.action}</span>
            <span class="activity-time">${formatTime(item.timestamp)}</span>
        </div>
    `).join('') || '<div style="color: var(--text-muted); font-size: 13px; padding: 12px;">暂无活动记录</div>';
}

function refreshAllData() {
    checkAgentStatus();
    refreshOverview();
    loadSessions();
    loadAgents();
}

function handleKeyDown(e) {
    if (e.key === 'Escape') {
        closeCommandPalette();
    }
    
    if (dom.commandPalette.style.display === 'block') {
        const items = dom.cpList.querySelectorAll('.cp-item');
        let currentIndex = -1;
        
        items.forEach((item, index) => {
            if (item.classList.contains('selected')) {
                currentIndex = index;
            }
        });
        
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (currentIndex < items.length - 1) {
                if (currentIndex >= 0) items[currentIndex].classList.remove('selected');
                items[currentIndex + 1].classList.add('selected');
                items[currentIndex + 1].scrollIntoView({ block: 'nearest' });
            }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (currentIndex > 0) {
                items[currentIndex].classList.remove('selected');
                items[currentIndex - 1].classList.add('selected');
                items[currentIndex - 1].scrollIntoView({ block: 'nearest' });
            }
        } else if (e.key === 'Enter') {
            e.preventDefault();
            const selected = dom.cpList.querySelector('.cp-item.selected');
            if (selected) {
                const label = selected.querySelector('.cp-label').textContent;
                handleCommandAction(label);
                closeCommandPalette();
            }
        }
    }
}

function handleCommandAction(label) {
    switch (label) {
        case '新建会话':
            createNewSession();
            switchView('chat');
            break;
        case '搜索会话':
            dom.chatSearchInput.focus();
            switchView('chat');
            break;
        case '切换极速模式':
            toggleFastMode();
            break;
        case '打开配置':
            switchView('config');
            break;
        case '查看概览':
            switchView('overview');
            break;
        case '查看记忆':
            switchView('memory');
            break;
        case '编辑 SOUL':
            switchView('soul');
            break;
        case '任务看板':
            switchView('tasks');
            break;
        case '进化监控':
            switchView('evolution');
            break;
        case '安全中心':
            switchView('security');
            break;
        case '自我改进':
            switchView('selfimprovement');
            break;
    }
}

async function runAgentStep() {
    try {
        const response = await fetch(`${API_BASE}/api/agent/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ observation: [] })
        });
        const data = await response.json();
        console.log('Agent step result:', data);
        refreshOverview();
    } catch (error) {
        console.error('Failed to run agent step:', error);
    }
}

async function loadMemory() {
    try {
        const [statsResponse, listResponse] = await Promise.all([
            fetch(`${API_BASE}/api/memory/stats`),
            fetch(`${API_BASE}/api/memory/list?tier=${currentMemoryTier}`)
        ]);
        
        const stats = await statsResponse.json();
        const listData = await listResponse.json();
        
        renderMemoryStats(stats);
        renderMemoryList(listData.memories || []);
    } catch (error) {
        console.error('Failed to load memory:', error);
        dom.memoryStats.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
        dom.memoryList.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
    }
}

function renderMemoryStats(stats) {
    dom.memoryStats.innerHTML = `
        <div class="memory-stat-item"><span>总记忆数:</span><span>${stats.total_entries || 0}</span></div>
        <div class="memory-stat-item"><span>L1:</span><span>${stats.L1 || 0}</span></div>
        <div class="memory-stat-item"><span>L2:</span><span>${stats.L2 || 0}</span></div>
        <div class="memory-stat-item"><span>L3:</span><span>${stats.L3 || 0}</span></div>
        <div class="memory-stat-item"><span>L4:</span><span>${stats.L4 || 0}</span></div>
        <div class="memory-stat-item"><span>L5:</span><span>${stats.L5 || 0}</span></div>
    `;
}

function renderMemoryList(memories) {
    if (memories.length === 0) {
        dom.memoryList.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 40px;">暂无记忆</div>';
        return;
    }
    
    dom.memoryList.innerHTML = memories.map(m => `
        <div class="memory-item">
            <div class="memory-item-header">
                <span class="memory-item-id">ID: ${m.memory_id || 'N/A'}</span>
                <span class="memory-item-time">${formatTime(m.timestamp)}</span>
            </div>
            <div class="memory-item-content">${m.content || m.data || '无内容'}</div>
        </div>
    `).join('');
}

function searchMemories() {
    const query = dom.memorySearchInput.value.trim();
    if (!query) {
        loadMemory();
        return;
    }
    
    fetch(`${API_BASE}/api/memory/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    })
    .then(response => response.json())
    .then(data => {
        renderMemoryList(data.results || []);
    })
    .catch(error => console.error('Search failed:', error));
}

async function addMemory() {
    const content = await showModalPrompt('添加记忆', '输入记忆内容:');
    if (!content) return;
    
    fetch(`${API_BASE}/api/memory/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, tier: currentMemoryTier })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            loadMemory();
        }
    })
    .catch(error => console.error('Add memory failed:', error));
}

async function loadSoul() {
    try {
        const response = await fetch(`${API_BASE}/api/soul/info`);
        const data = await response.json();
        
        if (data.identity) {
            dom.soulName.value = data.identity.name || 'AGI_Agent';
            dom.soulRole.value = data.identity.role_boundary || 'Autonomous Intelligence';
            
            if (data.identity.personality) {
                dom.soulRigor.value = data.identity.personality.rigor || 50;
                dom.soulCreativity.value = data.identity.personality.creativity || 50;
            }
        }
        
        renderSoulGoals(data.goals || {});
        renderSoulBoundaries(data.boundaries || {});
        renderSoulPermissions(data.permissions || {});
    } catch (error) {
        console.error('Failed to load SOUL:', error);
    }
}

function renderSoulGoals(goals) {
    dom.soulGoalsContent.innerHTML = `
        <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md);">
            <h4>使命</h4>
            <p style="color: var(--text-secondary); margin-top: 8px;">${goals.mission || '未设置'}</p>
        </div>
        <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md); margin-top: 12px;">
            <h4>目标节点</h4>
            <ul style="color: var(--text-secondary); margin-top: 8px; list-style: none; padding: 0;">
                ${(goals.nodes || []).map((node, i) => `
                    <li style="padding: 8px 0; border-bottom: 1px solid var(--border-color);">${i + 1}. ${node.name || node}</li>
                `).join('') || '<li>暂无目标节点</li>'}
            </ul>
        </div>
    `;
}

function renderSoulBoundaries(boundaries) {
    dom.soulBoundariesContent.innerHTML = `
        <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md);">
            <h4>禁止行为</h4>
            <ul style="color: var(--text-secondary); margin-top: 8px;">
                ${(boundaries.forbidden_actions || []).map((action, i) => `
                    <li>${i + 1}. ${action}</li>
                `).join('') || '<li>未设置</li>'}
            </ul>
        </div>
        <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md); margin-top: 12px;">
            <h4>伦理原则</h4>
            <ul style="color: var(--text-secondary); margin-top: 8px;">
                ${(boundaries.ethical_principles || []).map((principle, i) => `
                    <li>${i + 1}. ${principle}</li>
                `).join('') || '<li>未设置</li>'}
            </ul>
        </div>
        <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md); margin-top: 12px;">
            <h4>安全红线</h4>
            <ul style="color: var(--error-color); margin-top: 8px;">
                ${(boundaries.safety_redlines || []).map((redline, i) => `
                    <li>${i + 1}. ${redline}</li>
                `).join('') || '<li>未设置</li>'}
            </ul>
        </div>
    `;
}

function renderSoulPermissions(permissions) {
    dom.soulPermissionsContent.innerHTML = `
        <div style="padding: 16px; background: var(--bg-tertiary); border-radius: var(--radius-md);">
            <h4>权限条目</h4>
            <ul style="color: var(--text-secondary); margin-top: 8px;">
                ${(permissions.entries || []).map((entry, i) => `
                    <li style="padding: 8px 0; border-bottom: 1px solid var(--border-color);">${i + 1}. ${entry.name || entry}</li>
                `).join('') || '<li>暂无权限条目</li>'}
            </ul>
        </div>
    `;
}

function saveSoul() {
    const data = {
        identity: {
            name: dom.soulName.value,
            role_boundary: dom.soulRole.value,
            personality: {
                rigor: parseInt(dom.soulRigor.value),
                creativity: parseInt(dom.soulCreativity.value)
            }
        }
    };
    
    fetch(`${API_BASE}/api/soul/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addMessage('assistant', `SOUL已保存，版本: ${data.version}`);
        }
    })
    .catch(error => console.error('Save SOUL failed:', error));
}

function exportSoul() {
    fetch(`${API_BASE}/api/soul/export`)
    .then(response => response.json())
    .then(data => {
        const blob = new Blob([data.markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'SOUL.md';
        a.click();
        URL.revokeObjectURL(url);
        addMessage('assistant', 'SOUL已导出');
    })
    .catch(error => console.error('Export SOUL failed:', error));
}

async function loadTasks() {
    try {
        const [statsResponse, tasksResponse] = await Promise.all([
            fetch(`${API_BASE}/api/tasks/stats`),
            fetch(`${API_BASE}/api/tasks/list`)
        ]);
        
        const stats = await statsResponse.json();
        const tasksData = await tasksResponse.json();
        
        renderTasksStats(stats);
        renderTasksBoard(tasksData.tasks || []);
    } catch (error) {
        console.error('Failed to load tasks:', error);
    }
}

function renderTasksStats(stats) {
    const boardStats = stats.board || {};
    dom.tasksStats.innerHTML = `
        <span>待处理: <strong>${boardStats.pending || 0}</strong></span>
        <span>进行中: <strong>${boardStats.in_progress || 0}</strong></span>
        <span>已完成: <strong>${boardStats.completed || 0}</strong></span>
    `;
}

function renderTasksBoard(tasks) {
    const pending = tasks.filter(t => t.status === 'pending');
    const inProgress = tasks.filter(t => t.status === 'in_progress');
    const completed = tasks.filter(t => t.status === 'completed');
    
    dom.taskPending.innerHTML = renderTaskColumn(pending);
    dom.taskInProgress.innerHTML = renderTaskColumn(inProgress);
    dom.taskCompleted.innerHTML = renderTaskColumn(completed);
}

function renderTaskColumn(tasks) {
    if (tasks.length === 0) {
        return '<div style="color: var(--text-muted); text-align: center; padding: 20px;">无任务</div>';
    }
    
    return tasks.map(task => `
        <div class="task-card">
            <h4>${task.name}</h4>
            <p>${task.description || '无描述'}</p>
            <span class="task-priority ${task.priority?.toLowerCase() || 'medium'}">${task.priority || 'MEDIUM'}</span>
        </div>
    `).join('');
}

async function submitTask() {
    const name = await showModalPrompt('提交任务', '输入任务名称:');
    if (!name) return;
    const description = await showModalPrompt('提交任务', '输入任务描述:');
    const priority = await showModalPrompt('提交任务', '输入优先级 (low/medium/high/critical):', 'medium');
    
    fetch(`${API_BASE}/api/tasks/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description, priority })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            loadTasks();
        }
    })
    .catch(error => console.error('Submit task failed:', error));
}

async function loadEvolution() {
    try {
        const [statsResponse, proposalsResponse] = await Promise.all([
            fetch(`${API_BASE}/api/evolution/stats`),
            fetch(`${API_BASE}/api/evolution/proposals`)
        ]);
        
        const stats = await statsResponse.json();
        const proposalsData = await proposalsResponse.json();
        
        renderEvolutionStats(stats);
        renderProposals(proposalsData.proposals || []);
    } catch (error) {
        console.error('Failed to load evolution:', error);
    }
}

function renderEvolutionStats(stats) {
    const dualLoop = stats.dual_loop || {};
    dom.evolutionStats.innerHTML = `
        <div class="evolution-stat-card">
            <div class="stat-label">进化次数</div>
            <div class="stat-value">${dualLoop.evolution_count || 0}</div>
        </div>
        <div class="evolution-stat-card">
            <div class="stat-label">外层循环</div>
            <div class="stat-value">${dualLoop.outer_loop_count || 0}</div>
        </div>
        <div class="evolution-stat-card">
            <div class="stat-label">内层循环</div>
            <div class="stat-value">${dualLoop.inner_loop_count || 0}</div>
        </div>
        <div class="evolution-stat-card">
            <div class="stat-label">提案数</div>
            <div class="stat-value">${dualLoop.proposal_count || 0}</div>
        </div>
    `;
}

function renderProposals(proposals) {
    if (proposals.length === 0) {
        dom.proposalList.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 40px;">暂无进化提案</div>';
        return;
    }
    
    dom.proposalList.innerHTML = proposals.map(p => `
        <div class="proposal-card">
            <h4>${p.name || '未命名提案'}</h4>
            <p>${p.description || '无描述'}</p>
            <div class="proposal-meta">
                <span>状态: ${p.status || 'pending'}</span>
                <span>评分: ${p.score || 0}</span>
            </div>
        </div>
    `).join('');
}

function runEvolution() {
    fetch(`${API_BASE}/api/evolution/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ outer: true, inner: false })
    })
    .then(response => response.json())
    .then(data => {
        addMessage('assistant', '进化已执行');
        loadEvolution();
    })
    .catch(error => console.error('Run evolution failed:', error));
}

async function generateSkill() {
    const requirement = await showModalPrompt('生成技能', '输入技能需求:');
    if (!requirement) return;
    
    fetch(`${API_BASE}/api/evolution/generate_skill`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ requirement })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addMessage('assistant', `技能 "${data.name}" 已生成，ID: ${data.skill_id}`);
            loadSkills();
        }
    })
    .catch(error => console.error('Generate skill failed:', error));
}

async function loadSecurity() {
    try {
        const response = await fetch(`${API_BASE}/api/security/overview`);
        const data = await response.json();
        
        renderSecurityHardBoundary(data.hard_boundary || {});
        renderSecurityCircuitBreaker(data.circuit_breaker || {});
        renderSecurityRiskClassifier(data.risk_classifier || {});
        renderSecurityAudit(data.audit_log || []);
        renderSecurityCompliance(data.compliance || {});
    } catch (error) {
        console.error('Failed to load security:', error);
        renderSecurityFallback();
    }
}

function renderSecurityFallback() {
    dom.securityHardBoundary.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
    dom.securityCircuitBreaker.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
    dom.securityRiskClassifier.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
    dom.securityAudit.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
    dom.securityCompliance.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
}

function renderSecurityHardBoundary(boundary) {
    dom.securityHardBoundary.innerHTML = `
        <div class="status-item">
            <span class="status-dot ${boundary.active ? 'online' : 'offline'}"></span>
            <span class="status-label">状态</span>
            <span class="status-value">${boundary.active ? '激活' : '未激活'}</span>
        </div>
        <div class="status-item">
            <span class="status-label">规则数</span>
            <span class="status-value">${boundary.rule_count || 0}</span>
        </div>
    `;
}

function renderSecurityCircuitBreaker(circuit) {
    dom.securityCircuitBreaker.innerHTML = `
        <div class="status-item">
            <span class="status-dot ${circuit.tripped ? 'error' : 'online'}"></span>
            <span class="status-label">状态</span>
            <span class="status-value">${circuit.tripped ? '已熔断' : '正常'}</span>
        </div>
        <div class="status-item">
            <span class="status-label">故障数</span>
            <span class="status-value">${circuit.failure_count || 0}</span>
        </div>
        <div class="status-item">
            <span class="status-label">阈值</span>
            <span class="status-value">${circuit.threshold || 10}</span>
        </div>
    `;
}

function renderSecurityRiskClassifier(classifier) {
    dom.securityRiskClassifier.innerHTML = `
        <div class="status-item">
            <span class="status-label">高风险</span>
            <span class="status-value">${classifier.high_risk || 0}</span>
        </div>
        <div class="status-item">
            <span class="status-label">中风险</span>
            <span class="status-value">${classifier.medium_risk || 0}</span>
        </div>
        <div class="status-item">
            <span class="status-label">低风险</span>
            <span class="status-value">${classifier.low_risk || 0}</span>
        </div>
    `;
}

function renderSecurityAudit(audit) {
    if (audit.length === 0) {
        dom.securityAudit.innerHTML = '<div style="color: var(--text-muted);">暂无审计记录</div>';
        return;
    }
    
    dom.securityAudit.innerHTML = audit.slice(0, 10).map(entry => `
        <div style="padding: 8px; border-bottom: 1px solid var(--border-color); font-size: 12px;">
            <span style="color: var(--text-primary);">${entry.action || '未知操作'}</span>
            <span style="color: var(--text-muted); margin-left: 12px;">${formatTime(entry.timestamp)}</span>
        </div>
    `).join('');
}

function renderSecurityCompliance(compliance) {
    dom.securityCompliance.innerHTML = `
        <div style="display: flex; flex-wrap: wrap; gap: 12px;">
            ${compliance.checks?.map(check => `
                <div style="padding: 12px 16px; background: var(--bg-tertiary); border-radius: var(--radius-md);">
                    <span>${check.name || '检查项'}:</span>
                    <span style="margin-left: 8px; color: ${check.passed ? 'var(--success-color)' : 'var(--error-color)'}; font-weight: bold;">${check.passed ? '通过' : '失败'}</span>
                </div>
            `).join('') || '<div style="color: var(--text-muted);">暂无合规检查</div>'}
        </div>
    `;
}

async function loadSelfImprovement() {
    try {
        const response = await fetch(`${API_BASE}/api/self_improvement/overview`);
        const data = await response.json();
        
        renderSelfImprovementPerformance(data.performance || {});
        renderSelfImprovementDiagnostic(data.diagnostic || {});
        renderSelfImprovementProposals(data.proposals || []);
        renderSelfImprovementSafety(data.safety || {});
    } catch (error) {
        console.error('Failed to load self improvement:', error);
        renderSelfImprovementFallback();
    }
}

function renderSelfImprovementFallback() {
    dom.improvementPerformance.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
    dom.improvementDiagnostic.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
    dom.improvementProposals.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
    dom.improvementSafety.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
}

function renderSelfImprovementPerformance(performance) {
    dom.improvementPerformance.innerHTML = `
        <div class="status-item">
            <span class="status-label">综合评分</span>
            <span class="status-value">${performance.overall_score || 0}</span>
        </div>
        <div class="status-item">
            <span class="status-label">推理效率</span>
            <span class="status-value">${performance.reasoning_efficiency || 0}</span>
        </div>
        <div class="status-item">
            <span class="status-label">学习能力</span>
            <span class="status-value">${performance.learning_capability || 0}</span>
        </div>
        <div class="status-item">
            <span class="status-label">稳定性</span>
            <span class="status-value">${performance.stability || 0}</span>
        </div>
    `;
}

function renderSelfImprovementDiagnostic(diagnostic) {
    dom.improvementDiagnostic.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 8px;">
            ${diagnostic.issues?.map((issue, i) => `
                <div style="padding: 8px; background: var(--bg-secondary); border-radius: var(--radius-sm); font-size: 12px;">
                    <span style="color: ${issue.severity === 'high' ? 'var(--error-color)' : 'var(--warning-color)'};">${i + 1}. ${issue.description}</span>
                </div>
            `).join('') || '<div style="color: var(--text-muted);">暂无诊断问题</div>'}
        </div>
    `;
}

function renderSelfImprovementProposals(proposals) {
    if (proposals.length === 0) {
        dom.improvementProposals.innerHTML = '<div style="color: var(--text-muted);">暂无改进提案</div>';
        return;
    }
    
    dom.improvementProposals.innerHTML = proposals.map(p => `
        <div style="padding: 12px; background: var(--bg-secondary); border-radius: var(--radius-sm); margin-bottom: 8px;">
            <h4 style="font-size: 13px; margin-bottom: 4px;">${p.title}</h4>
            <p style="font-size: 12px; color: var(--text-secondary);">${p.description}</p>
            <span style="font-size: 11px; color: var(--text-muted);">优先级: ${p.priority}</span>
        </div>
    `).join('');
}

function renderSelfImprovementSafety(safety) {
    dom.improvementSafety.innerHTML = `
        <div class="status-item">
            <span class="status-dot ${safety.verified ? 'online' : 'warning'}"></span>
            <span class="status-label">安全验证</span>
            <span class="status-value">${safety.verified ? '已验证' : '待验证'}</span>
        </div>
        <div class="status-item">
            <span class="status-label">验证次数</span>
            <span class="status-value">${safety.verification_count || 0}</span>
        </div>
        <div class="status-item">
            <span class="status-label">安全等级</span>
            <span class="status-value">${safety.level || '中等'}</span>
        </div>
    `;
}

function runDiagnostic() {
    fetch(`${API_BASE}/api/self_improvement/diagnose`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        addMessage('assistant', '诊断已完成');
        loadSelfImprovement();
    })
    .catch(error => console.error('Run diagnostic failed:', error));
}

function generateProposals() {
    fetch(`${API_BASE}/api/self_improvement/proposals`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        addMessage('assistant', '改进提案已生成');
        loadSelfImprovement();
    })
    .catch(error => console.error('Generate proposals failed:', error));
}

async function loadSkills() {
    try {
        const response = await fetch(`${API_BASE}/api/skills/installed`);
        const data = await response.json();
        
        renderSkills(data.skills || []);
    } catch (error) {
        console.error('Failed to load skills:', error);
        dom.skillsList.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 40px;">加载失败</div>';
    }
}

function renderSkills(skills) {
    if (skills.length === 0) {
        dom.skillsList.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 40px;">暂无技能</div>';
        return;
    }
    
    dom.skillsList.innerHTML = skills.map(skill => `
        <div class="skill-card">
            <h4>${skill.name || '未命名技能'}</h4>
            <p>${skill.description || '无描述'}</p>
            <div style="display: flex; gap: 8px; margin-top: 8px;">
                <span style="padding: 4px 8px; background: var(--bg-secondary); border-radius: 4px; font-size: 11px;">状态: ${skill.has_skill_md ? '已安装' : '未安装'}</span>
                <span style="padding: 4px 8px; background: var(--bg-secondary); border-radius: 4px; font-size: 11px;">版本: ${skill.version || '1.0'}</span>
            </div>
        </div>
    `).join('');
}

async function loadKnowledge() {
    try {
        const response = await fetch(`${API_BASE}/api/knowledge/graph`);
        const data = await response.json();
        
        renderKnowledgeStats(data.stats || {});
        renderKnowledgeGraph(data.graph || {});
    } catch (error) {
        console.error('Failed to load knowledge:', error);
        dom.knowledgeStats.innerHTML = '<div style="color: var(--text-muted);">加载失败</div>';
        dom.knowledgeGraph.innerHTML = '<div class="knowledge-graph-visual">加载失败</div>';
    }
}

function renderKnowledgeStats(stats) {
    dom.knowledgeStats.innerHTML = `
        <span>节点数: <strong>${stats.nodes || 0}</strong></span>
        <span>边数: <strong>${stats.edges || 0}</strong></span>
        <span>相似度阈值: <strong>${stats.similarity_threshold || 0.8}</strong></span>
    `;
}

function renderKnowledgeGraph(graph) {
    if (!graph.nodes || graph.nodes.length === 0) {
        dom.knowledgeGraph.innerHTML = '<div class="knowledge-graph-visual">暂无知识图谱数据</div>';
        return;
    }
    
    dom.knowledgeGraph.innerHTML = `
        <div class="knowledge-graph-visual">
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <h4 style="color: var(--text-primary);">节点列表</h4>
                ${graph.nodes.map(node => `
                    <div style="padding: 8px 12px; background: var(--bg-secondary); border-radius: var(--radius-sm); font-size: 13px;">
                        <span style="color: var(--primary-color);">${node.name || node.id}</span>
                        ${node.category ? `<span style="color: var(--text-muted); margin-left: 8px;">[${node.category}]</span>` : ''}
                    </div>
                `).join('')}
            </div>
            ${graph.edges && graph.edges.length > 0 ? `
                <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 16px;">
                    <h4 style="color: var(--text-primary);">关系边</h4>
                    ${graph.edges.map(edge => `
                        <div style="padding: 8px 12px; background: var(--bg-secondary); border-radius: var(--radius-sm); font-size: 12px; color: var(--text-secondary);">
                            ${edge.source} --${edge.relation || '关联'}--> ${edge.target}
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

async function loadSynapticActivity() {
    try {
        const response = await fetch(`${API_BASE}/api/synaptic/activity`);
        const data = await response.json();
        
        renderSynapticStats(data);
        renderModuleActivity(data.modules || {});
    } catch (error) {
        console.error('Failed to load synaptic activity:', error);
    }
}

function renderSynapticStats(data) {
    if (!dom.synapticStats) return;
    
    dom.synapticStats.innerHTML = `
        <span class="synaptic-stat-item"><span>模块数:</span><span>${data.total_modules || 0}</span></span>
        <span class="synaptic-stat-item"><span>突触数:</span><span>${data.total_synapses || 0}</span></span>
        <span class="synaptic-stat-item"><span>活跃突触:</span><span>${data.active_synapses || 0}</span></span>
    `;
}

function renderModuleActivity(modules) {
    if (!dom.moduleActivityList) return;
    
    const moduleNames = {
        'memory': '记忆', 'knowledge_graph': '知识图谱', 'decision': '决策',
        'execution': '执行', 'perception': '感知', 'security': '安全',
        'soul': 'SOUL', 'skills': '技能', 'evolution': '进化',
        'self_improvement': '自我改进', 'metacognition': '元认知', 'homeostasis': '稳态'
    };
    
    dom.moduleActivityList.innerHTML = Object.entries(modules).map(([id, module]) => {
        const activity = module.spike_rate || 0;
        return `
            <div class="module-activity-item">
                <span class="module-activity-name">${moduleNames[id] || id}</span>
                <div class="module-activity-bar">
                    <div class="module-activity-fill" style="width: ${Math.min(100, activity * 10)}%"></div>
                </div>
            </div>
        `;
    }).join('');
}

async function loadSynapticConnections() {
    try {
        const response = await fetch(`${API_BASE}/api/synaptic/connections`);
        const data = await response.json();
        
        renderSynapticGraph(data);
    } catch (error) {
        console.error('Failed to load synaptic connections:', error);
    }
}

function renderSynapticGraph(data) {
    if (!dom.synapticGraph || !data.nodes || !data.edges) return;
    
    const width = dom.synapticGraph.clientWidth;
    const height = dom.synapticGraph.clientHeight;
    
    const nodePositions = {};
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 50;
    
    data.nodes.forEach((node, index) => {
        const angle = (index / data.nodes.length) * 2 * Math.PI - Math.PI / 2;
        nodePositions[node.id] = {
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle)
        };
    });
    
    const edgeColors = {
        'excitatory': '#22c55e',
        'inhibitory': '#ef4444',
        'modulatory': '#8b5cf6'
    };
    
    dom.synapticGraph.innerHTML = `
        <svg width="${width}" height="${height}">
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            ${data.edges.map(edge => {
                const source = nodePositions[edge.source];
                const target = nodePositions[edge.target];
                if (!source || !target) return '';
                
                const color = edgeColors[edge.type] || '#6b7280';
                const width = Math.max(1, edge.weight * 3);
                
                return `
                    <line x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}"
                        class="synaptic-edge"
                        stroke="${color}" stroke-width="${width}"
                        style="opacity: ${edge.weight}"/>
                `;
            }).join('')}
            ${data.nodes.map(node => {
                const pos = nodePositions[node.id];
                if (!pos) return '';
                
                return `
                    <g class="synaptic-node" data-node="${node.id}">
                        <circle cx="${pos.x}" cy="${pos.y}" r="16"
                            fill="var(--bg-card)" stroke="var(--primary-color)" stroke-width="2"
                            filter="url(#glow)"/>
                        <text x="${pos.x}" y="${pos.y + 4}" text-anchor="middle"
                            fill="var(--text-primary)" font-size="10">
                            ${node.name.substring(0, 4)}
                        </text>
                    </g>
                `;
            }).join('')}
        </svg>
    `;
}

async function loadOscillatorStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/synaptic/oscillator`);
        const data = await response.json();
        
        renderOscillator(data);
    } catch (error) {
        console.error('Failed to load oscillator:', error);
    }
}

function renderOscillator(data) {
    if (!dom.oscillatorDisplay) return;
    
    dom.oscillatorDisplay.innerHTML = `
        <div>
            <div class="oscillator-wave">
                <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
                    <div style="width: 100%; height: 2px; background: var(--border-color); position: relative;">
                        <div style="position: absolute; top: 50%; left: ${(data.theta + 1) / 2 * 100}%; width: 6px; height: 6px; background: var(--primary-color); border-radius: 50%; transform: translateY(-50%);"></div>
                    </div>
                </div>
            </div>
            <span class="oscillator-label">Theta (5Hz): ${data.theta ? data.theta.toFixed(2) : '--'}</span>
        </div>
        <div>
            <div class="oscillator-wave">
                <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
                    <div style="width: 100%; height: 2px; background: var(--border-color); position: relative;">
                        <div style="position: absolute; top: 50%; left: ${(data.gamma + 1) / 2 * 100}%; width: 6px; height: 6px; background: var(--success-color); border-radius: 50%; transform: translateY(-50%);"></div>
                    </div>
                </div>
            </div>
            <span class="oscillator-label">Gamma (40Hz): ${data.gamma ? data.gamma.toFixed(2) : '--'}</span>
        </div>
        <div>
            <div class="oscillator-wave">
                <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
                    <div style="width: 100%; height: 2px; background: var(--border-color); position: relative;">
                        <div style="position: absolute; top: 50%; left: ${(data.alpha + 1) / 2 * 100}%; width: 6px; height: 6px; background: var(--warning-color); border-radius: 50%; transform: translateY(-50%);"></div>
                    </div>
                </div>
            </div>
            <span class="oscillator-label">Alpha (10Hz): ${data.alpha ? data.alpha.toFixed(2) : '--'}</span>
        </div>
    `;
}

async function loadSignalFlow() {
    try {
        const response = await fetch(`${API_BASE}/api/synaptic/signal_flow`);
        const data = await response.json();
        
        renderSignalFlow(data);
    } catch (error) {
        console.error('Failed to load signal flow:', error);
    }
}

function renderSignalFlow(data) {
    if (!dom.signalFlowChart || !data.signal_flow) return;
    
    const flowItems = [];
    Object.entries(data.signal_flow).forEach(([module, stats]) => {
        flowItems.push({
            module,
            spikes: stats.spikes || 0
        });
    });
    
    flowItems.sort((a, b) => b.spikes - a.spikes);
    
    const moduleNames = {
        'memory': '记忆', 'knowledge_graph': '知识图谱', 'decision': '决策',
        'execution': '执行', 'perception': '感知', 'security': '安全',
        'soul': 'SOUL', 'skills': '技能', 'evolution': '进化',
        'self_improvement': '自我改进', 'metacognition': '元认知', 'homeostasis': '稳态'
    };
    
    dom.signalFlowChart.innerHTML = flowItems.map(item => `
        <div class="signal-flow-item">
            <div class="signal-flow-icon"></div>
            <span class="signal-flow-path">${moduleNames[item.module] || item.module}</span>
            <span class="signal-flow-value">${item.spikes} 脉冲</span>
        </div>
    `).join('');
}

async function loadSynapticData() {
    await Promise.all([
        loadSynapticActivity(),
        loadSynapticConnections(),
        loadOscillatorStatus(),
        loadSignalFlow()
    ]);
}

async function init() {
    initEventListeners();
    connectRealtimeSocket();
    updateBusStatusIndicator();
    checkAgentStatus();
    refreshOverview();
    loadSessions();
    loadAgents();
}

document.addEventListener('DOMContentLoaded', init);