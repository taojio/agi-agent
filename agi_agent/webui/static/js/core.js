const App = window.App || {};

App.core = {
    API_BASE: window.location.origin,

    state: {
        isAgentRunning: false,
        isSensorsEnabled: true,
        isVoiceRecording: false,
        recognition: null,
        serverAvailable: true,
        lastServerErrorTime: 0,
        currentChannel: 'general',
        channelMessages: {},
        realtimeSocket: null,
        realtimeConnected: false,
        busHasData: false,
        currentSessionId: null,
        sessions: [],
        currentMemoryTier: 'L1'
    },

    dom: {},

    initDOM() {
        this.dom = {
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
    },

    modalPromptCallback: null,

    showModalPrompt(title, placeholder, defaultValue = '') {
        return new Promise((resolve) => {
            this.modalPromptCallback = resolve;
            this.dom.modalTitle.textContent = title;
            this.dom.modalInput.placeholder = placeholder;
            this.dom.modalInput.value = defaultValue;
            this.dom.modalOverlay.style.display = 'flex';
            setTimeout(() => this.dom.modalInput.focus(), 50);
        });
    },

    closeModal(value = null) {
        this.dom.modalOverlay.style.display = 'none';
        this.dom.modalInput.value = '';
        if (this.modalPromptCallback) {
            this.modalPromptCallback(value);
            this.modalPromptCallback = null;
        }
    },

    initModalEvents() {
        const self = this;
        this.dom.modalConfirm.addEventListener('click', () => {
            self.closeModal(self.dom.modalInput.value.trim() || null);
        });

        this.dom.modalInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                self.closeModal(self.dom.modalInput.value.trim() || null);
            } else if (e.key === 'Escape') {
                self.closeModal(null);
            }
        });

        this.dom.modalOverlay.addEventListener('click', (e) => {
            if (e.target === self.dom.modalOverlay) {
                self.closeModal(null);
            }
        });
    },

    formatTime(timestamp) {
        if (!timestamp) return '刚刚';
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) return '刚刚';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
        return date.toLocaleDateString();
    },

    toArray(data) {
        if (data === null || data === undefined) return [];
        if (Array.isArray(data)) return data;
        if (typeof data === 'object') return Object.values(data);
        try { return Array.from(data); } catch(e) { return []; }
    },

    shouldLogServerError() {
        const now = Date.now();
        if (now - this.state.lastServerErrorTime > 30000) {
            this.state.lastServerErrorTime = now;
            return true;
        }
        return false;
    },

    init() {
        this.initDOM();
        this.initModalEvents();
    }
};

window.closeModal = () => App.core.closeModal();

window.App = App;
