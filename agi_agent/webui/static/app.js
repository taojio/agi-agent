const API_BASE = window.location.origin;

let metricsWebSocket = null;
let sensorWebSocket = null;
let isAgentRunning = false;
let isSensorsEnabled = true;
let isVoiceRecording = false;
let recognition = null;

let serverAvailable = true;
let lastServerErrorTime = 0;

let currentChannel = 'general';
let channelMessages = {};
let onlineAgentsRefreshInterval = null;

function shouldLogServerError() {
    const now = Date.now();
    if (now - lastServerErrorTime > 30000) {
        lastServerErrorTime = now;
        return true;
    }
    return false;
}

const dom = {
    agentStatus: document.getElementById('agentStatus'),
    statusDot: document.querySelector('.status-dot'),
    startAgentBtn: document.getElementById('startAgentBtn'),
    stopAgentBtn: document.getElementById('stopAgentBtn'),
    chatContainer: document.getElementById('chatContainer'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    voiceBtn: document.getElementById('voiceBtn'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    sensorToggle: document.getElementById('sensorToggle'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),
    metricFe: document.getElementById('metricFe'),
    metricConfidence: document.getElementById('metricConfidence'),
    metricNovelty: document.getElementById('metricNovelty'),
    metricEntropy: document.getElementById('metricEntropy'),
    metricStep: document.getElementById('metricStep'),
    metricLatency: document.getElementById('metricLatency'),
    barFe: document.getElementById('barFe'),
    barConfidence: document.getElementById('barConfidence'),
    barNovelty: document.getElementById('barNovelty'),
    barEntropy: document.getElementById('barEntropy'),
    screenResolution: document.getElementById('screenResolution'),
    screenColorDepth: document.getElementById('screenColorDepth'),
    screenRefreshRate: document.getElementById('screenRefreshRate'),
    audioMicrophone: document.getElementById('audioMicrophone'),
    audioSpeaker: document.getElementById('audioSpeaker'),
    audioVolume: document.getElementById('audioVolume'),
    audioStatus: document.getElementById('audioStatus'),
    keyboardKeys: document.getElementById('keyboardKeys'),
    keyboardLayout: document.getElementById('keyboardLayout'),
    keyboardCaps: document.getElementById('keyboardCaps'),
    keyboardStatus: document.getElementById('keyboardStatus'),
    mousePosition: document.getElementById('mousePosition'),
    mouseScroll: document.getElementById('mouseScroll'),
    mouseButtons: document.getElementById('mouseButtons'),
    mouseStatus: document.getElementById('mouseStatus'),
    settingInputDim: document.getElementById('settingInputDim'),
    settingMaxSteps: document.getElementById('settingMaxSteps'),
    settingLogInterval: document.getElementById('settingLogInterval'),
    settingSaveInterval: document.getElementById('settingSaveInterval'),
    settingFeThreshold: document.getElementById('settingFeThreshold'),
    settingFeThresholdValue: document.getElementById('settingFeThresholdValue'),
    settingNoveltyThreshold: document.getElementById('settingNoveltyThreshold'),
    settingNoveltyThresholdValue: document.getElementById('settingNoveltyThresholdValue'),
    settingAutoStart: document.getElementById('settingAutoStart'),
    settingSensorEnabled: document.getElementById('settingSensorEnabled'),
    settingVoiceInput: document.getElementById('settingVoiceInput'),
    settingTheme: document.getElementById('settingTheme'),
    
    reflexStatus: document.getElementById('reflexStatus'),
    reflexSpikeRate: document.getElementById('reflexSpikeRate'),
    reflexRuleMatch: document.getElementById('reflexRuleMatch'),
    reflexResponse: document.getElementById('reflexResponse'),
    reflexTotalRules: document.getElementById('reflexTotalRules'),
    reflexTotalPatterns: document.getElementById('reflexTotalPatterns'),
    reflexAvgConfidence: document.getElementById('reflexAvgConfidence'),
    reflexDelegateRate: document.getElementById('reflexDelegateRate'),
    deliberativeStatus: document.getElementById('deliberativeStatus'),
    delibCycles: document.getElementById('delibCycles'),
    delibCompleted: document.getElementById('delibCompleted'),
    delibConfidence: document.getElementById('delibConfidence'),
    delibSolutions: document.getElementById('delibSolutions'),
    delibPhase: document.getElementById('delibPhase'),
    delibMode: document.getElementById('delibMode'),
    delibSystem1: document.getElementById('delibSystem1'),
    delibSystem2: document.getElementById('delibSystem2'),
    delibSystem1Ratio: document.getElementById('delibSystem1Ratio'),
    metaStatus: document.getElementById('metaStatus'),
    metaViolations: document.getElementById('metaViolations'),
    metaStrategyChanges: document.getElementById('metaStrategyChanges'),
    metaLearningCycles: document.getElementById('metaLearningCycles'),
    metaSelfName: document.getElementById('metaSelfName'),
    metaSelfRole: document.getElementById('metaSelfRole'),
    metaHealthScore: document.getElementById('metaHealthScore'),
    metaCompLoad: document.getElementById('metaCompLoad'),
    metaMemoryUsage: document.getElementById('metaMemoryUsage'),
    metaThinkingStrategy: document.getElementById('metaThinkingStrategy'),
    metaActionStrategy: document.getElementById('metaActionStrategy'),
    metaStrengths: document.getElementById('metaStrengths'),
    metaWeaknesses: document.getElementById('metaWeaknesses'),
    metaAvgSuccessRate: document.getElementById('metaAvgSuccessRate'),
    
    actionDecomposition: document.getElementById('actionDecomposition'),
    actionPathPlanning: document.getElementById('actionPathPlanning'),
    actionExecutions: document.getElementById('actionExecutions'),
    actionSuccessRate: document.getElementById('actionSuccessRate'),
    actionActiveGoals: document.getElementById('actionActiveGoals'),
    actionExploration: document.getElementById('actionExploration'),
    
    microReinforcements: document.getElementById('microReinforcements'),
    mesoRuleUpdates: document.getElementById('mesoRuleUpdates'),
    mesoSkillGenerations: document.getElementById('mesoSkillGenerations'),
    macroOptimizations: document.getElementById('macroOptimizations'),
    metaOptimizations: document.getElementById('metaOptimizations'),
    
    securityRules: document.getElementById('securityRules'),
    securityViolations: document.getElementById('securityViolations'),
    securityRiskLevel: document.getElementById('securityRiskLevel'),
    securityCircuitBreaker: document.getElementById('securityCircuitBreaker'),
    
    onlineAgentsList: document.getElementById('onlineAgentsList'),
    
    selfSelfRecognition: document.getElementById('selfSelfRecognition'),
    selfCapabilityAwareness: document.getElementById('selfCapabilityAwareness'),
    selfLimitationAwareness: document.getElementById('selfLimitationAwareness'),
    selfExistenceAwareness: document.getElementById('selfExistenceAwareness'),
    selfTemporalContinuity: document.getElementById('selfTemporalContinuity'),
    barSelfRecognition: document.getElementById('barSelfRecognition'),
    barCapabilityAwareness: document.getElementById('barCapabilityAwareness'),
    barLimitationAwareness: document.getElementById('barLimitationAwareness'),
    barExistenceAwareness: document.getElementById('barExistenceAwareness'),
    barTemporalContinuity: document.getElementById('barTemporalContinuity'),
    identityName: document.getElementById('identityName'),
    identityRole: document.getElementById('identityRole'),
    identityGoals: document.getElementById('identityGoals'),
    introspectionHistory: document.getElementById('introspectionHistory'),
    
    thinkingMode: document.getElementById('thinkingMode'),
    thinkingSystem2: document.getElementById('thinkingSystem2'),
    thinkingConfidence: document.getElementById('thinkingConfidence'),
    problemInput: document.getElementById('problemInput'),
    decomposeBtn: document.getElementById('decomposeBtn'),
    decomposeResult: document.getElementById('decomposeResult'),
    criticalInput: document.getElementById('criticalInput'),
    criticalBtn: document.getElementById('criticalBtn'),
    criticalResult: document.getElementById('criticalResult'),
    
    decisionCount: document.getElementById('decisionCount'),
    decisionActiveGoals: document.getElementById('decisionActiveGoals'),
    simGoal: document.getElementById('simGoal'),
    simConfidence: document.getElementById('simConfidence'),
    simConfidenceValue: document.getElementById('simConfidenceValue'),
    simRiskLevel: document.getElementById('simRiskLevel'),
    makeDecisionBtn: document.getElementById('makeDecisionBtn'),
    decisionResult: document.getElementById('decisionResult'),
    
    personalitySignature: document.getElementById('personalitySignature'),
    personalityConsistency: document.getElementById('personalityConsistency'),
    barTraitCuriosity: document.getElementById('barTraitCuriosity'),
    barTraitAssertiveness: document.getElementById('barTraitAssertiveness'),
    barTraitCautiousness: document.getElementById('barTraitCautiousness'),
    barTraitCreativity: document.getElementById('barTraitCreativity'),
    barTraitPatience: document.getElementById('barTraitPatience'),
    barValueSurvival: document.getElementById('barValueSurvival'),
    barValueKnowledge: document.getElementById('barValueKnowledge'),
    barValueGrowth: document.getElementById('barValueGrowth'),
    
    memL1Count: document.getElementById('memL1Count'),
    memL2Count: document.getElementById('memL2Count'),
    memL3Count: document.getElementById('memL3Count'),
    memL4Count: document.getElementById('memL4Count'),
    memBarL1: document.getElementById('memBarL1'),
    memBarL2: document.getElementById('memBarL2'),
    memBarL3: document.getElementById('memBarL3'),
    memBarL4: document.getElementById('memBarL4'),
    memTotalCapacity: document.getElementById('memTotalCapacity'),
    memUsed: document.getElementById('memUsed'),
    memUtilization: document.getElementById('memUtilization'),
    memoryRecentList: document.getElementById('memoryRecentList'),
    
    kgNodeCount: document.getElementById('kgNodeCount'),
    kgEdgeCount: document.getElementById('kgEdgeCount'),
    kgClusterCount: document.getElementById('kgClusterCount'),
    kgActivity: document.getElementById('kgActivity'),
    knowledgeRecentList: document.getElementById('knowledgeRecentList'),
    
    learningPercentage: document.getElementById('learningPercentage'),
    learningCycles: document.getElementById('learningCycles'),
    learningKnowledge: document.getElementById('learningKnowledge'),
    learningSkills: document.getElementById('learningSkills'),
    learningRlProgress: document.getElementById('learningRlProgress'),
    learningEvoProgress: document.getElementById('learningEvoProgress'),
    learningMetaProgress: document.getElementById('learningMetaProgress'),
    barLearningRl: document.getElementById('barLearningRl'),
    barLearningEvo: document.getElementById('barLearningEvo'),
    barLearningMeta: document.getElementById('barLearningMeta'),
    
    logsList: document.getElementById('logsList'),
    logLevelFilter: document.getElementById('logLevelFilter'),
    
    uploadDropZone: document.getElementById('uploadDropZone'),
    fileInput: document.getElementById('fileInput'),
    fileSearchInput: document.getElementById('fileSearchInput'),
    fileSearchBtn: document.getElementById('fileSearchBtn'),
    fileSearchType: document.getElementById('fileSearchType'),
    fileTotalRecords: document.getElementById('fileTotalRecords'),
    fileTypeDistribution: document.getElementById('fileTypeDistribution'),
    fileIngestionList: document.getElementById('fileIngestionList'),
    fileStatusMessage: document.getElementById('fileStatusMessage'),
    refreshFileStatsBtn: document.getElementById('refreshFileStatsBtn')
};

let mentalArchitectureRefreshInterval = null;



async function checkAgentStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/agent/status`);
        const data = await response.json();
        if (!serverAvailable) {
            serverAvailable = true;
            console.log('Server connection restored');
        }
        isAgentRunning = data.status === 'running';
        updateAgentStatus(isAgentRunning);
    } catch (error) {
        if (serverAvailable) {
            serverAvailable = false;
            console.warn('Server unavailable:', error.message || error);
        }
    }
}

function updateAgentStatus(running) {
    isAgentRunning = running;
    if (running) {
        dom.statusDot.classList.remove('offline');
        dom.statusDot.classList.add('online');
        dom.agentStatus.querySelector('span:last-child').textContent = '运行中';
        dom.startAgentBtn.disabled = true;
        dom.stopAgentBtn.disabled = false;
    } else {
        dom.statusDot.classList.remove('online');
        dom.statusDot.classList.add('offline');
        dom.agentStatus.querySelector('span:last-child').textContent = '未启动';
        dom.startAgentBtn.disabled = false;
        dom.stopAgentBtn.disabled = true;
    }
}

function initEventListeners() {
    dom.startAgentBtn.addEventListener('click', startAgent);
    dom.stopAgentBtn.addEventListener('click', stopAgent);
    dom.sendBtn.addEventListener('click', sendMessage);
    dom.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    dom.voiceBtn.addEventListener('click', toggleVoiceInput);
    dom.clearChatBtn.addEventListener('click', clearChat);
    dom.sensorToggle.addEventListener('change', (e) => {
        isSensorsEnabled = e.target.checked;
        if (isSensorsEnabled) {
            initSensors();
        } else {
            stopSensors();
        }
    });
    dom.saveSettingsBtn.addEventListener('click', saveSettings);
    
    dom.settingFeThreshold.addEventListener('input', (e) => {
        dom.settingFeThresholdValue.textContent = e.target.value;
    });
    dom.settingNoveltyThreshold.addEventListener('input', (e) => {
        dom.settingNoveltyThresholdValue.textContent = e.target.value;
    });
    
    dom.settingTheme.addEventListener('change', (e) => {
        document.body.className = e.target.value;
        saveSettings();
    });

    const scanBtn = document.getElementById('scanPluginsBtn');
    const loadAllBtn = document.getElementById('loadAllPluginsBtn');
    if (scanBtn) scanBtn.addEventListener('click', loadPlugins);
    if (loadAllBtn) loadAllBtn.addEventListener('click', loadAllPlugins);

    // Skills 技能商店
    const skillSearchBtn = document.getElementById('skillSearchBtn');
    const skillSearchInput = document.getElementById('skillSearchInput');
    if (skillSearchBtn) skillSearchBtn.addEventListener('click', searchSkills);
    if (skillSearchInput) {
        skillSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchSkills();
        });
    }
    document.querySelectorAll('.skills-tab').forEach(tab => {
        tab.addEventListener('click', () => switchSkillsTab(tab.dataset.tab));
    });

    // 自我意识面板
    const refreshSelfAwarenessBtn = document.getElementById('refreshSelfAwarenessBtn');
    if (refreshSelfAwarenessBtn) refreshSelfAwarenessBtn.addEventListener('click', refreshSelfAwareness);
    
    // 思考面板
    if (dom.decomposeBtn) dom.decomposeBtn.addEventListener('click', decomposeProblem);
    if (dom.criticalBtn) dom.criticalBtn.addEventListener('click', criticalAnalyze);
    if (dom.problemInput) {
        dom.problemInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') decomposeProblem();
        });
    }
    if (dom.criticalInput) {
        dom.criticalInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') criticalAnalyze();
        });
    }
    
    // 决策面板
    if (dom.makeDecisionBtn) dom.makeDecisionBtn.addEventListener('click', simulateDecision);
    if (dom.simConfidence) {
        dom.simConfidence.addEventListener('input', (e) => {
            dom.simConfidenceValue.textContent = e.target.value;
        });
    }
    
    // 个性面板
    const refreshPersonalityBtn = document.getElementById('refreshPersonalityBtn');
    if (refreshPersonalityBtn) refreshPersonalityBtn.addEventListener('click', refreshPersonality);
    
    // 记忆面板
    const clearMemoryBtn = document.getElementById('clearMemoryBtn');
    if (clearMemoryBtn) clearMemoryBtn.addEventListener('click', clearMemory);
    
    // 知识图谱面板
    const refreshKnowledgeBtn = document.getElementById('refreshKnowledgeBtn');
    if (refreshKnowledgeBtn) refreshKnowledgeBtn.addEventListener('click', refreshKnowledge);
    
    // 日志面板
    const clearLogsBtn = document.getElementById('clearLogsBtn');
    if (clearLogsBtn) clearLogsBtn.addEventListener('click', clearLogs);
    if (dom.logLevelFilter) {
        dom.logLevelFilter.addEventListener('change', filterLogs);
    }

    // 文件摄入面板
    if (dom.uploadDropZone) {
        dom.uploadDropZone.addEventListener('click', () => dom.fileInput.click());
        dom.uploadDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dom.uploadDropZone.classList.add('dragover');
        });
        dom.uploadDropZone.addEventListener('dragleave', () => {
            dom.uploadDropZone.classList.remove('dragover');
        });
        dom.uploadDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dom.uploadDropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files);
            }
        });
    }
    if (dom.fileInput) {
        dom.fileInput.addEventListener('change', (e) => {
            handleFileUpload(e.target.files);
        });
    }
    if (dom.fileSearchBtn) {
        dom.fileSearchBtn.addEventListener('click', searchFiles);
    }
    if (dom.fileSearchInput) {
        dom.fileSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchFiles();
        });
    }
    if (dom.refreshFileStatsBtn) {
        dom.refreshFileStatsBtn.addEventListener('click', refreshFileStats);
    }

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('wheel', handleMouseWheel);
    window.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);
}

async function startAgent() {
    try {
        const response = await fetch(`${API_BASE}/api/agent/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        if (data.status === 'success') {
            updateAgentStatus(true);
            addMessage('agent', data.message);
            initWebSocket();
            startMentalArchitectureRefresh();
        }
    } catch (error) {
        console.error('Failed to start agent:', error);
        addMessage('agent', '启动智能体失败，请检查服务器连接');
    }
}

async function stopAgent() {
    try {
        const response = await fetch(`${API_BASE}/api/agent/stop`, {
            method: 'POST'
        });
        const data = await response.json();
        if (data.status === 'success') {
            updateAgentStatus(false);
            addMessage('agent', data.message);
            closeWebSocket();
            stopMentalArchitectureRefresh();
            stopOnlineAgentsRefresh();
        }
    } catch (error) {
        console.error('Failed to stop agent:', error);
    }
}

async function sendMessage() {
    const content = dom.chatInput.value.trim();
    if (!content) return;
    
    if (currentChannel === 'direct') {
        addMessage('user', content);
        dom.chatInput.value = '';
        
        try {
            const response = await fetch(`${API_BASE}/api/chat/send`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content })
            });
            const data = await response.json();
            addMessage('agent', data.response);
        } catch (error) {
            console.error('Failed to send message:', error);
            addMessage('agent', '发送消息失败，请检查网络连接');
        }
    } else {
        await sendMessageToChannel(content);
    }
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    let icon = '';
    if (role === 'agent') {
        icon = '<span class="agent-icon">🤖</span>';
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            ${icon}
            <p>${content}</p>
        </div>
    `;
    
    dom.chatContainer.appendChild(messageDiv);
    dom.chatContainer.scrollTop = dom.chatContainer.scrollHeight;
}

function clearChat() {
    dom.chatContainer.innerHTML = `
        <div class="message agent-message">
            <div class="message-content">
                <span class="agent-icon">🤖</span>
                <p>欢迎使用AGI智能体！我已准备就绪，随时为您服务。</p>
            </div>
        </div>
    `;
}

function toggleVoiceInput() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        addMessage('agent', '您的浏览器不支持语音识别功能');
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
        addMessage('agent', '正在听...请说话');
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        dom.chatInput.value = transcript;
        addMessage('user', transcript);
        stopVoiceRecording();
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        addMessage('agent', '语音识别出错，请重试');
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

function initWebSocket() {
    closeWebSocket();

    if (!isAgentRunning) {
        return;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

    metricsWebSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/metrics`);
    metricsWebSocket.onopen = () => {
        console.log('Metrics WebSocket connected');
    };
    metricsWebSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'metrics') {
            updateMetrics(data.data);
        }
    };
    metricsWebSocket.onerror = (error) => {
        if (isAgentRunning) console.warn('Metrics WebSocket error:', error?.message || error);
    };
    metricsWebSocket.onclose = function() {
        console.log('Metrics WebSocket disconnected, reconnecting...');
        setTimeout(() => {
            if (isAgentRunning) {
                initWebSocket();
            }
        }, 3000);
    };

    sensorWebSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/sensors`);
    sensorWebSocket.onopen = () => {
        console.log('Sensors WebSocket connected');
    };
    sensorWebSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Sensor ack:', data);
    };
    sensorWebSocket.onerror = (error) => {
        if (isSensorsEnabled) console.warn('Sensors WebSocket error:', error?.message || error);
    };
    sensorWebSocket.onclose = function() {
        console.log('Sensors WebSocket disconnected, reconnecting...');
        setTimeout(() => {
            if (isAgentRunning) {
                initWebSocket();
            }
        }, 3000);
    };
}

function closeWebSocket() {
    if (metricsWebSocket) {
        metricsWebSocket.close();
        metricsWebSocket = null;
    }
    if (sensorWebSocket) {
        sensorWebSocket.close();
        sensorWebSocket = null;
    }
}

function updateMetrics(metrics) {
    dom.metricFe.textContent = metrics.free_energy?.toFixed(4) || '--';
    dom.metricConfidence.textContent = metrics.confidence?.toFixed(4) || '--';
    dom.metricNovelty.textContent = metrics.novelty?.toFixed(4) || '--';
    dom.metricEntropy.textContent = metrics.entropy?.toFixed(4) || '--';
    dom.metricStep.textContent = metrics.step?.toString() || '--';
    dom.metricLatency.textContent = metrics.latency?.toFixed(1) || '--';
    
    dom.barFe.style.width = `${Math.min(100, (metrics.free_energy || 0) * 100)}%`;
    dom.barConfidence.style.width = `${Math.min(100, (metrics.confidence || 0) * 100)}%`;
    dom.barNovelty.style.width = `${Math.min(100, (metrics.novelty || 0) * 100)}%`;
    dom.barEntropy.style.width = `${Math.min(100, (metrics.entropy || 0) * 20)}%`;
}

function initSensors() {
    updateScreenInfo();
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then((stream) => {
            dom.audioMicrophone.textContent = '已连接';
            dom.audioStatus.textContent = '活动';
            dom.audioStatus.classList.add('active');
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(audioContext.destination);
        })
        .catch(() => {
            dom.audioMicrophone.textContent = '未授权';
            dom.audioStatus.textContent = '未连接';
            dom.audioStatus.classList.remove('active');
        });
    
    dom.audioSpeaker.textContent = '可用';
    dom.audioVolume.textContent = '50%';
    
    dom.keyboardLayout.textContent = navigator.language || 'zh-CN';
    dom.keyboardCaps.textContent = '关闭';
}

function stopSensors() {
    dom.screenResolution.textContent = '--';
    dom.screenColorDepth.textContent = '--';
    dom.screenRefreshRate.textContent = '--';
    dom.audioMicrophone.textContent = '--';
    dom.audioSpeaker.textContent = '--';
    dom.audioVolume.textContent = '--';
    dom.keyboardKeys.textContent = '无';
    dom.keyboardLayout.textContent = '--';
    dom.keyboardCaps.textContent = '--';
    dom.mousePosition.textContent = '--';
    dom.mouseScroll.textContent = '--';
    dom.mouseButtons.textContent = '--';
}

function updateScreenInfo() {
    dom.screenResolution.textContent = `${window.screen.width} x ${window.screen.height}`;
    dom.screenColorDepth.textContent = `${window.screen.colorDepth}位`;
    
    const mediaQuery = window.matchMedia('(min-resolution: 96dpi)');
    dom.screenRefreshRate.textContent = '60Hz';
}

function handleKeyDown(e) {
    if (!isSensorsEnabled) return;
    
    dom.keyboardStatus.textContent = '活动';
    dom.keyboardKeys.textContent = e.key;
    
    if (e.getModifierState('CapsLock')) {
        dom.keyboardCaps.textContent = '开启';
    }
    
    sendSensorData('keyboard', {
        key: e.key,
        code: e.code,
        type: 'keydown',
        capsLock: e.getModifierState('CapsLock'),
        shiftKey: e.shiftKey,
        ctrlKey: e.ctrlKey,
        altKey: e.altKey
    });
}

function handleKeyUp(e) {
    if (!isSensorsEnabled) return;
    
    setTimeout(() => {
        const activeElement = document.activeElement;
        if (activeElement.tagName !== 'INPUT' && activeElement.tagName !== 'TEXTAREA') {
            dom.keyboardStatus.textContent = '空闲';
            dom.keyboardKeys.textContent = '无';
        }
    }, 1000);
    
    sendSensorData('keyboard', {
        key: e.key,
        code: e.code,
        type: 'keyup',
        capsLock: e.getModifierState('CapsLock')
    });
}

function handleMouseMove(e) {
    if (!isSensorsEnabled) return;
    
    dom.mousePosition.textContent = `${e.clientX}, ${e.clientY}`;
    
    sendSensorData('mouse', {
        x: e.clientX,
        y: e.clientY,
        screenX: e.screenX,
        screenY: e.screenY,
        type: 'mousemove'
    });
}

function handleMouseWheel(e) {
    if (!isSensorsEnabled) return;
    
    dom.mouseScroll.textContent = e.deltaY > 0 ? '向下' : '向上';
    
    sendSensorData('mouse', {
        deltaX: e.deltaX,
        deltaY: e.deltaY,
        type: 'wheel'
    });
}

function handleMouseDown(e) {
    if (!isSensorsEnabled) return;
    
    const buttons = [];
    if (e.button === 0) buttons.push('左键');
    if (e.button === 1) buttons.push('中键');
    if (e.button === 2) buttons.push('右键');
    dom.mouseButtons.textContent = buttons.join(', ');
    
    sendSensorData('mouse', {
        button: e.button,
        type: 'mousedown'
    });
}

function handleMouseUp(e) {
    if (!isSensorsEnabled) return;
    
    dom.mouseButtons.textContent = '--';
    
    sendSensorData('mouse', {
        button: e.button,
        type: 'mouseup'
    });
}

function sendSensorData(type, data) {
    if (sensorWebSocket && sensorWebSocket.readyState === WebSocket.OPEN) {
        sensorWebSocket.send(JSON.stringify({ type, data }));
    }
    
    fetch(`${API_BASE}/api/sensors/data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, data })
    }).catch((e) => { /* server unavailable */ });
}

async function loadSettings() {
    try {
        const response = await fetch(`${API_BASE}/api/settings`);
        const settings = await response.json();
        
        dom.settingInputDim.value = settings.input_dim || 16;
        dom.settingMaxSteps.value = settings.max_steps || 1000;
        dom.settingLogInterval.value = settings.log_interval || 20;
        dom.settingSaveInterval.value = settings.save_interval || 1000;
        dom.settingFeThreshold.value = settings.free_energy_threshold || 0.3;
        dom.settingFeThresholdValue.textContent = settings.free_energy_threshold || 0.3;
        dom.settingNoveltyThreshold.value = settings.novelty_threshold || 0.5;
        dom.settingNoveltyThresholdValue.textContent = settings.novelty_threshold || 0.5;
        dom.settingAutoStart.checked = settings.auto_start || false;
        dom.settingSensorEnabled.checked = settings.sensor_enabled || true;
        dom.settingVoiceInput.checked = settings.voice_input || false;
        dom.settingTheme.value = settings.theme || 'dark';
        
        document.body.className = settings.theme || 'dark';
    } catch (error) {
        console.error('Failed to load settings:', error);
    }
}

async function saveSettings() {
    const settings = {
        input_dim: parseInt(dom.settingInputDim.value),
        max_steps: parseInt(dom.settingMaxSteps.value) || 0,
        log_interval: parseInt(dom.settingLogInterval.value),
        save_interval: parseInt(dom.settingSaveInterval.value),
        free_energy_threshold: parseFloat(dom.settingFeThreshold.value),
        novelty_threshold: parseFloat(dom.settingNoveltyThreshold.value),
        auto_start: dom.settingAutoStart.checked,
        sensor_enabled: dom.settingSensorEnabled.checked,
        voice_input: dom.settingVoiceInput.checked,
        theme: dom.settingTheme.value
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        const data = await response.json();
        if (data.status === 'success') {
            addMessage('agent', '设置已保存');
        }
    } catch (error) {
        console.error('Failed to save settings:', error);
        addMessage('agent', '保存设置失败');
    }
}

async function loadPlugins() {
    try {
        const [availableRes, loadedRes] = await Promise.all([
            fetch(`${API_BASE}/api/plugins/available`),
            fetch(`${API_BASE}/api/plugins/loaded`)
        ]);
        const availableData = await availableRes.json();
        const loadedData = await loadedRes.json();

        const pluginList = document.getElementById('pluginList');
        if (!pluginList) return;

        const loadedMap = {};
        (loadedData.plugins || []).forEach(p => { loadedMap[p.name] = p; });

        const allPlugins = (availableData.plugins || []).map(p => ({
            ...p,
            ...loadedMap[p.name]
        }));

        Object.values(loadedMap).forEach(p => {
            if (!allPlugins.find(a => a.name === p.name)) {
                allPlugins.unshift(p);
            }
        });

        pluginList.innerHTML = allPlugins.map(plugin => {
            const status = plugin.status || 'unloaded';
            const isLoaded = status !== 'unloaded';
            const isActive = status === 'active';
            return `
                <div class="plugin-item">
                    <div class="plugin-item-header">
                        <span class="plugin-item-name">${plugin.name}</span>
                        <span class="plugin-item-status plugin-status-${status}">${status}</span>
                    </div>
                    <span class="plugin-item-desc">${plugin.description || plugin.type || ''}</span>
                    <div class="plugin-item-actions">
                        ${!isLoaded ? `<button onclick="pluginAction('load', '${plugin.name}')">加载</button>` : ''}
                        ${isLoaded && !isActive ? `<button onclick="pluginAction('activate', '${plugin.name}')">激活</button>` : ''}
                        ${isActive ? `<button onclick="pluginAction('deactivate', '${plugin.name}')">停用</button>` : ''}
                        ${isLoaded ? `<button onclick="pluginAction('unload', '${plugin.name}')">卸载</button>` : ''}
                        ${isLoaded ? `<button onclick="pluginAction('reload', '${plugin.name}')">重载</button>` : ''}
                    </div>
                </div>
            `;
        }).join('');

        if (allPlugins.length === 0) {
            pluginList.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">暂无可用插件</div>';
        }
    } catch (error) {
        console.error('Failed to load plugins:', error);
    }
}

async function pluginAction(action, pluginName) {
    const url = action === 'load'
        ? `${API_BASE}/api/plugins/load?plugin_name=${encodeURIComponent(pluginName)}`
        : `${API_BASE}/api/plugins/${encodeURIComponent(pluginName)}/${action}`;

    try {
        const response = await fetch(url, { method: 'POST' });
        const data = await response.json();
        if (data.success === false || data.error) {
            addMessage('agent', `插件${action}失败: ${data.error || data.detail || '未知错误'}`);
        }
        await loadPlugins();
    } catch (error) {
        console.error(`Plugin ${action} failed:`, error);
        addMessage('agent', `插件${action}失败: ${error.message}`);
    }
}

async function loadAllPlugins() {
    try {
        const response = await fetch(`${API_BASE}/api/plugins/load_all`, { method: 'POST' });
        const data = await response.json();
        await loadPlugins();
        addMessage('agent', '已加载所有可用插件');
    } catch (error) {
        console.error('Failed to load all plugins:', error);
        addMessage('agent', '加载所有插件失败');
    }
}

// ============ Skills 技能商店 ============

let skillsStoreData = [];
let skillsInstalledData = [];
let skillsCurrentTab = 'store';

async function initSkillsStore() {
    await loadSkillsStatus();
    await loadInstalledSkills();
}

async function loadSkillsStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/skills/status`);
        const data = await response.json();
        const info = document.getElementById('skillsStoreInfo');
        if (!info) return;
        if (data.available) {
            info.textContent = `SkillHub v${data.cli_version} | 已安装 ${data.installed_count}`;
            info.classList.add('connected');
        } else {
            info.textContent = 'SkillHub 未安装';
            info.classList.remove('connected');
        }
    } catch (error) {
        console.error('Failed to load skills status:', error);
    }
}

async function searchSkills() {
    const input = document.getElementById('skillSearchInput');
    if (!input) return;
    const query = input.value.trim();
    if (!query) {
        addMessage('agent', '请输入搜索关键词');
        return;
    }

    const storeList = document.getElementById('skillsStoreList');
    if (storeList) storeList.innerHTML = '<div class="skills-loading">搜索中...</div>';

    // 切换到商店标签
    switchSkillsTab('store');

    try {
        const response = await fetch(`${API_BASE}/api/skills/search?q=${encodeURIComponent(query)}&limit=20`);
        const data = await response.json();
        if (data.success) {
            skillsStoreData = data.results || [];
            renderSkillCards();
        } else {
            if (storeList) storeList.innerHTML = `<div class="skills-empty">${data.error || '搜索失败'}</div>`;
        }
    } catch (error) {
        console.error('Skills search failed:', error);
        if (storeList) storeList.innerHTML = '<div class="skills-empty">搜索失败，请检查网络连接</div>';
    }
}

function renderSkillCards() {
    const storeList = document.getElementById('skillsStoreList');
    if (!storeList) return;

    if (skillsStoreData.length === 0) {
        storeList.innerHTML = '<div class="skills-empty">暂无搜索结果，请在上方输入关键词搜索</div>';
        return;
    }

    const installedSlugs = new Set(skillsInstalledData.map(s => s.slug));

    storeList.innerHTML = skillsStoreData.map(skill => {
        const desc = (skill.description || '').split('\n')[0];
        const isInstalled = installedSlugs.has(skill.slug);
        const source = skill.source || 'community';
        return `
            <div class="skill-card">
                <div class="skill-card-header">
                    <span class="skill-card-name">${escapeHtml(skill.name)}</span>
                    <div class="skill-card-meta">
                        <span class="skill-card-version">v${escapeHtml(skill.version || '?')}</span>
                        <span class="skill-card-badge skill-badge-source">${escapeHtml(source)}</span>
                        ${isInstalled ? '<span class="skill-card-badge skill-badge-installed">已安装</span>' : ''}
                    </div>
                </div>
                <div class="skill-card-slug">${escapeHtml(skill.slug)}</div>
                <div class="skill-card-desc">${escapeHtml(desc)}</div>
                <div class="skill-card-actions">
                    ${isInstalled
                        ? `<button class="btn-uninstall" onclick="uninstallSkill('${escapeHtml(skill.slug)}')">卸载</button>`
                        : `<button class="btn-install" onclick="installSkill('${escapeHtml(skill.slug)}')">安装</button>`
                    }
                </div>
            </div>
        `;
    }).join('');
}

async function loadInstalledSkills() {
    try {
        const response = await fetch(`${API_BASE}/api/skills/installed`);
        const data = await response.json();
        skillsInstalledData = data.skills || [];
        renderInstalledSkills();
        loadSkillsStatus();
    } catch (error) {
        console.error('Failed to load installed skills:', error);
    }
}

function renderInstalledSkills() {
    const list = document.getElementById('skillsInstalledList');
    if (!list) return;

    if (skillsInstalledData.length === 0) {
        list.innerHTML = '<div class="skills-empty">暂无已安装的技能，前往商店搜索并安装</div>';
        return;
    }

    list.innerHTML = skillsInstalledData.map(skill => {
        const desc = (skill.description || '').split('\n')[0];
        return `
            <div class="skill-card">
                <div class="skill-card-header">
                    <span class="skill-card-name">${escapeHtml(skill.name)}</span>
                    <div class="skill-card-meta">
                        ${skill.version ? `<span class="skill-card-version">v${escapeHtml(skill.version)}</span>` : ''}
                        <span class="skill-card-badge skill-badge-installed">已安装</span>
                    </div>
                </div>
                <div class="skill-card-slug">${escapeHtml(skill.slug)}</div>
                <div class="skill-card-desc">${escapeHtml(desc)}</div>
                <div class="skill-card-actions">
                    <button class="btn-uninstall" onclick="uninstallSkill('${escapeHtml(skill.slug)}')">卸载</button>
                </div>
            </div>
        `;
    }).join('');
}

function switchSkillsTab(tab) {
    skillsCurrentTab = tab;
    document.querySelectorAll('.skills-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tab);
    });
    const storeList = document.getElementById('skillsStoreList');
    const installedList = document.getElementById('skillsInstalledList');
    if (storeList) storeList.style.display = tab === 'store' ? 'flex' : 'none';
    if (installedList) installedList.style.display = tab === 'installed' ? 'flex' : 'none';
    if (tab === 'installed') loadInstalledSkills();
}

async function installSkill(slug) {
    addMessage('agent', `正在安装技能: ${slug}...`);
    try {
        const response = await fetch(`${API_BASE}/api/skills/install?slug=${encodeURIComponent(slug)}`, { method: 'POST' });
        const data = await response.json();
        if (response.ok && data.success !== false) {
            addMessage('agent', `技能 ${slug} 安装成功！`);
            await loadInstalledSkills();
            renderSkillCards();
        } else {
            const err = data.detail || data.error || '安装失败';
            addMessage('agent', `技能 ${slug} 安装失败: ${err}`);
        }
    } catch (error) {
        console.error('Install skill failed:', error);
        addMessage('agent', `技能 ${slug} 安装失败: ${error.message}`);
    }
}

async function uninstallSkill(slug) {
    addMessage('agent', `正在卸载技能: ${slug}...`);
    try {
        const response = await fetch(`${API_BASE}/api/skills/${encodeURIComponent(slug)}`, { method: 'DELETE' });
        const data = await response.json();
        if (response.ok && data.success !== false) {
            addMessage('agent', `技能 ${slug} 已卸载`);
            await loadInstalledSkills();
            renderSkillCards();
        } else {
            const err = data.detail || data.error || '卸载失败';
            addMessage('agent', `技能 ${slug} 卸载失败: ${err}`);
        }
    } catch (error) {
        console.error('Uninstall skill failed:', error);
        addMessage('agent', `技能 ${slug} 卸载失败: ${error.message}`);
    }
}

function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function startMentalArchitectureRefresh() {
    if (mentalArchitectureRefreshInterval) {
        clearInterval(mentalArchitectureRefreshInterval);
    }
    mentalArchitectureRefreshInterval = setInterval(refreshMentalArchitectureData, 3000);
    refreshMentalArchitectureData();
}

function stopMentalArchitectureRefresh() {
    if (mentalArchitectureRefreshInterval) {
        clearInterval(mentalArchitectureRefreshInterval);
        mentalArchitectureRefreshInterval = null;
    }
}

async function refreshMentalArchitectureData() {
    try {
        const [archRes, actionRes, evolutionRes, securityRes] = await Promise.all([
            fetch(`${API_BASE}/api/mental-architecture/status`),
            fetch(`${API_BASE}/api/autonomous-action/status`),
            fetch(`${API_BASE}/api/evolution/status`),
            fetch(`${API_BASE}/api/security/status`)
        ]);
        
        const archData = await archRes.json();
        const actionData = await actionRes.json();
        const evolutionData = await evolutionRes.json();
        const securityData = await securityRes.json();
        
        updateMentalArchitecture(archData);
        updateAutonomousAction(actionData);
        updateEvolution(evolutionData);
        updateSecurity(securityData);
    } catch (error) {
        console.warn('Mental architecture refresh failed:', error.message || error);
    }
}

function updateMentalArchitecture(data) {
    if (!data) return;
    
    const reflex = data.reflex || {};
    const spiking = reflex.spiking_activity || {};
    const reflexRules = reflex.rule_stats || {};
    const reflexInstinct = reflex.instinct_stats || {};
    
    if (dom.reflexStatus) dom.reflexStatus.textContent = reflex.is_active ? '活跃' : '休眠';
    if (dom.reflexSpikeRate) dom.reflexSpikeRate.textContent = (spiking.overall_spike_rate * 100).toFixed(1) + '%';
    
    let ruleMatch = 0;
    Object.values(reflexRules).forEach(r => { ruleMatch += r.usage_count || 0; });
    if (dom.reflexRuleMatch) dom.reflexRuleMatch.textContent = ruleMatch;
    
    if (dom.reflexResponse) dom.reflexResponse.textContent = reflex.response_history_count || 0;
    if (dom.reflexTotalRules) dom.reflexTotalRules.textContent = reflex.total_rules || 0;
    if (dom.reflexTotalPatterns) dom.reflexTotalPatterns.textContent = reflex.total_patterns || 0;
    if (dom.reflexAvgConfidence) dom.reflexAvgConfidence.textContent = (reflex.recent_avg_confidence || 0).toFixed(4);
    if (dom.reflexDelegateRate) dom.reflexDelegateRate.textContent = (reflex.delegate_rate || 0).toFixed(2);
    
    const deliberative = data.deliberative || {};
    const thinking = deliberative.thinking_engine_stats || {};
    
    if (dom.deliberativeStatus) dom.deliberativeStatus.textContent = thinking.completed_cycles > 0 ? '活跃' : '等待';
    if (dom.delibCycles) dom.delibCycles.textContent = thinking.total_thinking_cycles || 0;
    if (dom.delibCompleted) dom.delibCompleted.textContent = thinking.completed_cycles || 0;
    if (dom.delibConfidence) dom.delibConfidence.textContent = (thinking.avg_confidence || 0).toFixed(4);
    if (dom.delibSolutions) dom.delibSolutions.textContent = (thinking.optimizer_stats || {}).solutions_count || 0;
    if (dom.delibPhase) dom.delibPhase.textContent = deliberative.current_phase || '--';
    if (dom.delibMode) dom.delibMode.textContent = deliberative.mode || '--';
    if (dom.delibSystem1) dom.delibSystem1.textContent = deliberative.system1_usage || 0;
    if (dom.delibSystem2) dom.delibSystem2.textContent = deliberative.system2_usage || 0;
    if (dom.delibSystem1Ratio) dom.delibSystem1Ratio.textContent = (deliberative.system1_ratio || 0).toFixed(2);
    
    const meta = data.meta_cognitive || {};
    const guardian = meta.boundary_guardian || {};
    const regulator = meta.strategy_regulator || {};
    const learning = meta.meta_learning || {};
    const selfModel = meta.self_model_summary || {};
    const identity = selfModel.identity || {};
    const capabilities = selfModel.capabilities || {};
    const state = selfModel.state || {};
    const strategy = meta.strategy_regulator_summary || {};
    
    if (dom.metaStatus) dom.metaStatus.textContent = (guardian.safety_violations || 0) > 0 ? '告警' : '正常';
    if (dom.metaViolations) dom.metaViolations.textContent = guardian.safety_violations || 0;
    if (dom.metaStrategyChanges) dom.metaStrategyChanges.textContent = regulator.strategy_changes || strategy.strategy_changes || 0;
    if (dom.metaLearningCycles) dom.metaLearningCycles.textContent = learning.learning_cycles || 0;
    
    if (dom.metaSelfName) dom.metaSelfName.textContent = identity.name || '--';
    if (dom.metaSelfRole) dom.metaSelfRole.textContent = identity.role || '--';
    if (dom.metaHealthScore) dom.metaHealthScore.textContent = (state.health_score || 0).toFixed(2);
    if (dom.metaCompLoad) dom.metaCompLoad.textContent = (state.computational_load || 0).toFixed(2);
    if (dom.metaMemoryUsage) dom.metaMemoryUsage.textContent = (state.memory_usage || 0).toFixed(2);
    if (dom.metaThinkingStrategy) dom.metaThinkingStrategy.textContent = strategy.current_thinking_strategy || '--';
    if (dom.metaActionStrategy) dom.metaActionStrategy.textContent = strategy.current_action_strategy || '--';
    if (dom.metaStrengths) dom.metaStrengths.textContent = (capabilities.strengths || []).join(', ') || '--';
    if (dom.metaWeaknesses) dom.metaWeaknesses.textContent = (capabilities.weaknesses || []).join(', ') || '--';
    if (dom.metaAvgSuccessRate) dom.metaAvgSuccessRate.textContent = (capabilities.avg_success_rate || 0).toFixed(2);
}

function updateAutonomousAction(data) {
    if (!data) return;
    
    const decomp = data.decomposition || {};
    const planning = data.path_planning || {};
    const execution = data.execution || {};
    const exploration = data.exploration || {};
    
    if (dom.actionDecomposition) dom.actionDecomposition.textContent = decomp.total_decompositions || 0;
    if (dom.actionPathPlanning) dom.actionPathPlanning.textContent = planning.total_paths || 0;
    if (dom.actionExecutions) dom.actionExecutions.textContent = execution.total_executions || 0;
    if (dom.actionSuccessRate) dom.actionSuccessRate.textContent = ((execution.success_rate || 0) * 100).toFixed(0) + '%';
    if (dom.actionActiveGoals) dom.actionActiveGoals.textContent = data.active_goals || 0;
    if (dom.actionExploration) dom.actionExploration.textContent = exploration.total_actions || 0;
}

function updateEvolution(data) {
    if (!data) return;
    
    const quad = data.quad_level || {};
    
    if (dom.microReinforcements) dom.microReinforcements.textContent = quad.micro_reinforcements || 0;
    if (dom.mesoRuleUpdates) dom.mesoRuleUpdates.textContent = quad.meso_rule_updates || 0;
    if (dom.mesoSkillGenerations) dom.mesoSkillGenerations.textContent = quad.meso_skill_generations || 0;
    if (dom.macroOptimizations) dom.macroOptimizations.textContent = quad.macro_optimizations || 0;
    if (dom.metaOptimizations) dom.metaOptimizations.textContent = quad.meta_optimizations || 0;
}

function updateSecurity(data) {
    if (!data) return;
    
    if (dom.securityRules) dom.securityRules.textContent = data.hard_boundary_rules || 0;
    if (dom.securityViolations) dom.securityViolations.textContent = data.safety_violations || 0;
    if (dom.securityRiskLevel) dom.securityRiskLevel.textContent = data.risk_level || '--';
    if (dom.securityCircuitBreaker) dom.securityCircuitBreaker.textContent = data.circuit_breaker_status || '--';
}

let draggedPanel = null;

function initPanelDragAndDrop() {
    const panels = document.querySelectorAll('.panel[draggable="true"]');
    
    panels.forEach(panel => {
        panel.addEventListener('dragstart', handleDragStart);
        panel.addEventListener('dragover', handleDragOver);
        panel.addEventListener('dragleave', handleDragLeave);
        panel.addEventListener('drop', handleDrop);
    });
    
    const moveButtons = document.querySelectorAll('.panel-move-btn');
    moveButtons.forEach(btn => {
        btn.addEventListener('click', handleMoveButtonClick);
    });
    
    updateMoveButtonStates();
}

function handleDragStart(e) {
    draggedPanel = e.target.closest('.panel');
    draggedPanel.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', draggedPanel.id);
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    
    const targetPanel = e.target.closest('.panel');
    if (targetPanel && targetPanel !== draggedPanel) {
        targetPanel.classList.add('drag-over');
    }
}

function handleDragLeave(e) {
    const targetPanel = e.target.closest('.panel');
    if (targetPanel) {
        targetPanel.classList.remove('drag-over');
    }
}

function handleDrop(e) {
    e.preventDefault();
    
    const targetPanel = e.target.closest('.panel');
    if (targetPanel && targetPanel !== draggedPanel) {
        movePanel(draggedPanel, targetPanel);
    }
    
    document.querySelectorAll('.panel').forEach(p => {
        p.classList.remove('dragging', 'drag-over');
    });
    
    draggedPanel = null;
    updateMoveButtonStates();
}

function handleMoveButtonClick(e) {
    const btn = e.target;
    const panelId = btn.dataset.panel;
    const direction = btn.dataset.direction;
    
    const panel = document.getElementById(panelId);
    if (!panel) return;
    
    const parent = panel.parentElement;
    const siblings = Array.from(parent.children).filter(c => c.classList.contains('panel'));
    const currentIndex = siblings.indexOf(panel);
    
    if (direction === 'up' && currentIndex > 0) {
        movePanel(panel, siblings[currentIndex - 1]);
    } else if (direction === 'down' && currentIndex < siblings.length - 1) {
        movePanel(panel, siblings[currentIndex + 1]);
    }
    
    updateMoveButtonStates();
}

function movePanel(fromPanel, toPanel) {
    const parent = fromPanel.parentElement;
    const fromIndex = Array.from(parent.children).indexOf(fromPanel);
    const toIndex = Array.from(parent.children).indexOf(toPanel);
    
    if (fromIndex < toIndex) {
        parent.insertBefore(fromPanel, toPanel.nextSibling);
    } else {
        parent.insertBefore(fromPanel, toPanel);
    }
    
    fromPanel.classList.add('panel-moved');
    setTimeout(() => {
        fromPanel.classList.remove('panel-moved');
    }, 300);
}

function updateMoveButtonStates() {
    const containers = document.querySelectorAll('.left-panel, .middle-panel, .right-panel');
    
    containers.forEach(container => {
        const panels = Array.from(container.children).filter(c => c.classList.contains('panel'));
        
        panels.forEach((panel, index) => {
            const upBtn = panel.querySelector('[data-direction="up"]');
            const downBtn = panel.querySelector('[data-direction="down"]');
            
            if (upBtn) upBtn.disabled = index === 0;
            if (downBtn) downBtn.disabled = index === panels.length - 1;
        });
    });
}

function startOnlineAgentsRefresh() {
    if (onlineAgentsRefreshInterval) {
        clearInterval(onlineAgentsRefreshInterval);
    }
    onlineAgentsRefreshInterval = setInterval(refreshOnlineAgents, 5000);
    refreshOnlineAgents();
}

function stopOnlineAgentsRefresh() {
    if (onlineAgentsRefreshInterval) {
        clearInterval(onlineAgentsRefreshInterval);
        onlineAgentsRefreshInterval = null;
    }
}

async function refreshOnlineAgents() {
    if (!serverAvailable) return;
    try {
        const [onlineRes, swarmRes] = await Promise.all([
            fetch(`${API_BASE}/api/chat/online`),
            fetch(`${API_BASE}/api/swarm/agents`)
        ]);
        
        const onlineData = await onlineRes.json();
        const swarmData = await swarmRes.json();
        
        const onlineIds = new Set(onlineData.online || []);
        const agents = swarmData.agents || [];
        
        const onlineAgents = agents.filter(a => onlineIds.has(a.agent_id || a.id));
        
        if (dom.onlineAgentsList) {
            if (onlineAgents.length === 0) {
                dom.onlineAgentsList.innerHTML = '<span style="color: var(--text-muted);">暂无在线智能体</span>';
            } else {
                dom.onlineAgentsList.innerHTML = onlineAgents.map(a => 
                    `<span class="online-agent-tag">${escapeHtml(a.name || a.agent_id)}</span>`
                ).join('');
            }
        }
    } catch (error) {
        console.warn('Online agents refresh failed:', error.message || error);
    }
}

async function switchChannel(channelId) {
    currentChannel = channelId;
    
    document.querySelectorAll('.channel-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.channel === channelId);
    });
    
    await loadChannelMessages(channelId);
}

async function loadChannelMessages(channelId) {
    if (!serverAvailable) return;
    try {
        const response = await fetch(`${API_BASE}/api/channels/${channelId}/messages?limit=50`);
        const data = await response.json();
        
        if (data.messages && data.messages.length > 0) {
            channelMessages[channelId] = data.messages;
            renderChannelMessages(data.messages);
        } else {
            dom.chatContainer.innerHTML = '<div class="message agent-message"><div class="message-content"><span class="agent-icon">🤖</span><p>欢迎来到 ' + (channelId === 'general' ? '综合讨论' : channelId === 'tasks' ? '任务协作' : '自由交流') + ' 频道！</p></div></div>';
        }
    } catch (error) {
        console.warn('Channel messages load failed:', error.message || error);
    }
}

function renderChannelMessages(messages) {
    dom.chatContainer.innerHTML = messages.map(msg => {
        const isSystem = msg.message_type === 'system';
        const isUser = msg.sender_id === 'user';
        const isAgent = !isSystem && !isUser;
        
        let icon = '';
        let senderName = msg.sender_id;
        
        if (isSystem) {
            icon = '<span class="agent-icon">📢</span>';
            senderName = '系统';
        } else if (isAgent) {
            icon = '<span class="agent-icon">🤖</span>';
        }
        
        return `
            <div class="message ${isUser ? 'user-message' : 'agent-message'}">
                <div class="message-content">
                    ${icon}
                    ${!isSystem ? `<span class="message-sender">${escapeHtml(senderName)}:</span>` : ''}
                    <p>${escapeHtml(msg.content)}</p>
                </div>
            </div>
        `;
    }).join('');
    
    dom.chatContainer.scrollTop = dom.chatContainer.scrollHeight;
}

async function sendMessageToChannel(content) {
    if (!content.trim()) return;
    
    addMessage('user', content);
    dom.chatInput.value = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/channels/${currentChannel}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sender_id: 'user',
                content: content,
                message_type: 'text'
            })
        });
        
        if (response.ok) {
            await loadChannelMessages(currentChannel);
        }
    } catch (error) {
        console.error('Failed to send channel message:', error);
        addMessage('agent', '发送消息失败，请检查网络连接');
    }
}

// ============ 自我意识 API ============

async function refreshSelfAwareness() {
    try {
        const response = await fetch(`${API_BASE}/api/self-awareness/status`);
        const data = await response.json();
        updateSelfAwareness(data);
    } catch (error) {
        console.warn('Self-awareness refresh failed:', error);
    }
}

function updateSelfAwareness(data) {
    if (!data) return;
    
    const metrics = data.metrics || {};
    if (dom.selfSelfRecognition) dom.selfSelfRecognition.textContent = (metrics.self_recognition || 0).toFixed(2);
    if (dom.selfCapabilityAwareness) dom.selfCapabilityAwareness.textContent = (metrics.capability_awareness || 0).toFixed(2);
    if (dom.selfLimitationAwareness) dom.selfLimitationAwareness.textContent = (metrics.limitation_awareness || 0).toFixed(2);
    if (dom.selfExistenceAwareness) dom.selfExistenceAwareness.textContent = (metrics.existence_awareness || 0).toFixed(2);
    if (dom.selfTemporalContinuity) dom.selfTemporalContinuity.textContent = (metrics.temporal_continuity || 0).toFixed(2);
    
    if (dom.barSelfRecognition) dom.barSelfRecognition.style.width = `${(metrics.self_recognition || 0) * 100}%`;
    if (dom.barCapabilityAwareness) dom.barCapabilityAwareness.style.width = `${(metrics.capability_awareness || 0) * 100}%`;
    if (dom.barLimitationAwareness) dom.barLimitationAwareness.style.width = `${(metrics.limitation_awareness || 0) * 100}%`;
    if (dom.barExistenceAwareness) dom.barExistenceAwareness.style.width = `${(metrics.existence_awareness || 0) * 100}%`;
    if (dom.barTemporalContinuity) dom.barTemporalContinuity.style.width = `${(metrics.temporal_continuity || 0) * 100}%`;
    
    const identity = data.identity || {};
    if (dom.identityName) dom.identityName.textContent = identity.name || '--';
    if (dom.identityRole) dom.identityRole.textContent = identity.role || '--';
    if (dom.identityGoals) dom.identityGoals.textContent = (identity.goals || []).join(', ') || '--';
    
    const introspection = data.introspection || [];
    if (dom.introspectionHistory) {
        dom.introspectionHistory.innerHTML = introspection.slice(0, 5).map(item => 
            `<div class="introspection-history-item">${item.timestamp || ''}: ${item.content || ''}</div>`
        ).join('');
    }
}

// ============ 思考 API ============

async function decomposeProblem() {
    const problem = dom.problemInput?.value?.trim();
    if (!problem) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/thinking/decompose`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ problem })
        });
        const data = await response.json();
        
        if (dom.decomposeResult) {
            dom.decomposeResult.innerHTML = `
                <h4>问题分解结果</h4>
                <ul>${(data.subproblems || []).map((p, i) => `<li>${i + 1}. ${p}</li>`).join('')}</ul>
            `;
        }
    } catch (error) {
        console.error('Problem decomposition failed:', error);
        if (dom.decomposeResult) dom.decomposeResult.innerHTML = '<div>分解失败，请重试</div>';
    }
}

async function criticalAnalyze() {
    const input = dom.criticalInput?.value?.trim();
    if (!input) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/thinking/critical`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input })
        });
        const data = await response.json();
        
        if (dom.criticalResult) {
            dom.criticalResult.innerHTML = `
                <h4>批判性分析结果</h4>
                <ul>
                    ${data.strengths ? `<li><strong>优点:</strong> ${data.strengths}</li>` : ''}
                    ${data.weaknesses ? `<li><strong>缺点:</strong> ${data.weaknesses}</li>` : ''}
                    ${data.biases ? `<li><strong>偏见:</strong> ${data.biases}</li>` : ''}
                    ${data.improvements ? `<li><strong>改进建议:</strong> ${data.improvements}</li>` : ''}
                </ul>
            `;
        }
    } catch (error) {
        console.error('Critical analysis failed:', error);
        if (dom.criticalResult) dom.criticalResult.innerHTML = '<div>分析失败，请重试</div>';
    }
}

// ============ 决策 API ============

async function simulateDecision() {
    const goal = dom.simGoal?.value || '测试决策';
    const confidence = parseFloat(dom.simConfidence?.value) || 0.7;
    const riskLevel = dom.simRiskLevel?.value || 'medium';
    
    try {
        const response = await fetch(`${API_BASE}/api/decision/make`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ goal, confidence, risk_level: riskLevel })
        });
        const data = await response.json();
        
        if (dom.decisionResult) {
            dom.decisionResult.innerHTML = `
                <div class="decision-outcome">${data.outcome || '--'}</div>
                <div><strong>置信度:</strong> ${(data.confidence || 0).toFixed(2)}</div>
                <div><strong>风险等级:</strong> ${data.risk_level || '--'}</div>
                <div><strong>预期收益:</strong> ${(data.expected_utility || 0).toFixed(2)}</div>
                <div><strong>理由:</strong> ${data.reasoning || '--'}</div>
            `;
        }
    } catch (error) {
        console.error('Decision simulation failed:', error);
        if (dom.decisionResult) dom.decisionResult.innerHTML = '<div>决策失败，请重试</div>';
    }
}

// ============ 个性 API ============

async function refreshPersonality() {
    try {
        const response = await fetch(`${API_BASE}/api/personality/status`);
        const data = await response.json();
        updatePersonality(data);
    } catch (error) {
        console.warn('Personality refresh failed:', error);
    }
}

function updatePersonality(data) {
    if (!data) return;
    
    if (dom.personalitySignature) dom.personalitySignature.textContent = data.signature || '--';
    if (dom.personalityConsistency) dom.personalityConsistency.textContent = (data.consistency || 0).toFixed(3);
    
    const traits = data.traits || {};
    if (dom.barTraitCuriosity) dom.barTraitCuriosity.style.width = `${(traits.curiosity || 0) * 100}%`;
    if (dom.barTraitAssertiveness) dom.barTraitAssertiveness.style.width = `${(traits.assertiveness || 0) * 100}%`;
    if (dom.barTraitCautiousness) dom.barTraitCautiousness.style.width = `${(traits.cautiousness || 0) * 100}%`;
    if (dom.barTraitCreativity) dom.barTraitCreativity.style.width = `${(traits.creativity || 0) * 100}%`;
    if (dom.barTraitPatience) dom.barTraitPatience.style.width = `${(traits.patience || 0) * 100}%`;
    
    const values = data.values || {};
    if (dom.barValueSurvival) dom.barValueSurvival.style.width = `${(values.survival || 0) * 100}%`;
    if (dom.barValueKnowledge) dom.barValueKnowledge.style.width = `${(values.knowledge || 0) * 100}%`;
    if (dom.barValueGrowth) dom.barValueGrowth.style.width = `${(values.growth || 0) * 100}%`;
}

// ============ 记忆系统 API ============

async function refreshMemory() {
    try {
        const response = await fetch(`${API_BASE}/api/memory/status`);
        const data = await response.json();
        updateMemory(data);
    } catch (error) {
        console.warn('Memory refresh failed:', error);
    }
}

function updateMemory(data) {
    if (!data) return;
    
    const layers = data.layers || {};
    if (dom.memL1Count) dom.memL1Count.textContent = layers.l1_count || 0;
    if (dom.memL2Count) dom.memL2Count.textContent = layers.l2_count || 0;
    if (dom.memL3Count) dom.memL3Count.textContent = layers.l3_count || 0;
    if (dom.memL4Count) dom.memL4Count.textContent = layers.l4_count || 0;
    
    if (dom.memBarL1) dom.memBarL1.style.width = `${Math.min(100, (layers.l1_usage || 0) * 100)}%`;
    if (dom.memBarL2) dom.memBarL2.style.width = `${Math.min(100, (layers.l2_usage || 0) * 100)}%`;
    if (dom.memBarL3) dom.memBarL3.style.width = `${Math.min(100, (layers.l3_usage || 0) * 100)}%`;
    if (dom.memBarL4) dom.memBarL4.style.width = `${Math.min(100, (layers.l4_usage || 0) * 100)}%`;
    
    if (dom.memTotalCapacity) dom.memTotalCapacity.textContent = data.total_capacity || '--';
    if (dom.memUsed) dom.memUsed.textContent = data.used || '--';
    if (dom.memUtilization) dom.memUtilization.textContent = `${(data.utilization || 0) * 100}%`;
    
    const recent = data.recent || [];
    if (dom.memoryRecentList) {
        dom.memoryRecentList.innerHTML = recent.slice(0, 5).map(item => 
            `<div class="memory-recent-item">${item.content || ''}<span class="memory-recent-time">${item.timestamp || ''}</span></div>`
        ).join('');
    }
}

async function clearMemory() {
    try {
        const response = await fetch(`${API_BASE}/api/memory/clear`, { method: 'POST' });
        const data = await response.json();
        if (data.status === 'success') {
            addMessage('agent', '记忆已清空');
            refreshMemory();
        }
    } catch (error) {
        console.error('Clear memory failed:', error);
    }
}

// ============ 知识图谱 API ============

async function refreshKnowledge() {
    try {
        const response = await fetch(`${API_BASE}/api/knowledge/status`);
        const data = await response.json();
        updateKnowledge(data);
    } catch (error) {
        console.warn('Knowledge refresh failed:', error);
    }
}

function updateKnowledge(data) {
    if (!data) return;
    
    if (dom.kgNodeCount) dom.kgNodeCount.textContent = data.node_count || 0;
    if (dom.kgEdgeCount) dom.kgEdgeCount.textContent = data.edge_count || 0;
    if (dom.kgClusterCount) dom.kgClusterCount.textContent = data.cluster_count || 0;
    if (dom.kgActivity) dom.kgActivity.textContent = `${(data.activity || 0) * 100}%`;
    
    const recent = data.recent || [];
    if (dom.knowledgeRecentList) {
        dom.knowledgeRecentList.innerHTML = recent.slice(0, 5).map(item => 
            `<div class="knowledge-recent-item">${item || ''}</div>`
        ).join('');
    }
}

// ============ 学习进度 API ============

async function refreshLearning() {
    try {
        const response = await fetch(`${API_BASE}/api/learning/status`);
        const data = await response.json();
        updateLearning(data);
    } catch (error) {
        console.warn('Learning refresh failed:', error);
    }
}

function updateLearning(data) {
    if (!data) return;
    
    const progress = data.progress || {};
    const total = Math.min(100, (progress.total || 0) * 100);
    if (dom.learningPercentage) dom.learningPercentage.textContent = `${Math.round(total)}%`;
    
    if (dom.learningCycles) dom.learningCycles.textContent = data.cycles || 0;
    if (dom.learningKnowledge) dom.learningKnowledge.textContent = data.knowledge || '--';
    if (dom.learningSkills) dom.learningSkills.textContent = data.skills || '--';
    
    const rlProgress = Math.min(100, (progress.reinforcement || 0) * 100);
    const evoProgress = Math.min(100, (progress.evolution || 0) * 100);
    const metaProgress = Math.min(100, (progress.meta_learning || 0) * 100);
    
    if (dom.learningRlProgress) dom.learningRlProgress.textContent = `${Math.round(rlProgress)}%`;
    if (dom.learningEvoProgress) dom.learningEvoProgress.textContent = `${Math.round(evoProgress)}%`;
    if (dom.learningMetaProgress) dom.learningMetaProgress.textContent = `${Math.round(metaProgress)}%`;
    
    if (dom.barLearningRl) dom.barLearningRl.style.width = `${rlProgress}%`;
    if (dom.barLearningEvo) dom.barLearningEvo.style.width = `${evoProgress}%`;
    if (dom.barLearningMeta) dom.barLearningMeta.style.width = `${metaProgress}%`;
}

// ============ 日志系统 ============

let allLogs = [];

async function loadLogs() {
    try {
        const response = await fetch(`${API_BASE}/api/logs`);
        const data = await response.json();
        allLogs = data.logs || [];
        renderLogs();
    } catch (error) {
        console.warn('Logs load failed:', error);
    }
}

function renderLogs() {
    if (!dom.logsList) return;
    
    const filter = dom.logLevelFilter?.value || 'all';
    const filteredLogs = filter === 'all' 
        ? allLogs 
        : allLogs.filter(log => log.level === filter);
    
    dom.logsList.innerHTML = filteredLogs.slice(-20).reverse().map(log => `
        <div class="log-item">
            <span class="log-time">${log.timestamp || ''}</span>
            <span class="log-level ${log.level || 'info'}">${log.level || 'INFO'}</span>
            <span class="log-message">${escapeHtml(log.message || '')}</span>
        </div>
    `).join('');
    
    dom.logsList.scrollTop = 0;
}

function filterLogs() {
    renderLogs();
}

function clearLogs() {
    if (dom.logsList) {
        dom.logsList.innerHTML = '';
        allLogs = [];
    }
}

// ============ 更新主刷新函数 ============

function startMentalArchitectureRefresh() {
    if (mentalArchitectureRefreshInterval) {
        clearInterval(mentalArchitectureRefreshInterval);
    }
    mentalArchitectureRefreshInterval = setInterval(() => {
        refreshMentalArchitectureData();
        refreshSelfAwareness();
        refreshPersonality();
        refreshMemory();
        refreshKnowledge();
        refreshLearning();
        loadLogs();
        refreshFileStats();
    }, 3000);
    refreshMentalArchitectureData();
    refreshSelfAwareness();
    refreshPersonality();
    refreshMemory();
    refreshKnowledge();
    refreshLearning();
    loadLogs();
    refreshFileStats();
}

// ============ 文件摄入系统 API ============

async function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    
    setFileStatus('info', `正在上传 ${files.length} 个文件...`);
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`${API_BASE}/api/file-ingestion/upload`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                setFileStatus('success', `文件 ${file.name} 摄入成功！`);
            } else {
                setFileStatus('error', `文件 ${file.name} 摄入失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            setFileStatus('error', `文件 ${file.name} 上传失败: ${error.message}`);
        }
    }
    
    await refreshFileStats();
}

async function searchFiles() {
    const query = dom.fileSearchInput?.value?.trim();
    if (!query) return;
    
    const searchType = dom.fileSearchType?.value || 'content';
    
    try {
        const response = await fetch(`${API_BASE}/api/file-ingestion/search?query=${encodeURIComponent(query)}&search_type=${searchType}`);
        const data = await response.json();
        
        renderFileSearchResults(data.results || []);
    } catch (error) {
        console.error('File search failed:', error);
    }
}

function renderFileSearchResults(results) {
    if (!dom.fileIngestionList) return;
    
    if (results.length === 0) {
        dom.fileIngestionList.innerHTML = '<div class="file-item"><div class="file-item-info">未找到匹配的文件</div></div>';
        return;
    }
    
    dom.fileIngestionList.innerHTML = results.map(result => `
        <div class="file-item">
            <div class="file-item-info">
                <div class="file-item-name">${escapeHtml(result.content || '')}</div>
                <div class="file-item-meta">类型: ${result.content_type || '--'} | 相似度: ${(result.score || 0).toFixed(2)}</div>
            </div>
            <div class="file-item-actions">
                <button onclick="viewFileRecord('${result.record_id}')">查看</button>
            </div>
        </div>
    `).join('');
}

async function viewFileRecord(recordId) {
    try {
        const response = await fetch(`${API_BASE}/api/file-ingestion/record/${recordId}`);
        const data = await response.json();
        
        if (response.ok) {
            setFileStatus('info', `记录 ${recordId}: ${data.content_type}`);
        }
    } catch (error) {
        console.error('View record failed:', error);
    }
}

async function deleteFileRecord(recordId) {
    if (!confirm('确定要删除这个记录吗？')) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/file-ingestion/record/${recordId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            setFileStatus('success', '记录已删除');
            await refreshFileStats();
        }
    } catch (error) {
        setFileStatus('error', '删除失败');
    }
}

async function refreshFileStats() {
    try {
        const response = await fetch(`${API_BASE}/api/file-ingestion/stats`);
        const data = await response.json();
        
        if (data.status === 'not_initialized') {
            return;
        }
        
        const storage = data.storage || {};
        if (dom.fileTotalRecords) {
            dom.fileTotalRecords.textContent = storage.total_records || 0;
        }
        
        if (dom.fileTypeDistribution) {
            const dist = storage.type_distribution || {};
            const distStr = Object.entries(dist).map(([k, v]) => `${k}: ${v}`).join(', ') || '--';
            dom.fileTypeDistribution.textContent = distStr;
        }
        
        renderFileList(storage.total_records || 0);
    } catch (error) {
        console.warn('File stats refresh failed:', error);
    }
}

function renderFileList(count) {
    if (!dom.fileIngestionList) return;
    
    if (count === 0) {
        dom.fileIngestionList.innerHTML = '<div class="file-item"><div class="file-item-info">暂无摄入文件</div></div>';
        return;
    }
}

function setFileStatus(type, message) {
    if (!dom.fileStatusMessage) return;
    
    dom.fileStatusMessage.className = `status-message ${type}`;
    dom.fileStatusMessage.textContent = message;
    
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            dom.fileStatusMessage.className = 'status-message';
        }, 3000);
    }
}

async function init() {
    await I18N.init();
    await loadSettings();
    await checkAgentStatus();
    setInterval(checkAgentStatus, 5000);
    initEventListeners();
    initSensors();
    initWebSocket();
    initSkillsStore();
    initPanelDragAndDrop();
    initLanguageSelector();

    if (isAgentRunning) {
        updateAgentStatus(true);
        startMentalArchitectureRefresh();
    }
    
    startOnlineAgentsRefresh();

    setInterval(() => {
        if (currentChannel) {
            loadChannelMessages(currentChannel);
        }
    }, 3000);

    updateUIWithCurrentLang();
}

function initLanguageSelector() {
    const selector = document.getElementById('languageSelector');
    if (selector) {
        selector.value = I18N.getCurrentLang();
        selector.addEventListener('change', async (e) => {
            await I18N.loadLang(e.target.value);
            updateUIWithCurrentLang();
        });
    }
}

function updateUIWithCurrentLang() {
    const statusText = isAgentRunning ? I18N.get('app.status.online') : I18N.get('app.status.offline');
    const statusSpan = dom.agentStatus?.querySelector('span:last-child');
    if (statusSpan) statusSpan.textContent = statusText;

    const startBtn = document.getElementById('startAgentBtn');
    if (startBtn) startBtn.textContent = I18N.get('app.startBtn');

    const stopBtn = document.getElementById('stopAgentBtn');
    if (stopBtn) stopBtn.textContent = I18N.get('app.stopBtn');

    const chatInput = document.getElementById('chatInput');
    if (chatInput) chatInput.placeholder = I18N.get('chat.inputPlaceholder');

    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) sendBtn.textContent = I18N.get('chat.sendBtn');

    const channelTabs = document.querySelectorAll('.channel-tab');
    channelTabs.forEach(tab => {
        const channel = tab.dataset.channel;
        if (channel) {
            tab.textContent = I18N.get(`chat.channels.${channel}`);
        }
    });

    const onlineLabel = document.querySelector('.online-label');
    if (onlineLabel) onlineLabel.textContent = I18N.get('chat.onlineAgents');

    const panelHeaders = {
        'panel-chat': 'chat.title',
        'panel-self-awareness': 'selfAwareness.title',
        'panel-memory': 'memory.title',
        'panel-metrics': 'metrics.title',
        'panel-thinking': 'thinking.title',
        'panel-decision': 'decision.title',
        'panel-personality': 'personality.title',
        'panel-knowledge': 'knowledge.title',
        'panel-learning': 'learning.title',
        'panel-logs': 'logs.title',
        'panel-file-ingestion': 'fileIngestion.title',
        'panel-mental': 'mental.title',
        'panel-action': 'action.title',
        'panel-evolution': 'evolution.title'
    };

    Object.entries(panelHeaders).forEach(([panelId, key]) => {
        const panel = document.getElementById(panelId);
        if (panel) {
            const h2 = panel.querySelector('h2');
            if (h2) h2.textContent = I18N.get(key);
        }
    });

    const selfAwarenessLabels = {
        '自我识别': I18N.get('selfAwareness.metrics.selfRecognition'),
        '能力意识': I18N.get('selfAwareness.metrics.capabilityAwareness'),
        '局限意识': I18N.get('selfAwareness.metrics.limitationAwareness'),
        '存在意识': I18N.get('selfAwareness.metrics.existenceAwareness'),
        '时间连续性': I18N.get('selfAwareness.metrics.temporalContinuity')
    };
    document.querySelectorAll('.self-metric-label').forEach(label => {
        const text = label.textContent;
        if (selfAwarenessLabels[text]) {
            label.textContent = selfAwarenessLabels[text];
        }
    });

    const memorySectionNames = {
        'L1 瞬时记忆': I18N.get('memory.sections.l1'),
        'L2 短期记忆': I18N.get('memory.sections.l2'),
        'L3 长期记忆': I18N.get('memory.sections.l3'),
        'L4 语义记忆': I18N.get('memory.sections.l4')
    };
    document.querySelectorAll('.memory-section-name').forEach(name => {
        const text = name.textContent;
        if (memorySectionNames[text]) {
            name.textContent = memorySectionNames[text];
        }
    });

    const metricLabels = {
        '自由能': I18N.get('metrics.labels.freeEnergy'),
        '置信度': I18N.get('metrics.labels.confidence'),
        '新颖度': I18N.get('metrics.labels.novelty'),
        '熵': I18N.get('metrics.labels.entropy'),
        '步数': I18N.get('metrics.labels.step'),
        '延迟(ms)': I18N.get('metrics.labels.latency')
    };
    document.querySelectorAll('.metric-label').forEach(label => {
        const text = label.textContent;
        if (metricLabels[text]) {
            label.textContent = metricLabels[text];
        }
    });

    const thinkingLabels = {
        '思考模式:': I18N.get('thinking.stats.mode') + ':',
        '系统2调用:': I18N.get('thinking.stats.system2') + ':',
        '思考置信度:': I18N.get('thinking.stats.confidence') + ':'
    };
    document.querySelectorAll('.thinking-stat-label').forEach(label => {
        const text = label.textContent;
        if (thinkingLabels[text]) {
            label.textContent = thinkingLabels[text];
        }
    });

    const problemInput = document.getElementById('problemInput');
    if (problemInput) problemInput.placeholder = I18N.get('thinking.inputPlaceholder');

    const decomposeBtn = document.getElementById('decomposeBtn');
    if (decomposeBtn) decomposeBtn.textContent = I18N.get('thinking.decomposeBtn');

    const criticalInput = document.getElementById('criticalInput');
    if (criticalInput) criticalInput.placeholder = I18N.get('thinking.criticalInputPlaceholder');

    const criticalBtn = document.getElementById('criticalBtn');
    if (criticalBtn) criticalBtn.textContent = I18N.get('thinking.criticalBtn');

    const personalityTraits = {
        '好奇心': I18N.get('personality.traits.curiosity'),
        '果断性': I18N.get('personality.traits.assertiveness'),
        '谨慎性': I18N.get('personality.traits.cautiousness'),
        '创造力': I18N.get('personality.traits.creativity'),
        '耐心': I18N.get('personality.traits.patience')
    };
    document.querySelectorAll('.trait-label').forEach(label => {
        const text = label.textContent;
        if (personalityTraits[text]) {
            label.textContent = personalityTraits[text];
        }
    });

    const personalityValues = {
        '生存': I18N.get('personality.values.survival'),
        '知识': I18N.get('personality.values.knowledge'),
        '成长': I18N.get('personality.values.growth')
    };
    document.querySelectorAll('.value-label').forEach(label => {
        const text = label.textContent;
        if (personalityValues[text]) {
            label.textContent = personalityValues[text];
        }
    });

    const logFilterOptions = {
        '全部': I18N.get('logs.filter.all'),
        '信息': I18N.get('logs.filter.info'),
        '警告': I18N.get('logs.filter.warning'),
        '错误': I18N.get('logs.filter.error')
    };
    const logFilter = document.getElementById('logLevelFilter');
    if (logFilter) {
        Array.from(logFilter.options).forEach(opt => {
            if (logFilterOptions[opt.textContent]) {
                opt.textContent = logFilterOptions[opt.textContent];
            }
        });
    }

    const uploadDropZone = document.querySelector('.upload-text');
    if (uploadDropZone) uploadDropZone.textContent = I18N.get('fileIngestion.upload.dropZone');

    const fileSearchInput = document.getElementById('fileSearchInput');
    if (fileSearchInput) fileSearchInput.placeholder = I18N.get('fileIngestion.search.placeholder');

    const fileSearchBtn = document.getElementById('fileSearchBtn');
    if (fileSearchBtn) fileSearchBtn.textContent = I18N.get('fileIngestion.search.btn');

    const fileSearchType = document.getElementById('fileSearchType');
    if (fileSearchType) {
        Array.from(fileSearchType.options).forEach(opt => {
            if (opt.textContent === '内容搜索') opt.textContent = I18N.get('fileIngestion.search.content');
            if (opt.textContent === '语义搜索') opt.textContent = I18N.get('fileIngestion.search.embedding');
        });
    }

    const mentalLayers = {
        '反射层': I18N.get('mental.layers.reflex'),
        '慎思层': I18N.get('mental.layers.deliberative'),
        '元认知层': I18N.get('mental.layers.meta')
    };
    document.querySelectorAll('.layer-name').forEach(name => {
        const text = name.textContent;
        if (mentalLayers[text]) {
            name.textContent = mentalLayers[text];
        }
    });

    const actionStatLabels = {
        '任务拆解': I18N.get('action.stats.decomposition'),
        '路径规划': I18N.get('action.stats.pathPlanning'),
        '执行次数': I18N.get('action.stats.executions'),
        '成功率': I18N.get('action.stats.successRate'),
        '活跃目标': I18N.get('action.stats.activeGoals'),
        '探索行动': I18N.get('action.stats.exploration')
    };
    document.querySelectorAll('.action-stat-label').forEach(label => {
        const text = label.textContent;
        if (actionStatLabels[text]) {
            label.textContent = actionStatLabels[text];
        }
    });

    const actionStepsLabel = document.querySelector('.action-control-row label');
    if (actionStepsLabel) actionStepsLabel.textContent = I18N.get('action.controls.steps') + ':';

    const executeBtn = document.getElementById('executeRunStepsBtn');
    if (executeBtn) executeBtn.textContent = I18N.get('action.controls.executeBtn');

    const evolutionStages = {
        '微进化': I18N.get('evolution.stages.micro'),
        '突触级': I18N.get('evolution.stages.microSub'),
        '中进化': I18N.get('evolution.stages.meso'),
        '规则技能级': I18N.get('evolution.stages.mesoSub'),
        '宏进化': I18N.get('evolution.stages.macro'),
        '架构级': I18N.get('evolution.stages.macroSub'),
        '元进化': I18N.get('evolution.stages.meta'),
        '系统级': I18N.get('evolution.stages.metaSub')
    };
    document.querySelectorAll('.stage-name, .stage-subtitle').forEach(el => {
        const text = el.textContent;
        if (evolutionStages[text]) {
            el.textContent = evolutionStages[text];
        }
    });
}

document.addEventListener('DOMContentLoaded', init);
