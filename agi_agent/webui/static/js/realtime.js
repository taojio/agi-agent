const App = window.App || {};

App.realtime = {
    connectRealtimeSocket() {
        const core = App.core;
        if (core.state.realtimeSocket) {
            core.state.realtimeSocket.close();
        }

        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        core.state.realtimeSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/realtime`);

        core.state.realtimeSocket.onopen = () => {
            core.state.realtimeConnected = true;
            console.log('Realtime WebSocket connected');
        };

        core.state.realtimeSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'realtime_update') {
                    this.processRealtimeUpdates(data.updates);
                }
            } catch (error) {
                console.error('Failed to parse realtime update:', error);
            }
        };

        core.state.realtimeSocket.onclose = () => {
            core.state.realtimeConnected = false;
            console.log('Realtime WebSocket disconnected');
            setTimeout(() => this.connectRealtimeSocket(), 5000);
        };

        core.state.realtimeSocket.onerror = (error) => {
            console.error('Realtime WebSocket error:', error);
        };
    },

    processRealtimeUpdates(updates) {
        updates.forEach(update => {
            switch (update.module) {
                case 'synaptic':
                    this.handleSynapticUpdate(update);
                    break;
                case 'agent':
                    this.handleAgentUpdate(update.data);
                    break;
                case 'tasks':
                    if (App.tasks) {
                        App.tasks.handleTasksUpdate(update.data);
                    }
                    break;
                case 'memory':
                    if (App.memory) {
                        App.memory.handleMemoryUpdate(update.data);
                    }
                    break;
                case 'knowledge':
                    if (App.knowledge) {
                        App.knowledge.handleKnowledgeUpdate(update.data);
                    }
                    break;
            }
        });
    },

    handleSynapticUpdate(update) {
        const core = App.core;
        core.state.busHasData = update.has_data;
        this.updateBusStatusIndicator();

        if (update.has_data && update.data) {
            this.renderModuleActivity(update.data);
            this.renderSynapticStats(update.data);
        } else {
            this.renderEmptyBusState();
        }
    },

    handleAgentUpdate(data) {
        const core = App.core;
        if (App.chat) {
            App.chat.updateAgentInfo(data);
        }
        if (core.dom.statFreeEnergy) {
            core.dom.statFreeEnergy.textContent = typeof data.free_energy === 'number' 
                ? data.free_energy.toFixed(2) : '0.00';
        }
        if (core.dom.statConfidence) {
            core.dom.statConfidence.textContent = typeof data.confidence === 'number' 
                ? `${(data.confidence * 100).toFixed(0)}%` : '0%';
        }
    },

    updateBusStatusIndicator() {
        const core = App.core;
        if (!core.dom.gatewayStatus) return;

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
            core.dom.gatewayStatus.appendChild(indicator);
        }

        const statusIndicator = document.getElementById('busStatusIndicator');
        if (core.state.busHasData) {
            statusIndicator.innerHTML = '<span style="color: var(--success-color);">●</span><span style="color: var(--success-color);">总线有数据</span>';
            statusIndicator.style.background = 'rgba(34, 197, 94, 0.1)';
        } else {
            statusIndicator.innerHTML = '<span style="color: var(--warning-color);">⚠</span><span style="color: var(--warning-color);">总线无数据</span>';
            statusIndicator.style.background = 'rgba(251, 191, 36, 0.1)';
        }
    },

    renderEmptyBusState() {
        const core = App.core;

        if (core.dom.synapticStats) {
            core.dom.synapticStats.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">⚠</div>
                    <div class="warning-text">总线无数据</div>
                    <div class="warning-hint">系统等待数据输入...</div>
                </div>
            `;
        }

        if (core.dom.moduleActivityList) {
            core.dom.moduleActivityList.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">🔌</div>
                    <div class="warning-text">暂无模块活动</div>
                    <div class="warning-hint">启动Agent或输入数据以激活总线</div>
                </div>
            `;
        }

        if (core.dom.synapticGraph) {
            core.dom.synapticGraph.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">🕸️</div>
                    <div class="warning-text">突触网络未激活</div>
                </div>
            `;
        }

        if (core.dom.signalFlowChart) {
            core.dom.signalFlowChart.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">💤</div>
                    <div class="warning-text">无信号流动</div>
                </div>
            `;
        }

        if (core.dom.oscillatorDisplay) {
            core.dom.oscillatorDisplay.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">⏸️</div>
                    <div class="warning-text">振荡器未运行</div>
                </div>
            `;
        }
    },

    renderModuleActivity(data) {
        const core = App.core;
        if (!core.dom.moduleActivityList) return;

        const modules = [
            { name: '感知', value: data.sensory_processing || 0 },
            { name: '推理', value: data.reasoning || 0 },
            { name: '记忆', value: data.memory_access || 0 },
            { name: '行动', value: data.action_selection || 0 },
            { name: '自我意识', value: data.self_awareness || 0 },
            { name: '情绪', value: data.emotional_state || 0 }
        ];

        core.dom.moduleActivityList.innerHTML = modules.map(m => `
            <div class="module-activity-item">
                <span class="module-name">${m.name}</span>
                <div class="module-bar-container">
                    <div class="module-bar" style="width: ${Math.min(m.value * 100, 100)}%"></div>
                </div>
                <span class="module-value">${typeof m.value === 'number' ? (m.value * 100).toFixed(0) : '--'}%</span>
            </div>
        `).join('');
    },

    renderSynapticStats(data) {
        const core = App.core;
        if (!core.dom.synapticStats) return;

        core.dom.synapticStats.innerHTML = `
            <span>自由能: <strong>${typeof data.free_energy === 'number' ? data.free_energy.toFixed(4) : '--'}</strong></span>
            <span>置信度: <strong>${typeof data.confidence === 'number' ? (data.confidence * 100).toFixed(0) : '--'}%</strong></span>
            <span>新奇度: <strong>${typeof data.novelty === 'number' ? (data.novelty * 100).toFixed(0) : '--'}%</strong></span>
            <span>复杂度: <strong>${typeof data.complexity === 'number' ? (data.complexity * 100).toFixed(0) : '--'}%</strong></span>
        `;
    },

    init() {
        this.connectRealtimeSocket();
        this.updateBusStatusIndicator();
    }
};

window.App = App;
