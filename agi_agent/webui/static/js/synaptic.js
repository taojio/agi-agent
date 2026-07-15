const App = window.App || {};

App.synaptic = {
    async loadSynapticData() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/synaptic`);
            const data = await response.json();
            this.renderSynapticData(data.synaptic || {});
        } catch (error) {
            console.error('Failed to load synaptic data:', error);
        }
    },

    renderSynapticData(data) {
        const core = App.core;

        if (core.dom.synapticStats) {
            core.dom.synapticStats.innerHTML = `
                <span>连接数: ${data.connection_count || 0}</span>
                <span>活动模块: ${data.active_modules || 0}</span>
                <span>信号速率: ${data.signal_rate || 0}/s</span>
            `;
        }

        if (data.connections && data.connections.length > 0) {
            this.renderSynapticGraph(data.connections);
        }

        if (data.oscillators) {
            this.renderOscillators(data.oscillators);
        }

        if (data.signal_flow) {
            this.renderSignalFlow(data.signal_flow);
        }
    },

    renderSynapticGraph(connections) {
        const core = App.core;
        core.dom.synapticGraph.innerHTML = `
            <div style="padding: 10px;">
                <div style="font-weight: bold; margin-bottom: 10px;">突触连接图</div>
                <div style="max-height: 200px; overflow-y: auto;">
                    ${connections.slice(0, 20).map(c => `
                        <div style="padding: 4px; font-size: 12px; color: var(--text-secondary);">
                            <span>${c.from}</span> → <span>${c.to}</span> 
                            <span style="color: ${c.strength > 0.5 ? '#4CAF50' : '#FF9800'}; margin-left: 10px;">${(c.strength * 100).toFixed(0)}%</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    },

    renderOscillators(oscillators) {
        const core = App.core;
        core.dom.oscillatorDisplay.innerHTML = `
            <div style="padding: 10px;">
                <div style="font-weight: bold; margin-bottom: 10px;">振荡器状态</div>
                ${oscillators.map(o => `
                    <div style="display: flex; justify-content: space-between; padding: 4px;">
                        <span style="font-size: 12px;">${o.name}</span>
                        <span style="font-size: 12px; color: var(--text-muted);">频率: ${o.frequency || 0}Hz</span>
                        <span style="font-size: 12px; color: ${o.phase > 0.5 ? '#4CAF50' : '#2196F3'};">相位: ${(o.phase * 100).toFixed(0)}%</span>
                    </div>
                `).join('')}
            </div>
        `;
    },

    renderSignalFlow(signalFlow) {
        const core = App.core;
        core.dom.signalFlowChart.innerHTML = `
            <div style="padding: 10px;">
                <div style="font-weight: bold; margin-bottom: 10px;">信号流向</div>
                <div style="max-height: 150px; overflow-y: auto;">
                    ${signalFlow.map(s => `
                        <div style="padding: 4px; font-size: 12px;">
                            <span style="color: #2196F3;">${s.source}</span> → 
                            <span style="color: #4CAF50;">${s.target}</span>
                            <span style="color: var(--text-muted); margin-left: 10px;">延迟: ${s.latency || 0}ms</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    },

    init() {
        this.loadSynapticData();
    }
};

window.App = App;
