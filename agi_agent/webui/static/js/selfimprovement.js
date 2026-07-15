const App = window.App || {};

App.selfimprovement = {
    async loadSelfImprovement() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/self_improvement`);
            const data = await response.json();
            this.renderSelfImprovement(data.improvement || {});
        } catch (error) {
            console.error('Failed to load self improvement:', error);
        }
    },

    renderSelfImprovement(data) {
        const core = App.core;

        if (core.dom.improvementPerformance) {
            const performance = data.performance || {};
            core.dom.improvementPerformance.innerHTML = `
                <div style="padding: 10px;">
                    <div>推理准确率: ${typeof performance.reasoning_accuracy === 'number' ? (performance.reasoning_accuracy * 100).toFixed(0) : '--'}%</div>
                    <div style="margin-top: 8px;">响应时间: ${typeof performance.response_time_ms === 'number' ? performance.response_time_ms : '--'}ms</div>
                    <div>内存效率: ${typeof performance.memory_efficiency === 'number' ? (performance.memory_efficiency * 100).toFixed(0) : '--'}%</div>
                </div>
            `;
        }

        if (core.dom.improvementDiagnostic) {
            const diagnostic = data.diagnostic || {};
            const issues = diagnostic.issues || [];
            core.dom.improvementDiagnostic.innerHTML = `
                <div style="padding: 10px;">
                    <div>诊断状态: ${diagnostic.last_run ? '已完成' : '未运行'}</div>
                    <div style="margin-top: 8px;">问题数: ${issues.length}</div>
                    ${issues.length > 0 ? `
                        <div style="margin-top: 8px; font-size: 12px; color: #FF9800;">
                            ${issues.slice(0, 3).map(i => `- ${i}`).join('<br>')}
                        </div>
                    ` : ''}
                </div>
            `;
        }

        if (core.dom.improvementProposals) {
            const proposals = data.proposals || [];
            core.dom.improvementProposals.innerHTML = proposals.length > 0 ? proposals.map(p => `
                <div style="padding: 10px; border-bottom: 1px solid var(--border-color);">
                    <div style="font-weight: bold;">${p.title}</div>
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">${p.description}</div>
                    <div style="font-size: 11px; color: ${p.implemented ? '#4CAF50' : '#FF9800'}; margin-top: 4px;">${p.implemented ? '已实施' : '待实施'}</div>
                </div>
            `).join('') : '<div style="color: var(--text-muted); padding: 20px;">暂无改进提案</div>';
        }

        if (core.dom.improvementSafety) {
            const safety = data.safety || {};
            core.dom.improvementSafety.innerHTML = `
                <div style="padding: 10px;">
                    <div>安全检查通过率: ${typeof safety.compliance_rate === 'number' ? (safety.compliance_rate * 100).toFixed(0) : '--'}%</div>
                    <div style="margin-top: 8px;">风险等级: ${safety.risk_level || '低'}</div>
                    <div>最近检查: ${core.formatTime(safety.last_check)}</div>
                </div>
            `;
        }
    },

    async runDiagnostic() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/self_improvement/diagnostic`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                utils.showSuccess('诊断完成');
                this.loadSelfImprovement();
            } else {
                utils.showError('诊断失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('诊断失败: ' + (error.message || '未知错误'));
        }
    },

    async generateProposals() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/self_improvement/proposals`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                utils.showSuccess('改进提案已生成');
                this.loadSelfImprovement();
            } else {
                utils.showError('生成失败: ' + (data.error || '未知错误'));
            }
        } catch (error) {
            utils.showError('生成失败: ' + (error.message || '未知错误'));
        }
    },

    init() {
        this.loadSelfImprovement();
    }
};

window.App = App;
