const App = window.App || {};

App.security = {
    async loadSecurity() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/security`);
            const data = await response.json();
            this.renderSecurity(data.security || {});
        } catch (error) {
            console.error('Failed to load security:', error);
        }
    },

    renderSecurity(data) {
        const core = App.core;

        if (core.dom.securityHardBoundary) {
            const hardBoundary = data.hard_boundary || {};
            core.dom.securityHardBoundary.innerHTML = `
                <div style="padding: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>状态:</span>
                        <span style="color: ${hardBoundary.enabled ? '#4CAF50' : '#f44336'};">${hardBoundary.enabled ? '已启用' : '已禁用'}</span>
                    </div>
                    <div style="margin-top: 8px;">规则数: ${hardBoundary.rules_count || 0}</div>
                    <div>拦截数: ${hardBoundary.blocked_count || 0}</div>
                </div>
            `;
        }

        if (core.dom.securityCircuitBreaker) {
            const breaker = data.circuit_breaker || {};
            core.dom.securityCircuitBreaker.innerHTML = `
                <div style="padding: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>状态:</span>
                        <span style="color: ${breaker.state === 'closed' ? '#4CAF50' : '#f44336'};">${breaker.state || '未知'}</span>
                    </div>
                    <div style="margin-top: 8px;">失败计数: ${breaker.failure_count || 0}</div>
                    <div>阈值: ${breaker.threshold || 0}</div>
                </div>
            `;
        }

        if (core.dom.securityRiskClassifier) {
            const classifier = data.risk_classifier || {};
            core.dom.securityRiskClassifier.innerHTML = `
                <div style="padding: 10px;">
                    <div>模型版本: ${classifier.model_version || '1.0'}</div>
                    <div style="margin-top: 8px;">风险检测率: ${typeof classifier.detection_rate === 'number' ? (classifier.detection_rate * 100).toFixed(0) : '--'}%</div>
                    <div>误报率: ${typeof classifier.false_positive_rate === 'number' ? (classifier.false_positive_rate * 100).toFixed(0) : '--'}%</div>
                </div>
            `;
        }

        if (core.dom.securityAudit) {
            const audit = data.audit || {};
            core.dom.securityAudit.innerHTML = `
                <div style="padding: 10px;">
                    <div>审计日志数: ${audit.log_count || 0}</div>
                    <div style="margin-top: 8px;">最近审计: ${core.formatTime(audit.last_audit)}</div>
                    <div>异常事件: ${audit.anomaly_count || 0}</div>
                </div>
            `;
        }

        if (core.dom.securityCompliance) {
            const compliance = data.compliance || {};
            core.dom.securityCompliance.innerHTML = `
                <div style="padding: 10px;">
                    <div>合规检查: ${compliance.passed || 0}/${compliance.total || 0}</div>
                    <div style="margin-top: 8px;">合规率: ${compliance.total > 0 ? ((compliance.passed / compliance.total) * 100).toFixed(0) : '--'}%</div>
                </div>
            `;
        }
    },

    init() {
        this.loadSecurity();
    }
};

window.App = App;
