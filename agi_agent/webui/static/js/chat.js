const App = window.App || {};

App.chat = {
    currentMessages: [],

    async sendMessage() {
        const core = App.core;
        const message = core.dom.chatInput.value.trim();
        if (!message) return;

        this.addMessage('user', message);
        core.dom.chatInput.value = '';

        try {
            const response = await fetch(`${core.API_BASE}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    session_id: core.state.currentSessionId,
                    fast_mode: core.dom.fastModeToggle?.checked || false
                })
            });

            const data = await response.json();
            if (data.response) {
                this.addMessage('agent', data.response);
            } else {
                this.addMessage('system', data.error || '未知错误');
            }
        } catch (error) {
            this.addMessage('system', '发送失败: ' + (error.message || '网络错误'));
        }
    },

    addMessage(sender, content) {
        const core = App.core;
        const messageId = Date.now();
        const message = { id: messageId, sender, content, timestamp: Date.now() };
        this.currentMessages.push(message);

        const msgClass = sender === 'user' ? 'user-message' : (sender === 'agent' ? 'agent-message' : 'system-message');
        const avatar = sender === 'user' ? '👤' : (sender === 'agent' ? '🤖' : '📢');

        const msgElement = document.createElement('div');
        msgElement.className = `message ${msgClass}`;
        msgElement.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-sender">${sender === 'user' ? '用户' : (sender === 'agent' ? 'Agent' : '系统')}</span>
                    <span class="message-time">${core.formatTime(message.timestamp)}</span>
                </div>
                <div class="message-text">${content}</div>
            </div>
        `;

        core.dom.chatMessages.appendChild(msgElement);
        core.dom.chatMessages.scrollTop = core.dom.chatMessages.scrollHeight;
    },

    handleSlashCommand(command) {
        const core = App.core;
        const parts = command.split(' ');
        const cmd = parts[0].toLowerCase();

        switch (cmd) {
            case '/clear':
                core.dom.chatMessages.innerHTML = '';
                this.currentMessages = [];
                utils.showSuccess('聊天记录已清除');
                break;
            case '/new':
                this.createNewSession();
                break;
            case '/save':
                this.saveCurrentSession();
                break;
            case '/session':
                if (parts[1]) {
                    this.loadSession(parts[1]);
                } else {
                    utils.showInfo('用法: /session <会话ID>');
                }
                break;
            case '/agent':
                if (parts[1]) {
                    if (App.agents) {
                        App.agents.loadAgentDetail(parts[1]);
                    }
                } else {
                    utils.showInfo('用法: /agent <agent名称>');
                }
                break;
            case '/help':
                this.showHelp();
                break;
            default:
                utils.showError(`未知命令: ${cmd}`);
        }
    },

    showHelp() {
        const helpText = `
            <strong>可用命令:</strong><br>
            /clear - 清除聊天记录<br>
            /new - 创建新会话<br>
            /save - 保存当前会话<br>
            /session <ID> - 切换会话<br>
            /agent <名称> - 查看Agent详情<br>
            /help - 显示帮助
        `;
        this.addMessage('system', helpText);
    },

    toggleVoiceInput() {
        const core = App.core;
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            utils.showError('您的浏览器不支持语音输入');
            return;
        }

        if (core.state.isVoiceRecording) {
            this.stopVoiceRecording();
        } else {
            this.startVoiceRecording();
        }
    },

    startVoiceRecording() {
        const core = App.core;
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        core.state.recognition = new SpeechRecognition();
        core.state.recognition.continuous = true;
        core.state.recognition.interimResults = true;
        core.state.recognition.lang = 'zh-CN';

        core.state.recognition.onstart = () => {
            core.state.isVoiceRecording = true;
            core.dom.voiceBtn.style.backgroundColor = '#f44336';
            core.dom.voiceBtn.textContent = '🎤 停止';
            utils.showInfo('正在听...');
        };

        core.state.recognition.onresult = (event) => {
            let transcript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                transcript += event.results[i][0].transcript;
            }
            core.dom.chatInput.value = transcript;
        };

        core.state.recognition.onerror = (event) => {
            console.error('语音识别错误:', event.error);
            this.stopVoiceRecording();
            utils.showError('语音识别失败: ' + event.error);
        };

        core.state.recognition.onend = () => {
            if (core.state.isVoiceRecording) {
                this.stopVoiceRecording();
            }
        };

        core.state.recognition.start();
    },

    stopVoiceRecording() {
        const core = App.core;
        if (core.state.recognition) {
            core.state.recognition.stop();
            core.state.recognition = null;
        }
        core.state.isVoiceRecording = false;
        core.dom.voiceBtn.style.backgroundColor = '';
        core.dom.voiceBtn.textContent = '🎤 语音';
    },

    async createNewSession() {
        const core = App.core;
        await core.createNewSession();
        this.showSuccess('新会话已创建');
    },
Session() {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: '新会话' })
            });
            const data = await response.json();
            if (data.session_id) {
                core.state.currentSessionId = data.session_id;
                core.dom.chatSessionName.textContent = '新会话';
                core.dom.chatMessages.innerHTML = '';
                this.currentMessages = [];
                if (App.sessions) {
                    App.sessions.loadSessions();
                }
                utils.showSuccess('新会话已创建');
            }
        } catch (error) {
            utils.showError('创建会话失败: ' + (error.message || '未知错误'));
        }
    },

    async saveCurrentSession() {
        const core = App.core;
        if (!core.state.currentSessionId) {
            utils.showError('没有当前会话');
            return;
        }

        const name = await core.showModalPrompt('保存会话', '输入会话名称', core.dom.chatSessionName.textContent);
        if (!name) return;

        try {
            const response = await fetch(`${core.API_BASE}/api/sessions/${core.state.currentSessionId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            });
            const data = await response.json();
            if (data.success) {
                core.dom.chatSessionName.textContent = name;
                if (App.sessions) {
                    App.sessions.loadSessions();
                }
                utils.showSuccess('会话已保存');
            }
        } catch (error) {
            utils.showError('保存会话失败: ' + (error.message || '未知错误'));
        }
    },

    async loadSession(sessionId) {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/sessions/${sessionId}`);
            const data = await response.json();
            if (data.session) {
                core.state.currentSessionId = sessionId;
                core.dom.chatSessionName.textContent = data.session.name || '未命名会话';
                core.dom.chatMessages.innerHTML = '';
                this.currentMessages = [];

                if (data.session.messages) {
                    data.session.messages.forEach(msg => {
                        this.addMessage(msg.sender, msg.content);
                    });
                }
                utils.showSuccess(`会话 "${data.session.name}" 已加载`);
            }
        } catch (error) {
            utils.showError('加载会话失败: ' + (error.message || '未知错误'));
        }
    },

    async exportSession() {
        const core = App.core;
        if (!core.state.currentSessionId) {
            utils.showError('没有当前会话');
            return;
        }

        try {
            const response = await fetch(`${core.API_BASE}/api/sessions/${core.state.currentSessionId}/export`);
            const data = await response.json();
            if (data.session) {
                const blob = new Blob([JSON.stringify(data.session, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `session_${core.state.currentSessionId}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                utils.showSuccess('会话已导出');
            }
        } catch (error) {
            utils.showError('导出会话失败: ' + (error.message || '未知错误'));
        }
    },

    async handleFileUpload(event) {
        const core = App.core;
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        utils.showInfo(`正在上传文件: ${file.name}`);

        try {
            const response = await fetch(`${core.API_BASE}/api/files/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                utils.showSuccess(`文件 "${file.name}" 上传成功`);
                await this.startFileLearning(file.name);
            } else {
                utils.showError(`上传失败: ${data.error}`);
            }
        } catch (error) {
            utils.showError('上传失败: ' + (error.message || '未知错误'));
        }

        core.dom.fileInput.value = '';
    },

    async startFileLearning(fileName) {
        const core = App.core;
        try {
            const response = await fetch(`${core.API_BASE}/api/files/learn`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: fileName })
            });
            const data = await response.json();
            if (data.success) {
                utils.showSuccess(`文件 "${fileName}" 学习完成`);
            } else {
                utils.showError(`学习失败: ${data.error}`);
            }
        } catch (error) {
            utils.showError('学习失败: ' + (error.message || '未知错误'));
        }
    },

    updateAgentInfo(data) {
        const core = App.core;
        if (core.dom.agentInfoName) {
            core.dom.agentInfoName.textContent = data.name || 'AGI Agent';
        }
        if (core.dom.agentInfoStep) {
            core.dom.agentInfoStep.textContent = `Step: ${data.step || 0}`;
        }
        if (core.dom.agentInfoStatus) {
            core.dom.agentInfoStatus.textContent = data.status || 'Idle';
        }
        if (core.dom.agentInfoDim) {
            core.dom.agentInfoDim.textContent = `Dim: ${data.dimension || 'N/A'}`;
        }
    },

    init() {}
};

window.loadSession = (id) => App.chat.loadSession(id);

window.App = App;
