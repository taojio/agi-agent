document.addEventListener('DOMContentLoaded', () => {
    App.core.init();
    App.realtime.init();
    App.sessions.init();
    App.chat.init();
    App.memory.init();
    App.soul.init();
    App.tasks.init();
    App.evolution.init();
    App.security.init();
    App.selfimprovement.init();
    App.skills.init();
    App.knowledge.init();
    App.synaptic.init();
    App.agents.init();
    App.plugins.init();

    initRouter();
    initEventListeners();
});

function initRouter() {
    const core = App.core;

    core.dom.navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            switchView(e.target.getAttribute('data-view'));
        });
    });

    core.dom.mobileNavItems.forEach(item => {
        item.addEventListener('click', (e) => {
            switchView(e.target.getAttribute('data-view'));
            const mobileMenu = document.getElementById('mobileMenu');
            if (mobileMenu) mobileMenu.style.display = 'none';
        });
    });
}

function switchView(viewName) {
    const core = App.core;
    if (!viewName) return;

    core.dom.navItems.forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-view') === viewName);
    });

    core.dom.mobileNavItems.forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-view') === viewName);
    });

    core.dom.views.forEach(view => {
        view.style.display = view.id === `${viewName}View` ? 'block' : 'none';
    });

    window.history.pushState({ view: viewName }, '', `#${viewName}`);
}

function switchConfigTab(tabName) {
    const core = App.core;
    core.dom.configTabs.forEach(tab => {
        tab.classList.toggle('active', tab.id === `${tabName}Tab`);
    });
    core.dom.configSections.forEach(section => {
        section.style.display = section.id === `${tabName}Section` ? 'block' : 'none';
    });
}

function initEventListeners() {
    const core = App.core;

    if (core.dom.chatInput) {
        core.dom.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                App.chat.sendMessage();
            }
        });
    }

    if (core.dom.sendBtn) {
        core.dom.sendBtn.addEventListener('click', () => App.chat.sendMessage());
    }

    if (core.dom.voiceBtn) {
        core.dom.voiceBtn.addEventListener('click', () => App.chat.toggleVoiceInput());
    }

    if (core.dom.clearChatBtn) {
        core.dom.clearChatBtn.addEventListener('click', () => {
            core.dom.chatMessages.innerHTML = '';
            App.chat.currentMessages = [];
        });
    }

    if (core.dom.newSessionBtn) {
        core.dom.newSessionBtn.addEventListener('click', () => App.chat.createNewSession());
    }

    if (core.dom.exportSessionBtn) {
        core.dom.exportSessionBtn.addEventListener('click', () => App.chat.exportSession());
    }

    if (core.dom.chatSearchInput) {
        core.dom.chatSearchInput.addEventListener('input', (e) => {
        });
    }

    if (core.dom.configTemperature) {
        core.dom.configTemperature.addEventListener('input', (e) => {
            if (core.dom.configTemperatureValue) {
                core.dom.configTemperatureValue.textContent = e.target.value;
            }
        });
    }

    if (core.dom.saveConfigBtn) {
        core.dom.saveConfigBtn.addEventListener('click', async () => {
            const config = {
                temperature: parseFloat(core.dom.configTemperature?.value || '0.7')
            };
            try {
                const response = await fetch(`${core.API_BASE}/api/config`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                if (data.success) {
                    utils.showSuccess('配置已保存');
                } else {
                    utils.showError('保存失败: ' + (data.error || '未知错误'));
                }
            } catch (error) {
                utils.showError('保存失败: ' + (error.message || '未知错误'));
            }
        });
    }

    if (core.dom.resetConfigBtn) {
        core.dom.resetConfigBtn.addEventListener('click', async () => {
            try {
                const response = await fetch(`${core.API_BASE}/api/config/reset`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await response.json();
                if (data.success) {
                    if (core.dom.configTemperature) core.dom.configTemperature.value = '0.7';
                    if (core.dom.configTemperatureValue) core.dom.configTemperatureValue.textContent = '0.7';
                    utils.showSuccess('配置已重置');
                } else {
                    utils.showError('重置失败: ' + (data.error || '未知错误'));
                }
            } catch (error) {
                utils.showError('重置失败: ' + (error.message || '未知错误'));
            }
        });
    }

    if (core.dom.refreshBtn) {
        core.dom.refreshBtn.addEventListener('click', () => {
            location.reload();
        });
    }

    if (core.dom.fastModeBtn) {
        core.dom.fastModeBtn.addEventListener('click', () => {
            if (core.dom.fastModeToggle) {
                core.dom.fastModeToggle.checked = !core.dom.fastModeToggle.checked;
            }
        });
    }

    if (core.dom.commandPaletteBtn) {
        core.dom.commandPaletteBtn.addEventListener('click', () => {
            if (core.dom.commandPalette) {
                core.dom.commandPalette.style.display = core.dom.commandPalette.style.display === 'flex' ? 'none' : 'flex';
                if (core.dom.cpSearch) core.dom.cpSearch.focus();
            }
        });
    }

    if (core.dom.cpSearch) {
        core.dom.cpSearch.addEventListener('input', (e) => {
        });
    }

    if (core.dom.cpList) {
        core.dom.cpList.addEventListener('click', (e) => {
            const item = e.target.closest('.cp-item');
            if (item) {
                const action = item.getAttribute('data-action');
                if (action) {
                    switchView(action);
                    if (core.dom.commandPalette) core.dom.commandPalette.style.display = 'none';
                }
            }
        });
    }

    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (core.dom.commandPalette) {
                core.dom.commandPalette.style.display = core.dom.commandPalette.style.display === 'flex' ? 'none' : 'flex';
                if (core.dom.cpSearch) core.dom.cpSearch.focus();
            }
        }

        if (e.key === 'Escape') {
            if (core.dom.commandPalette && core.dom.commandPalette.style.display === 'flex') {
                core.dom.commandPalette.style.display = 'none';
            }
        }
    });

    if (core.dom.memoryTierBtns) {
        core.dom.memoryTierBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tier = e.target.getAttribute('data-tier');
                if (tier) {
                    App.memory.loadMemories(tier);
                    core.dom.memoryTierBtns.forEach(b => b.classList.toggle('active', b === e.target));
                }
            });
        });
    }

    if (core.dom.memorySearchInput) {
        core.dom.memorySearchInput.addEventListener('input', (e) => {
            App.memory.searchMemories(e.target.value);
        });
    }

    if (core.dom.addMemoryBtn) {
        core.dom.addMemoryBtn.addEventListener('click', () => App.memory.addMemory());
    }

    if (core.dom.soulTabs) {
        core.dom.soulTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.getAttribute('data-tab');
                if (tabName) {
                    core.dom.soulTabs.forEach(t => t.classList.toggle('active', t === e.target));
                    core.dom.soulSections.forEach(s => {
                        s.style.display = s.id === `${tabName}Section` ? 'block' : 'none';
                    });
                }
            });
        });
    }

    if (core.dom.saveSoulBtn) {
        core.dom.saveSoulBtn.addEventListener('click', () => App.soul.saveSoul());
    }

    if (core.dom.exportSoulBtn) {
        core.dom.exportSoulBtn.addEventListener('click', () => App.soul.exportSoul());
    }

    if (core.dom.submitTaskBtn) {
        core.dom.submitTaskBtn.addEventListener('click', () => App.tasks.submitTask());
    }

    if (core.dom.runEvolutionBtn) {
        core.dom.runEvolutionBtn.addEventListener('click', () => App.evolution.runEvolution());
    }

    if (core.dom.generateSkillBtn) {
        core.dom.generateSkillBtn.addEventListener('click', () => App.evolution.generateSkill());
    }

    if (core.dom.runDiagnosticBtn) {
        core.dom.runDiagnosticBtn.addEventListener('click', () => App.selfimprovement.runDiagnostic());
    }

    if (core.dom.generateProposalsBtn) {
        core.dom.generateProposalsBtn.addEventListener('click', () => App.selfimprovement.generateProposals());
    }

    if (core.dom.loadSkillsBtn) {
        core.dom.loadSkillsBtn.addEventListener('click', () => App.skills.loadSkills());
    }

    if (core.dom.newAgentBtn) {
        core.dom.newAgentBtn.addEventListener('click', () => App.agents.createNewAgent());
    }

    if (core.dom.saveAllSessionsBtn) {
        core.dom.saveAllSessionsBtn.addEventListener('click', () => App.sessions.saveAllSessions());
    }

    if (core.dom.exportAllSessionsBtn) {
        core.dom.exportAllSessionsBtn.addEventListener('click', () => App.sessions.exportAllSessions());
    }

    if (core.dom.fileInput) {
        core.dom.fileInput.addEventListener('change', (e) => App.chat.handleFileUpload(e));
    }

    if (core.dom.attachBtn) {
        core.dom.attachBtn.addEventListener('click', () => {
            if (core.dom.fileInput) core.dom.fileInput.click();
        });
    }

    window.addEventListener('popstate', (e) => {
        if (e.state && e.state.view) {
            switchView(e.state.view);
        }
    });

    const hash = window.location.hash.slice(1);
    if (hash) {
        switchView(hash);
    } else {
        switchView('dashboard');
    }
}
