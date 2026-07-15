const visualization = {
    renderMemoryTimeline(container, memories) {
        if (!memories || memories.length === 0) {
            container.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">📅</div>
                    <div class="warning-text">暂无记忆数据</div>
                </div>
            `;
            return;
        }
        
        const grouped = this.groupMemoriesByDate(memories);
        
        container.innerHTML = `
            <div class="memory-timeline">
                ${Object.entries(grouped).map(([date, items]) => `
                    <div class="timeline-date">
                        <div class="timeline-date-label">${date}</div>
                        <div class="timeline-items">
                            ${items.map(memory => `
                                <div class="timeline-item">
                                    <div class="timeline-dot"></div>
                                    <div class="timeline-content">
                                        <div class="timeline-title">${memory.title || memory.content?.substring(0, 50) || '无标题'}</div>
                                        <div class="timeline-meta">
                                            <span>层级: ${memory.tier}</span>
                                            <span>${memory.timestamp ? utils.formatTime(memory.timestamp) : ''}</span>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    },
    
    groupMemoriesByDate(memories) {
        const groups = {};
        memories.forEach(memory => {
            const date = memory.timestamp 
                ? new Date(memory.timestamp).toLocaleDateString('zh-CN') 
                : '未知日期';
            if (!groups[date]) groups[date] = [];
            groups[date].push(memory);
        });
        return Object.entries(groups).sort((a, b) => new Date(b[0]) - new Date(a[0])).reduce((acc, [date, items]) => {
            acc[date] = items.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
            return acc;
        }, {});
    },
    
    renderMemoryGraph(container, memories) {
        if (!memories || memories.length < 2) {
            container.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">🔗</div>
                    <div class="warning-text">需要至少2条记忆才能生成关联图</div>
                </div>
            `;
            return;
        }
        
        const canvas = document.createElement('canvas');
        canvas.className = 'memory-graph-canvas';
        container.innerHTML = '';
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        const width = container.clientWidth;
        const height = Math.min(400, container.clientHeight);
        canvas.width = width;
        canvas.height = height;
        
        const nodes = memories.slice(0, 10).map((m, i) => ({
            x: width * 0.2 + (width * 0.6 * (i % 5)) / 5,
            y: height * 0.2 + (height * 0.6 * Math.floor(i / 5)) / 2,
            radius: 20 + Math.random() * 10,
            memory: m
        }));
        
        ctx.fillStyle = 'var(--bg-card)';
        ctx.fillRect(0, 0, width, height);
        
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 150) {
                    ctx.beginPath();
                    ctx.strokeStyle = `rgba(0, 212, 255, ${0.3 * (1 - distance / 150)})`;
                    ctx.lineWidth = 1;
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.stroke();
                }
            }
        }
        
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.fillStyle = 'var(--primary-color)';
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.beginPath();
            ctx.fillStyle = 'var(--bg-primary)';
            ctx.arc(node.x, node.y, node.radius - 4, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = 'var(--text-primary)';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const title = node.memory.title || node.memory.content?.substring(0, 10) || '...';
            ctx.fillText(title, node.x, node.y);
        });
    },
    
    renderKnowledgeGraph(container, data) {
        if (!data || !data.nodes || data.nodes.length === 0) {
            container.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">🔗</div>
                    <div class="warning-text">知识图谱暂无数据</div>
                </div>
            `;
            return;
        }
        
        const canvas = document.createElement('canvas');
        canvas.className = 'knowledge-graph-canvas';
        container.innerHTML = '';
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        const width = container.clientWidth;
        const height = Math.min(500, container.clientHeight);
        canvas.width = width;
        canvas.height = height;
        
        const nodes = data.nodes.map((node, i) => ({
            ...node,
            x: width * 0.5 + Math.cos(i * 0.5) * (width * 0.3),
            y: height * 0.5 + Math.sin(i * 0.5) * (height * 0.3),
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: 15 + Math.log((node.weight || 1) + 1) * 5
        }));
        
        const edges = data.edges || [];
        
        let animationId;
        const animate = () => {
            ctx.fillStyle = 'var(--bg-card)';
            ctx.fillRect(0, 0, width, height);
            
            edges.forEach(edge => {
                const source = nodes.find(n => n.id === edge.source);
                const target = nodes.find(n => n.id === edge.target);
                
                if (source && target) {
                    ctx.beginPath();
                    ctx.strokeStyle = 'rgba(124, 58, 237, 0.4)';
                    ctx.lineWidth = edge.weight || 1;
                    ctx.moveTo(source.x, source.y);
                    ctx.lineTo(target.x, target.y);
                    ctx.stroke();
                }
            });
            
            nodes.forEach(node => {
                ctx.beginPath();
                const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius);
                gradient.addColorStop(0, 'var(--secondary-color)');
                gradient.addColorStop(1, 'rgba(124, 58, 237, 0.3)');
                ctx.fillStyle = gradient;
                ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.fillStyle = 'var(--text-primary)';
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const label = node.label || node.name || node.id;
                ctx.fillText(label.length > 8 ? label.substring(0, 8) + '...' : label, node.x, node.y);
            });
            
            nodes.forEach((node, i) => {
                node.vx *= 0.98;
                node.vy *= 0.98;
                
                nodes.forEach((other, j) => {
                    if (i !== j) {
                        const dx = node.x - other.x;
                        const dy = node.y - other.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        const minDist = node.radius + other.radius + 20;
                        
                        if (distance < minDist) {
                            const force = (minDist - distance) / distance * 0.5;
                            node.vx += dx * force;
                            node.vy += dy * force;
                        }
                    }
                });
                
                edges.forEach(edge => {
                    if (edge.source === node.id || edge.target === node.id) {
                        const other = edge.source === node.id 
                            ? nodes.find(n => n.id === edge.target)
                            : nodes.find(n => n.id === edge.source);
                        
                        if (other) {
                            const dx = other.x - node.x;
                            const dy = other.y - node.y;
                            const distance = Math.sqrt(dx * dx + dy * dy);
                            const springLen = 100;
                            const force = (distance - springLen) / distance * 0.05;
                            node.vx += dx * force;
                            node.vy += dy * force;
                        }
                    }
                });
                
                node.x += node.vx;
                node.y += node.vy;
                
                node.x = Math.max(node.radius, Math.min(width - node.radius, node.x));
                node.y = Math.max(node.radius, Math.min(height - node.radius, node.y));
            });
            
            animationId = requestAnimationFrame(animate);
        };
        
        animate();
        
        container.addEventListener('mouseenter', () => {
            if (!animationId) animate();
        });
        
        container.addEventListener('mouseleave', () => {
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
        });
    },
    
    renderEvolutionChart(container, history) {
        if (!history || history.length === 0) {
            container.innerHTML = `
                <div class="empty-state-warning">
                    <div class="warning-icon">📈</div>
                    <div class="warning-text">暂无进化历史数据</div>
                </div>
            `;
            return;
        }
        
        const canvas = document.createElement('canvas');
        canvas.className = 'evolution-chart-canvas';
        container.innerHTML = '';
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        const width = container.clientWidth;
        const height = 200;
        canvas.width = width;
        canvas.height = height;
        
        const padding = 40;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;
        
        ctx.fillStyle = 'var(--bg-card)';
        ctx.fillRect(0, 0, width, height);
        
        const values = history.map(h => h.fitness || h.value || 0);
        const maxValue = Math.max(...values, 1);
        const minValue = Math.min(...values, 0);
        const valueRange = maxValue - minValue || 1;
        
        ctx.strokeStyle = 'var(--border-color)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding + (chartHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
            
            const value = maxValue - (valueRange / 4) * i;
            ctx.fillStyle = 'var(--text-muted)';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(value.toFixed(2), padding - 5, y + 3);
        }
        
        ctx.strokeStyle = 'var(--primary-color)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        history.forEach((point, i) => {
            const x = padding + (chartWidth / (history.length - 1)) * i;
            const y = padding + chartHeight - ((point.fitness || point.value || 0) - minValue) / valueRange * chartHeight;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            
            ctx.beginPath();
            ctx.fillStyle = 'var(--primary-color)';
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        });
        
        ctx.stroke();
        
        ctx.fillStyle = 'var(--text-secondary)';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('进化代数', width / 2, height - 10);
        
        ctx.save();
        ctx.translate(10, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('适应度', 0, 0);
        ctx.restore();
    },
    
    renderModuleActivity(container, data) {
        const modules = [
            { name: '感知', value: data.sensory_processing || 0, color: '#00d4ff' },
            { name: '推理', value: data.reasoning || 0, color: '#7c3aed' },
            { name: '记忆', value: data.memory_access || 0, color: '#f472b6' },
            { name: '行动', value: data.action_selection || 0, color: '#22c55e' },
            { name: '自我意识', value: data.self_awareness || 0, color: '#f59e0b' },
            { name: '情绪', value: data.emotional_state || 0, color: '#ef4444' }
        ];
        
        container.innerHTML = modules.map(m => `
            <div class="module-activity-item">
                <span class="module-name">${m.name}</span>
                <div class="module-bar-container">
                    <div class="module-bar" style="width: ${Math.min(m.value * 100, 100)}%; background: ${m.color}"></div>
                </div>
                <span class="module-value">${typeof m.value === 'number' ? (m.value * 100).toFixed(0) : '--'}%</span>
            </div>
        `).join('');
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = visualization;
}