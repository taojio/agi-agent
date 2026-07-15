let realtimeSocket = null;
let realtimeConnected = false;
let busHasData = false;

const socket = {
    connect() {
        if (realtimeSocket) {
            realtimeSocket.close();
        }
        
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        realtimeSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/realtime`);
        
        realtimeSocket.onopen = () => {
            realtimeConnected = true;
            console.log('Realtime WebSocket connected');
            utils.showInfo('WebSocket 连接已建立');
        };
        
        realtimeSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'realtime_update') {
                    this.processUpdates(data.updates);
                }
            } catch (error) {
                console.error('Failed to parse realtime update:', error);
            }
        };
        
        realtimeSocket.onclose = () => {
            realtimeConnected = false;
            console.log('Realtime WebSocket disconnected');
            setTimeout(() => this.connect(), 5000);
        };
        
        realtimeSocket.onerror = (error) => {
            console.error('Realtime WebSocket error:', error);
            utils.showWarning('WebSocket 连接异常');
        };
    },
    
    processUpdates(updates) {
        updates.forEach(update => {
            switch (update.module) {
                case 'synaptic':
                    if (typeof handleSynapticUpdate === 'function') {
                        handleSynapticUpdate(update);
                    }
                    break;
                case 'agent':
                    if (typeof handleAgentUpdate === 'function') {
                        handleAgentUpdate(update.data);
                    }
                    break;
                case 'tasks':
                    if (typeof handleTasksUpdate === 'function') {
                        handleTasksUpdate(update.data);
                    }
                    break;
                case 'memory':
                    if (typeof handleMemoryUpdate === 'function') {
                        handleMemoryUpdate(update.data);
                    }
                    break;
                case 'knowledge':
                    if (typeof handleKnowledgeUpdate === 'function') {
                        handleKnowledgeUpdate(update.data);
                    }
                    break;
            }
        });
    },
    
    isConnected() {
        return realtimeConnected;
    },
    
    send(data) {
        if (realtimeSocket && realtimeConnected) {
            realtimeSocket.send(JSON.stringify(data));
        }
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = socket;
}