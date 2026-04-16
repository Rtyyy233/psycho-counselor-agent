/**
 * Psychological Counselor Web Client
 *
 * Handles WebSocket communication, UI updates, and session management.
 */

// ========== State ==========
let ws = null;
let currentSessionId = null;
let isConnected = false;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// ========== DOM Elements ==========
const sidebar = document.getElementById('sidebar');
const btnMenu = document.getElementById('btn-menu');
const btnNewChat = document.getElementById('btn-new-chat');
const btnClear = document.getElementById('btn-clear');
const sessionList = document.getElementById('session-list');
const chatTitle = document.getElementById('chat-title');
const messagesContainer = document.getElementById('messages-container');
const messageInput = document.getElementById('message-input');
const btnSend = document.getElementById('btn-send');
const indicatorChatter = document.getElementById('indicator-chatter');
const indicatorAnalyst = document.getElementById('indicator-analyst');
const indicatorSupervisor = document.getElementById('indicator-supervisor');
const statsMessages = document.getElementById('stats-messages');
const statsTokens = document.getElementById('stats-tokens');
const sidebarOverlay = document.getElementById('sidebar-overlay');

// ========== WebSocket Functions ==========

function connectWebSocket(sessionId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat?session_id=${sessionId}`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        isConnected = true;
        reconnectAttempts = 0;
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    ws.onclose = () => {
        isConnected = false;
        console.log('WebSocket closed');

        // Auto-reconnect on disconnect
        if (currentSessionId && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnecting... attempt ${reconnectAttempts}`);
                connectWebSocket(currentSessionId);
            }, 1000 * reconnectAttempts);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function sendMessage(type, content) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type, content }));
    }
}

// ========== Message Handlers ==========

function handleMessage(data) {
    switch (data.type) {
        case 'session_info':
            currentSessionId = data.session_id;
            chatTitle.textContent = data.title || 'Psychological Counselor';
            loadMessages(data.messages);
            break;

        case 'message':
            hideIndicator('chatter');
            addMessage(data.content, data.sender);
            break;

        case 'indicator':
            showIndicator(data.agent, data.status);
            break;

        case 'context_stats':
            updateContextStats(data);
            break;

        case 'pong':
            // Heartbeat response, no action needed
            break;
    }
}

function loadMessages(messages) {
    messagesContainer.innerHTML = '';

    messages.forEach((msg) => {
        addMessage(msg.content, msg.role, msg.timestamp);
    });
}

function addMessage(content, role, timestamp = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role === 'user' ? 'user' : 'assistant'}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = timestamp
        ? new Date(timestamp).toLocaleTimeString()
        : new Date().toLocaleTimeString();

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// ========== Indicator Functions ==========

function showIndicator(agent, status) {
    const indicatorMap = {
        chatter: indicatorChatter,
        analyst: indicatorAnalyst,
        supervisor: indicatorSupervisor
    };

    const indicator = indicatorMap[agent];
    if (!indicator) return;

    if (status === 'typing' || status === 'thinking') {
        indicator.style.display = 'flex';
    } else if (status === 'done' || status === 'idle') {
        indicator.style.display = 'none';
    }
}

function hideIndicator(agent) {
    showIndicator(agent, 'done');
}

function hideAllIndicators() {
    indicatorChatter.style.display = 'none';
    indicatorAnalyst.style.display = 'none';
    indicatorSupervisor.style.display = 'none';
}

// ========== Context Stats ==========

function updateContextStats(data) {
    statsMessages.textContent = `${data.messages} messages`;

    const tokens = data.tokens || 0;
    statsTokens.textContent = `~${tokens} tokens`;

    if (data.summarized) {
        addMessage(`[Context summarized: ${data.summary}]`, 'assistant');
    }
}

// ========== Session Management ==========

async function loadSessionList() {
    try {
        const response = await fetch('/api/sessions');
        const sessions = await response.json();

        sessionList.innerHTML = '';

        sessions.forEach((session) => {
            const item = document.createElement('div');
            item.className = `session-item ${session.id === currentSessionId ? 'active' : ''}`;
            item.dataset.sessionId = session.id;

            const titleDiv = document.createElement('div');
            titleDiv.className = 'session-item-title';
            titleDiv.textContent = session.title;

            const metaDiv = document.createElement('div');
            metaDiv.className = 'session-item-meta';
            metaDiv.textContent = `${session.message_count} messages • ${formatDate(session.updated_at)}`;

            item.appendChild(titleDiv);
            item.appendChild(metaDiv);

            item.addEventListener('click', () => switchSession(session.id));

            sessionList.appendChild(item);
        });
    } catch (error) {
        console.error('Failed to load sessions:', error);
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;

    // Less than 24 hours
    if (diff < 86400000) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    // Less than 7 days
    if (diff < 604800000) {
        return date.toLocaleDateString([], { weekday: 'short' });
    }

    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

async function switchSession(sessionId) {
    if (sessionId === currentSessionId) {
        closeSidebarOnMobile();
        return;
    }

    // Disconnect current WebSocket
    if (ws) {
        ws.close();
        ws = null;
    }

    // Clear messages
    messagesContainer.innerHTML = '<div class="welcome-message"><p>Loading conversation...</p></div>';
    hideAllIndicators();
    updateContextStats({ messages: 0, tokens: 0 });

    // Update active state in sidebar
    document.querySelectorAll('.session-item').forEach((item) => {
        item.classList.toggle('active', item.dataset.sessionId === sessionId);
    });

    // Connect to new session
    connectWebSocket(sessionId);
    closeSidebarOnMobile();
}

async function createNewSession() {
    try {
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: 'New Conversation' })
        });

        const session = await response.json();
        await loadSessionList();
        switchSession(session.id);
    } catch (error) {
        console.error('Failed to create session:', error);
    }
}

async function clearCurrentConversation() {
    if (!currentSessionId) return;

    if (!confirm('Clear this conversation? This cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch(`/api/sessions/${currentSessionId}/clear`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (response.ok) {
            // Clear UI
            messagesContainer.innerHTML = '<div class="welcome-message"><p>Conversation cleared.</p></div>';
            hideAllIndicators();
            updateContextStats({ messages: 0, tokens: 0 });
            // Reload session to get empty message list
            connectWebSocket(currentSessionId);
        } else {
            alert('Failed to clear conversation');
        }
    } catch (error) {
        console.error('Failed to clear conversation:', error);
        alert('Failed to clear conversation');
    }
}

// ========== Sidebar Toggle ==========

function toggleSidebar() {
    sidebar.classList.toggle('open');
    if (sidebarOverlay) {
        sidebarOverlay.classList.toggle('open', sidebar.classList.contains('open'));
    }
}

function closeSidebarOnMobile() {
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('open');
        if (sidebarOverlay) {
            sidebarOverlay.classList.remove('open');
        }
    }
}

// ========== Input Handling ==========

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

async function sendUserMessage() {
    const content = messageInput.value.trim();
    if (!content || !isConnected) return;

    // Disable input while sending
    messageInput.value = '';
    messageInput.style.height = 'auto';
    btnSend.disabled = true;

    // Show typing indicator
    showIndicator('chatter', 'typing');

    // Send message
    sendMessage('message', content);

    // Add user message to UI immediately
    addMessage(content, 'user');

    // Re-enable input after a short delay (response comes via WebSocket)
    setTimeout(() => {
        btnSend.disabled = false;
        messageInput.focus();
    }, 500);
}

// ========== Event Listeners ==========

btnMenu.addEventListener('click', toggleSidebar);
btnNewChat.addEventListener('click', createNewSession);
btnClear.addEventListener('click', clearCurrentConversation);

messageInput.addEventListener('input', autoResizeTextarea);

messageInput.addEventListener('keydown', (e) => {
    // Send on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendUserMessage();
    }
});

btnSend.addEventListener('click', sendUserMessage);

// Close sidebar when clicking outside on mobile
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768 &&
        sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== btnMenu) {
        closeSidebarOnMobile();
    }
});

// Close sidebar when clicking overlay
if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeSidebarOnMobile);
}

// ========== Initialize ==========

async function init() {
    // Load session list
    await loadSessionList();

    // Get or create default session
    try {
        const response = await fetch('/api/sessions');
        const sessions = await response.json();

        if (sessions.length > 0) {
            // Connect to most recent session
            switchSession(sessions[0].id);
        } else {
            // Create first session
            await createNewSession();
        }
    } catch (error) {
        console.error('Failed to initialize:', error);
    }
}

// Start the app
init();