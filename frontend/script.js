// API base URL - use relative path to work from any host
const API_URL = '/api'

// Global state
let currentSessionId = null

// DOM elements
let chatMessages, chatInput, sendButton, totalDocuments, documentTitles, themeToggle

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  // Get DOM elements after page loads
  chatMessages = document.getElementById('chatMessages')
  chatInput = document.getElementById('chatInput')
  sendButton = document.getElementById('sendButton')
  totalDocuments = document.getElementById('totalDocuments')
  documentTitles = document.getElementById('documentTitles')
  themeToggle = document.getElementById('themeToggle')

  setupEventListeners()
  initializeTheme()
  createNewSession()
  loadDocumentStats()

  // New Chat button logic
  const newChatButton = document.getElementById('newChatButton')
  if (newChatButton) {
    newChatButton.addEventListener('click', async () => {
      try {
        // Call backend to create new session and clear previous one
        const response = await fetch(`${API_URL}/session/new`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prev_session_id: currentSessionId }),
        })
        if (!response.ok) throw new Error('Failed to start new chat')
        const data = await response.json()
        currentSessionId = data.session_id
        chatMessages.innerHTML = ''
        addMessage(
          'Welcome to the Document Materials Assistant! I can help you with questions about documents, sections and specific content. What would you like to know?',
          'assistant',
          null,
          true
        )
        chatInput.value = ''
        chatInput.disabled = false
        sendButton.disabled = false
        chatInput.focus()
      } catch (error) {
        chatMessages.innerHTML = ''
        addMessage('Error: Could not start a new chat session.', 'assistant')
      }
    })
  }
})

// Event Listeners
function setupEventListeners() {
  // Chat functionality
  sendButton.addEventListener('click', sendMessage)
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage()
  })

  // Theme toggle functionality
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme)
    // Support keyboard navigation
    themeToggle.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        toggleTheme()
      }
    })
  }

  // Suggested questions
  document.querySelectorAll('.suggested-item').forEach((button) => {
    button.addEventListener('click', (e) => {
      const question = e.target.getAttribute('data-question')
      chatInput.value = question
      sendMessage()
    })
  })
}

// Chat Functions
async function sendMessage() {
  const query = chatInput.value.trim()
  if (!query) return

  // Disable input
  chatInput.value = ''
  chatInput.disabled = true
  sendButton.disabled = true

  // Add user message
  addMessage(query, 'user')

  // Add loading message - create a unique container for it
  const loadingMessage = createLoadingMessage()
  chatMessages.appendChild(loadingMessage)
  chatMessages.scrollTop = chatMessages.scrollHeight

  try {
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        session_id: currentSessionId,
      }),
    })

    if (!response.ok) throw new Error('Query failed')

    const data = await response.json()

    // Update session ID if new
    if (!currentSessionId) {
      currentSessionId = data.session_id
    }

    // Replace loading message with response
    loadingMessage.remove()
    addMessage(data.answer, 'assistant', data.sources)
  } catch (error) {
    // Replace loading message with error
    loadingMessage.remove()
    addMessage(`Error: ${error.message}`, 'assistant')
  } finally {
    chatInput.disabled = false
    sendButton.disabled = false
    chatInput.focus()
  }
}

function createLoadingMessage() {
  const messageDiv = document.createElement('div')
  messageDiv.className = 'message assistant'
  messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `
  return messageDiv
}

function addMessage(content, type, sources = null, isWelcome = false) {
  const messageId = Date.now()
  const messageDiv = document.createElement('div')
  messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`
  messageDiv.id = `message-${messageId}`

  // Convert markdown to HTML for assistant messages
  const displayContent =
    type === 'assistant' ? marked.parse(content) : escapeHtml(content)

  let html = `<div class="message-content">${displayContent}</div>`

  if (sources && sources.length > 0) {
    // Process sources to handle both string and object formats
    const processedSources = sources.map(source => {
      if (typeof source === 'object' && source.text && source.link) {
        // Source with embedded link
        return `<a href="${source.link}" target="_blank" rel="noopener noreferrer">${source.text}</a>`
      } else if (typeof source === 'string') {
        // Legacy string-only source
        return source
      }
      return source
    })

    html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources</summary>
                <div class="sources-content">${processedSources.join(', ')}</div>
            </details>
        `
  }

  messageDiv.innerHTML = html
  chatMessages.appendChild(messageDiv)
  chatMessages.scrollTop = chatMessages.scrollHeight

  return messageId
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
  currentSessionId = null
  chatMessages.innerHTML = ''
  addMessage(
    'Welcome to the Document Materials Assistant! I can help you with questions about documents, sections and specific content. What would you like to know?',
    'assistant',
    null,
    true
  )
}

// Load document statistics
async function loadDocumentStats() {
  try {
    console.log('Loading document stats...')
    const response = await fetch(`${API_URL}/documents`)
    if (!response.ok) throw new Error('Failed to load document stats')

    const data = await response.json()
    console.log('Document data received:', data)

    // Update stats in UI
    if (totalDocuments) {
      totalDocuments.textContent = data.total_documents
    }

    // Update document titles
    if (documentTitles) {
      if (data.document_titles && data.document_titles.length > 0) {
        documentTitles.innerHTML = data.document_titles
          .map((title) => `<div class="document-title-item">${title}</div>`)
          .join('')
      } else {
        documentTitles.innerHTML =
          '<span class="no-documents">No documents available</span>'
      }
    }
  } catch (error) {
    console.error('Error loading document stats:', error)
    // Set default values on error
    if (totalDocuments) {
      totalDocuments.textContent = '0'
    }
    if (documentTitles) {
      documentTitles.innerHTML =
        '<span class="error">Failed to load documents</span>'
    }
  }
}

// Theme Functions
function initializeTheme() {
  // Check for saved theme preference or default to dark
  const savedTheme = localStorage.getItem('theme') || 'dark'
  setTheme(savedTheme)
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark'
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark'
  setTheme(newTheme)
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme)
  localStorage.setItem('theme', theme)
  
  // Update button aria-label for accessibility
  if (themeToggle) {
    themeToggle.setAttribute('aria-label', 
      theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'
    )
  }
}
