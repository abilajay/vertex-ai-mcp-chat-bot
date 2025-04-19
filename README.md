# Gemini MCP Client with Tool Integration

This project demonstrates an interactive chat assistant powered by Google Gemini (via Vertex AI) integrated with external tools through an MCP (Model Control Protocol) server using a Node.js backend. The assistant can execute real-time function calls, maintain session memory, and provide friendly, context-aware responses.

## ✨ Features

- ✅ Interactive assistant using Gemini 2.0 Flash model.
- 🛠️ Tool calling via MCP (e.g., browser navigation, search).
- 🧠 Session memory to maintain context across conversations.
- 🔁 Handles multi-step tool calls and provides clear status updates.
- 🔍 Built-in system prompt to guide assistant behavior.

---

## 📦 Requirements

- Python 3.8+
- Node.js installed
- Google Cloud Vertex AI enabled project
- `@playwright/mcp` module installed (globally or locally)

---

## 🔧 Setup

### 1. Clone this repository

```bash
git clone https://github.com/your-repo/gemini-mcp-chat.git
cd gemini-mcp-chat
