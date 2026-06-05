const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const promptEl = document.querySelector("#prompt");
const sendButton = document.querySelector("#sendButton");
const resetButton = document.querySelector("#resetButton");
const modelLine = document.querySelector("#modelLine");
const systemPrompt = document.querySelector("#systemPrompt");
const contextMode = document.querySelector("#contextMode");
const temperature = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperatureValue");
const maxTokens = document.querySelector("#maxTokens");
const topK = document.querySelector("#topK");
const deviceValue = document.querySelector("#deviceValue");
const stepValue = document.querySelector("#stepValue");
const speedValue = document.querySelector("#speedValue");

let conversation = [];
let isGenerating = false;

function render() {
  messagesEl.innerHTML = "";
  if (conversation.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Start a conversation with the local checkpoint. The visible transcript stays here while the context setting controls what is sent to the model.";
    messagesEl.appendChild(empty);
    return;
  }

  for (const message of conversation) {
    messagesEl.appendChild(createMessageElement(message));
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function createMessageElement(message) {
  const row = document.createElement("article");
  row.className = `message ${message.role}`;
  const role = document.createElement("div");
  role.className = "role";
  role.textContent = message.role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message.content;
  row.append(role, bubble);
  return row;
}

function addStatusMessage(text, kind = "pending") {
  const row = createMessageElement({ role: kind === "error" ? "error" : "model", content: text });
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

function requestMessages(userText) {
  const messages = [];
  const system = systemPrompt.value.trim();
  if (system) {
    messages.push({ role: "system", content: system });
  }
  let prior = [];
  if (contextMode.value === "recent") {
    prior = conversation.slice(-4);
  } else if (contextMode.value === "full") {
    prior = conversation.slice();
  }
  while (prior.length > 0 && prior[0].role !== "user") {
    prior.shift();
  }
  messages.push(...prior);
  messages.push({ role: "user", content: userText });
  return messages;
}

async function sendMessage(text) {
  if (isGenerating) return;
  isGenerating = true;
  sendButton.disabled = true;
  resetButton.disabled = true;
  const payloadMessages = requestMessages(text);
  const assistantMessage = { role: "assistant", content: "" };
  conversation.push({ role: "user", content: text }, assistantMessage);
  render();
  const startedAt = performance.now();

  try {
    const response = await fetch("/api/chat_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: payloadMessages,
        max_new_tokens: Number(maxTokens.value),
        temperature: Number(temperature.value),
        top_k: Number(topK.value),
      }),
    });
    if (!response.ok || !response.body) {
      throw new Error("Generation failed");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalMessage = null;
    let lastUsage = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const eventText of events) {
        const line = eventText.split("\n").find((item) => item.startsWith("data: "));
        if (!line) continue;
        const data = JSON.parse(line.slice(6));
        if (data.type === "error") {
          throw new Error(data.error || "Generation failed");
        }
        if (data.type === "token") {
          assistantMessage.content = data.content || "";
          updateLastAssistantBubble(assistantMessage.content);
        }
        if (data.type === "done") {
          finalMessage = data.message;
          lastUsage = data.usage;
        }
      }
    }

    if (!finalMessage) {
      const content = assistantMessage.content.trim();
      finalMessage = { role: "assistant", content };
    }
    assistantMessage.content = finalMessage.content;
    const speed = lastUsage?.tokens_per_second;
    speedValue.textContent = speed ? `${speed.toFixed(1)} tok/s` : "...";
    render();
  } catch (error) {
    assistantMessage.role = "error";
    assistantMessage.content = error.message;
    render();
  } finally {
    const elapsed = Math.max((performance.now() - startedAt) / 1000, 0.001);
    if (speedValue.textContent === "...") {
      speedValue.textContent = `${elapsed.toFixed(1)}s`;
    }
    isGenerating = false;
    sendButton.disabled = false;
    resetButton.disabled = false;
    promptEl.focus();
  }
}

function updateLastAssistantBubble(content) {
  const rows = messagesEl.querySelectorAll(".message.assistant");
  const row = rows[rows.length - 1];
  const bubble = row?.querySelector(".bubble");
  if (bubble) {
    bubble.textContent = content;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = "";
  promptEl.style.height = "auto";
  sendMessage(text);
});

promptEl.addEventListener("input", () => {
  promptEl.style.height = "auto";
  promptEl.style.height = `${Math.min(promptEl.scrollHeight, 180)}px`;
});

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

resetButton.addEventListener("click", () => {
  conversation = [];
  speedValue.textContent = "...";
  render();
  promptEl.focus();
});

temperature.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperature.value).toFixed(1);
});

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!data.ok) throw new Error("status failed");
    const model = data.model;
    modelLine.textContent = model.repo_id;
    deviceValue.textContent = model.device;
    stepValue.textContent = model.step ?? "...";
  } catch (error) {
    modelLine.textContent = "Model status unavailable";
  }
}

render();
loadStatus();
