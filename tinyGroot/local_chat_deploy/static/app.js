const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const promptEl = document.querySelector("#prompt");
const sendButton = document.querySelector("#sendButton");
const stopButton = document.querySelector("#stopButton");
const resetButton = document.querySelector("#resetButton");
const settingsToggle = document.querySelector("#settingsToggle");
const sidePane = document.querySelector("#sidePane");
const modelLine = document.querySelector("#modelLine");
const systemPrompt = document.querySelector("#systemPrompt");
const contextMode = document.querySelector("#contextMode");
const temperature = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperatureValue");
const maxTokens = document.querySelector("#maxTokens");
const topK = document.querySelector("#topK");
const deviceValue = document.querySelector("#deviceValue");
const deviceBadge = document.querySelector("#deviceBadge");
const stepValue = document.querySelector("#stepValue");
const speedValue = document.querySelector("#speedValue");

const SUGGESTIONS = [
  "What is 17 * 24?",
  "Write a haiku about autumn.",
  "Explain gravity to a five year old.",
];

let conversation = [];
let isGenerating = false;
let abortController = null;

/* ---------------- rendering ---------------- */

function escapeHtml(text) {
  return text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function renderMarkdown(raw) {
  const blocks = [];
  // Pull out fenced code blocks before escaping so their contents are verbatim.
  let text = raw.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, _lang, code) => {
    const i = blocks.length;
    blocks.push(`<pre><code>${escapeHtml(code.replace(/\n+$/, ""))}</code></pre>`);
    return `\u0000${i}\u0000`;
  });
  text = escapeHtml(text);
  text = text.replace(/`([^`\n]+)`/g, (_m, c) => `<code>${c}</code>`);
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  const lines = text.split("\n");
  const out = [];
  let list = null; // "ul" | "ol"
  let para = [];

  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${para.join("<br>")}</p>`);
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };

  for (const line of lines) {
    const codePlaceholder = /^\u0000\d+\u0000$/.test(line.trim());
    if (codePlaceholder) {
      flushPara();
      flushList();
      out.push(line.trim());
      continue;
    }
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+\.\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const want = ul ? "ul" : "ol";
      if (list !== want) {
        flushList();
        out.push(`<${want}>`);
        list = want;
      }
      out.push(`<li>${(ul || ol)[1]}</li>`);
      continue;
    }
    flushList();
    if (line.trim() === "") {
      flushPara();
    } else {
      para.push(line);
    }
  }
  flushPara();
  flushList();

  let html = out.join("");
  html = html.replace(/\u0000(\d+)\u0000/g, (_m, i) => blocks[Number(i)]);
  return html;
}

function avatarFor(role) {
  if (role === "user") return "you";
  if (role === "error") return "!";
  return "🌱";
}

function createMessageElement(message) {
  const row = document.createElement("article");
  row.className = `message ${message.role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = avatarFor(message.role);

  const main = document.createElement("div");
  main.className = "message-main";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  setBubbleContent(bubble, message);

  const meta = document.createElement("div");
  meta.className = "meta-row";
  populateMeta(meta, message);

  main.append(bubble, meta);
  row.append(avatar, main);
  return row;
}

function setBubbleContent(bubble, message, { streaming = false } = {}) {
  if (message.role === "assistant" && !message.content) {
    bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    return;
  }
  if (message.role === "assistant") {
    bubble.innerHTML = renderMarkdown(message.content) + (streaming ? '<span class="cursor"></span>' : "");
  } else {
    bubble.textContent = message.content;
  }
}

function populateMeta(meta, message) {
  meta.innerHTML = "";
  if (message.role === "assistant" && message.content) {
    const copy = document.createElement("button");
    copy.className = "msg-action";
    copy.type = "button";
    copy.textContent = "Copy";
    copy.addEventListener("click", () => {
      navigator.clipboard?.writeText(message.content);
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1200);
    });
    meta.appendChild(copy);

    const isLast = conversation[conversation.length - 1] === message;
    if (isLast) {
      const regen = document.createElement("button");
      regen.className = "msg-action";
      regen.type = "button";
      regen.textContent = "Regenerate";
      regen.addEventListener("click", regenerate);
      meta.appendChild(regen);
    }
  }
  if (message.stat) {
    const stat = document.createElement("span");
    stat.className = "msg-stat";
    stat.textContent = message.stat;
    meta.appendChild(stat);
  }
}

function render() {
  messagesEl.innerHTML = "";
  if (conversation.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    const chips = SUGGESTIONS.map((s) => `<button type="button">${escapeHtml(s)}</button>`).join("");
    empty.innerHTML = `
      <h2>Chat with your local checkpoint</h2>
      <p>The transcript stays in your browser; the Context control decides what history is actually sent to the model.</p>
      <div class="suggestions">${chips}</div>`;
    empty.querySelectorAll(".suggestions button").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (isGenerating) return;
        sendMessage(btn.textContent);
      });
    });
    messagesEl.appendChild(empty);
    return;
  }
  for (const message of conversation) {
    messagesEl.appendChild(createMessageElement(message));
  }
  scrollToBottom();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------------- conversation logic ---------------- */

function requestMessages(userText) {
  const messages = [];
  const system = systemPrompt.value.trim();
  if (system) messages.push({ role: "system", content: system });

  let prior = [];
  if (contextMode.value === "recent") prior = conversation.slice(-4);
  else if (contextMode.value === "full") prior = conversation.slice();
  // Drop leading assistant/error turns and any failed turns so history stays valid.
  prior = prior.filter((m) => m.role === "user" || m.role === "assistant");
  while (prior.length && prior[0].role !== "user") prior.shift();

  messages.push(...prior);
  messages.push({ role: "user", content: userText });
  return messages;
}

function setGenerating(active) {
  isGenerating = active;
  sendButton.hidden = active;
  stopButton.hidden = !active;
  resetButton.disabled = active;
  promptEl.disabled = active;
}

async function sendMessage(text) {
  if (isGenerating) return;
  const userMessage = { role: "user", content: text };
  const assistantMessage = { role: "assistant", content: "", stat: null };
  conversation.push(userMessage, assistantMessage);
  render();
  await streamInto(assistantMessage, requestMessages(text));
}

async function regenerate() {
  if (isGenerating || conversation.length < 2) return;
  // Find the last user message and trim everything after it.
  let idx = conversation.length - 1;
  while (idx >= 0 && conversation[idx].role !== "user") idx -= 1;
  if (idx < 0) return;
  const userText = conversation[idx].content;
  conversation = conversation.slice(0, idx + 1);
  const assistantMessage = { role: "assistant", content: "", stat: null };
  conversation.push(assistantMessage);
  render();
  await streamInto(assistantMessage, requestMessages(userText));
}

async function streamInto(assistantMessage, payloadMessages) {
  setGenerating(true);
  abortController = new AbortController();
  const startedAt = performance.now();
  const row = messagesEl.querySelector(".message.assistant:last-child");
  const bubble = row?.querySelector(".bubble");
  const meta = row?.querySelector(".meta-row");

  try {
    const response = await fetch("/api/chat_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: abortController.signal,
      body: JSON.stringify({
        messages: payloadMessages,
        max_new_tokens: Number(maxTokens.value),
        temperature: Number(temperature.value),
        top_k: Number(topK.value),
      }),
    });
    if (!response.ok || !response.body) throw new Error(`Generation failed (HTTP ${response.status})`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let usage = null;
    let finished = false;

    // Stop on the `done` event itself; with keep-alive the connection may not
    // close, so waiting for reader EOF alone would hang and leave the UI locked.
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const eventText of events) {
        const line = eventText.split("\n").find((item) => item.startsWith("data: "));
        if (!line) continue;
        const data = JSON.parse(line.slice(6));
        if (data.type === "error") throw new Error(data.error || "Generation failed");
        if (data.type === "token") {
          assistantMessage.content = data.content || "";
          if (bubble) setBubbleContent(bubble, assistantMessage, { streaming: true });
          scrollToBottom();
        }
        if (data.type === "done") {
          assistantMessage.content = data.message?.content ?? assistantMessage.content;
          usage = data.usage;
          finished = true;
        }
      }
    }
    await reader.cancel().catch(() => {});

    finalizeStats(assistantMessage, usage, startedAt);
    if (bubble) setBubbleContent(bubble, assistantMessage);
    if (meta) populateMeta(meta, assistantMessage);
  } catch (error) {
    if (error.name === "AbortError") {
      // Keep whatever streamed so far; mark it as stopped.
      if (!assistantMessage.content) assistantMessage.content = "_(stopped)_";
      assistantMessage.stat = "stopped";
      if (bubble) setBubbleContent(bubble, assistantMessage);
      if (meta) populateMeta(meta, assistantMessage);
    } else {
      assistantMessage.role = "error";
      assistantMessage.content = error.message;
      render();
    }
  } finally {
    abortController = null;
    setGenerating(false);
    promptEl.focus();
  }
}

function finalizeStats(message, usage, startedAt) {
  const seconds = usage?.seconds ?? (performance.now() - startedAt) / 1000;
  const tps = usage?.tokens_per_second;
  if (tps) {
    speedValue.textContent = `${tps.toFixed(1)} tok/s`;
    message.stat = `${usage.completion_tokens} tok · ${tps.toFixed(1)} tok/s`;
  } else {
    speedValue.textContent = `${seconds.toFixed(1)}s`;
    message.stat = `${seconds.toFixed(1)}s`;
  }
}

/* ---------------- events ---------------- */

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = promptEl.value.trim();
  if (!text || isGenerating) return;
  promptEl.value = "";
  promptEl.style.height = "auto";
  sendMessage(text);
});

stopButton.addEventListener("click", () => {
  abortController?.abort();
});

promptEl.addEventListener("input", () => {
  promptEl.style.height = "auto";
  promptEl.style.height = `${Math.min(promptEl.scrollHeight, 200)}px`;
});

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

resetButton.addEventListener("click", () => {
  if (isGenerating) return;
  conversation = [];
  speedValue.textContent = "…";
  render();
  promptEl.focus();
});

settingsToggle.addEventListener("click", () => {
  const collapsed = sidePane.classList.toggle("collapsed");
  settingsToggle.setAttribute("aria-expanded", String(!collapsed));
});

temperature.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperature.value).toFixed(1);
});

/* ---------------- status ---------------- */

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!data.ok) throw new Error("status failed");
    const model = data.model;
    modelLine.textContent = model.repo_id;
    modelLine.title = model.repo_id;
    deviceValue.textContent = model.device;
    stepValue.textContent = model.step ?? "—";
    if (model.device) {
      deviceBadge.textContent = model.device;
      deviceBadge.hidden = false;
    }
  } catch (error) {
    modelLine.textContent = "Model status unavailable";
  }
}

// Collapse settings by default on narrow screens.
if (window.matchMedia("(max-width: 900px)").matches) {
  sidePane.classList.add("collapsed");
}

render();
loadStatus();
