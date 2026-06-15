const agent = new Worker('worker.js');

const log = document.getElementById('log');
const runBtn = document.getElementById('runBtn');
const userInput = document.getElementById('userInput');

let currentResponseContainer = null;

function appendTrace(content, className = '') {
  const div = document.createElement('div');
  div.className = `msg trace ${className}`.trim();
  div.innerText = content;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function appendChunk(content) {
  if (!currentResponseContainer) {
    currentResponseContainer = document.createElement('div');
    currentResponseContainer.className = 'msg agent-reply';
    currentResponseContainer.innerHTML = '<span class="badge">LLM Explainer</span>';
    log.appendChild(currentResponseContainer);
  }

  const span = document.createElement('span');
  span.innerText = content;
  currentResponseContainer.appendChild(span);
  log.scrollTop = log.scrollHeight;
}

runBtn.onclick = () => {
  runBtn.disabled = true;
  currentResponseContainer = null;

  appendTrace(`> USER: ${userInput.value}`, 'user-trace');
  agent.postMessage({ prompt: userInput.value });
};

agent.onmessage = (event) => {
  const { type, content } = event.data;

  if (type === 'trace') {
    appendTrace(content);
  }

  if (type === 'chunk') {
    appendChunk(content);
  }

  if (type === 'done') {
    runBtn.disabled = false;
    currentResponseContainer = null;
  }
};

agent.onerror = (error) => {
  appendTrace(`WORKER ERROR: ${error.message}`);
  runBtn.disabled = false;
  currentResponseContainer = null;
};
