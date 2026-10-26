const $ = id => document.getElementById(id);
const state = { ready: false, busy: false, loading: false, messages: [], controller: null };
const welcomeTemplate = $('welcome').cloneNode(true);
const signed = value => value > 0 ? `+${value.toFixed(1)}` : value < 0 ? `−${Math.abs(value).toFixed(1)}` : '0.0';

function notice(message = '') {
  $('notice').textContent = message;
  $('notice').hidden = !message;
}

function controls() {
  const enabled = $('enabled').checked;
  $('value-one').textContent = signed(Number($('dose-one').value));
  $('value-two').textContent = signed(Number($('dose-two').value));
  for (const id of ['dose-one', 'dose-two', 'enabled', 'reset-doses', 'length', 'new-chat']) $(id).disabled = state.busy;
  for (const id of ['neuron-one', 'neuron-two']) $(id).style.opacity = enabled ? '1' : '.45';
  $('dose-one').disabled = state.busy || !enabled;
  $('dose-two').disabled = state.busy || !enabled;
  $('mode-label').replaceChildren();
  const dot = document.createElement('span');
  dot.className = `dot${enabled ? '' : ' inactive'}`;
  $('mode-label').append(dot, document.createTextNode(enabled ? 'Steering on' : 'Baseline · steering off'));
  $('send').disabled = !state.busy && (!state.ready || !$('prompt').value.trim());
  $('send').textContent = state.busy ? '■' : '↑';
  $('send').classList.toggle('stopping', state.busy);
  $('send').setAttribute('aria-label', state.busy ? 'Stop generation' : 'Send message');
  $('composer-hint').textContent = state.busy ? 'Qwen is replying…' : state.ready ? 'Local model. Your conversation stays here.' : 'Load the model to start chatting.';
}

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Please check your settings and try again.');
  return body;
}

function applyStatus(status) {
  state.ready = status.ready;
  state.loading = status.phase === 'loading';
  if (document.activeElement !== $('model-path') && !$('model-path').dataset.edited) $('model-path').value = status.model_dir;
  $('status-dot').classList.toggle('inactive', !state.ready);
  $('model-status').textContent = state.ready ? `Ready · ${status.device === 'mps' ? 'Apple Silicon' : status.device.toUpperCase()}` : state.loading ? 'Loading and calibrating…' : status.weights_present ? 'Weights found. Ready to load.' : 'Weights not downloaded · 9.3 GB';
  $('disk-space').textContent = `${status.free_gib} GiB free on the selected disk.`;
  $('load-model').textContent = state.ready ? 'Model ready' : state.loading ? 'Loading…' : 'Load model';
  $('load-model').disabled = state.ready || state.loading;
  controls();
}

async function refreshStatus() {
  try { applyStatus(await api('/api/status')); }
  catch { $('model-status').textContent = 'Server unavailable'; state.ready = false; controls(); }
}

function addMessage(role, text, waiting = false) {
  $('welcome')?.remove();
  const row = document.createElement('article');
  row.className = `message ${role}${waiting ? ' waiting' : ''}`;
  const label = document.createElement('div');
  label.className = 'message-role';
  label.textContent = role === 'user' ? 'YOU' : $('enabled').checked ? 'QWEN · STEERED' : 'QWEN · BASELINE';
  const content = document.createElement('div');
  content.className = 'message-content';
  content.textContent = text;
  row.append(label, content);
  $('messages').append(row);
  $('messages').scrollTop = $('messages').scrollHeight;
  return { row, content };
}

async function send(event) {
  event.preventDefault();
  if (state.busy) {
    try { await api('/api/stop', {}); } catch (error) { notice(error.message); }
    return;
  }
  const prompt = $('prompt').value.trim();
  if (!prompt || !state.ready) return;
  notice();
  state.busy = true;
  const previous = [...state.messages];
  state.messages.push({ role: 'user', content: prompt });
  const userRow = addMessage('user', prompt);
  const reply = addMessage('assistant', '', true);
  $('prompt').value = '';
  $('prompt').style.height = 'auto';
  state.controller = new AbortController();
  controls();
  let answer = '';
  let streamError = '';
  let finished = false;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, signal: state.controller.signal,
      body: JSON.stringify({messages: state.messages, enabled: $('enabled').checked, doses: [Number($('dose-one').value), Number($('dose-two').value)], max_new_tokens: Number($('length').value)})
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Could not send this message.');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const item = JSON.parse(line);
        if (item.error) streamError = item.error;
        if (item.text) {
          answer += item.text;
          reply.row.classList.remove('waiting');
          reply.content.textContent = answer;
          $('messages').scrollTop = $('messages').scrollHeight;
        }
        if (item.done) finished = true;
      }
    }
    if (streamError) throw new Error(streamError);
    if (!finished) throw new Error('The connection ended before the reply finished.');
    if (answer.trim()) state.messages.push({ role: 'assistant', content: answer });
    else { state.messages = previous; reply.row.remove(); userRow.row.remove(); $('prompt').value = prompt; notice('No reply was generated. Try again.'); }
  } catch (error) {
    state.messages = previous;
    reply.row.remove();
    userRow.row.remove();
    $('prompt').value = prompt;
    notice(error.message);
  } finally {
    reply.row.classList.remove('waiting');
    state.busy = false;
    state.controller = null;
    controls();
    $('prompt').focus();
  }
}

$('composer').addEventListener('submit', send);
$('prompt').addEventListener('input', () => { $('prompt').style.height = 'auto'; $('prompt').style.height = `${Math.min($('prompt').scrollHeight, 140)}px`; controls(); });
$('prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!state.busy) $('composer').requestSubmit(); } });
for (const id of ['dose-one', 'dose-two', 'enabled']) $(id).addEventListener('input', controls);
$('reset-doses').addEventListener('click', () => { $('dose-one').value = 1; $('dose-two').value = -1; controls(); });
$('new-chat').addEventListener('click', () => { state.messages = []; $('messages').replaceChildren(welcomeTemplate.cloneNode(true)); notice(); $('prompt').focus(); });
$('model-path').addEventListener('input', () => { $('model-path').dataset.edited = 'true'; });
$('load-model').addEventListener('click', async () => {
  notice(); $('load-model').disabled = true; $('load-model').textContent = 'Loading…';
  try { applyStatus(await api('/api/load', {model_dir: $('model-path').value})); }
  catch (error) { notice(error.message); $('setup').open = true; await refreshStatus(); }
});
$('messages').addEventListener('click', event => { const button = event.target.closest('[data-prompt]'); if (button) { $('prompt').value = button.dataset.prompt; controls(); $('prompt').focus(); } });
window.addEventListener('beforeunload', () => state.controller?.abort());
controls();
refreshStatus();
setInterval(() => { if (!state.busy && !document.hidden) refreshStatus(); }, 4000);
