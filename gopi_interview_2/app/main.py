"""Entry point.

    python -m app.main            # interactive CLI chat
    python -m app.main --serve    # minimal web UI + JSON API on http://127.0.0.1:8000

JSON API (all state lives in the backend session, the page only mirrors it):
    POST /chat            {message, session_id, edit_turn_id?} -> {reply, messages}
    GET  /history?session_id=...                               -> {messages}
    POST /session/clear   {session_id}                         -> {ok, messages: []}
"""
import argparse
import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app import config, reservations
from app.agent import Agent, SessionBusy, TurnNotFound
from app.llm import get_llm
from app.rag import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        reservations.init_db()
        _agent = Agent(get_llm(), Retriever())
    return _agent


# --------------------------------------------------------------------------- web
app = FastAPI(title=f"{config.HOTEL_NAME} Assistant")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    edit_turn_id: str | None = None      # set when the message replaces an earlier user message


class SessionRequest(BaseModel):
    session_id: str = "default"


@app.post("/chat")
def chat(req: ChatRequest):
    agent = get_agent()
    try:
        result = agent.handle(req.message, req.session_id, req.edit_turn_id)
    except SessionBusy:
        return JSONResponse(status_code=409, content={
            "detail": "A reply is still being generated. Please wait for it to finish.",
            "messages": agent.transcript(req.session_id)})
    except TurnNotFound:
        return JSONResponse(status_code=404, content={
            "detail": "That message is no longer part of this conversation.",
            "messages": agent.transcript(req.session_id)})
    if result.error == "stale":
        return JSONResponse(status_code=409, content={
            "detail": "The conversation changed while the reply was being generated.",
            "messages": result.messages})
    if result.error:
        return JSONResponse(status_code=503, content={"detail": result.reply, "messages": result.messages})
    return {"reply": result.reply, "messages": result.messages}


@app.get("/history")
def history(session_id: str = "default") -> dict:
    return {"messages": get_agent().transcript(session_id)}


@app.post("/session/clear")
def clear_session(req: SessionRequest) -> dict:
    get_agent().reset(req.session_id)
    return {"ok": True, "messages": []}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__HOTEL__ Assistant</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;background:#f6f8fa;color:#222}
 header{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:.75rem}
 header h2{margin:0;font-size:1.25rem}
 #clear{display:inline-flex;align-items:center;gap:.35rem;padding:.4rem .7rem;border:1px solid #ccc;border-radius:6px;background:#fff;color:#444;cursor:pointer;font-size:.9rem;white-space:nowrap}
 #clear:hover{background:#fbeaea;border-color:#b33;color:#b33}
 #log{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1rem;min-height:320px;max-height:60vh;overflow-y:auto}
 .msg{position:relative;margin:.5rem 0;padding:.5rem 2.4rem .5rem .6rem;border-radius:6px;white-space:pre-wrap;word-break:break-word}
 .msg.u{color:#1a4d8f;background:#eef3fb}
 .msg.a{color:#222;margin-bottom:1rem}
 .msg .who{font-weight:600;margin-right:.3rem}
 .edit{position:absolute;top:.3rem;right:.3rem;border:none;background:transparent;color:#1a4d8f;cursor:pointer;font-size:1rem;padding:.15rem .35rem;border-radius:4px;opacity:0;transition:opacity .15s}
 .msg.u:hover .edit,.msg.u:focus-within .edit,.msg.u.show-actions .edit{opacity:1}
 .edit:hover{background:#dce6f7}
 .edit:disabled{cursor:not-allowed;color:#999}
 @media (hover:none){.edit{opacity:.35}}
 .typing{color:#666;font-style:italic;margin:.5rem 0}
 .typing .dot{display:inline-block;width:6px;height:6px;margin-left:3px;border-radius:50%;background:#888;animation:blink 1.2s infinite}
 .typing .dot:nth-child(2){animation-delay:.2s}.typing .dot:nth-child(3){animation-delay:.4s}
 @keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
 .error{color:#b00020;background:#fdecee;border:1px solid #f5c2c7;border-radius:6px;padding:.5rem .7rem;margin:.5rem 0}
 .empty{color:#888;text-align:center;margin-top:6rem}
 #editbar{display:none;align-items:center;gap:.5rem;margin-top:.75rem;padding:.4rem .7rem;background:#fff7e0;border:1px solid #f0d58c;border-radius:6px;font-size:.9rem}
 #editbar.on{display:flex}
 #editbar button{margin-left:auto;padding:.25rem .6rem}
 form{display:flex;gap:.5rem;margin-top:.5rem}
 input{flex:1;padding:.6rem;font-size:1rem} button{padding:.6rem 1rem}
 input:disabled,button:disabled{opacity:.6;cursor:not-allowed}
 #confirm{border:none;border-radius:8px;padding:1.25rem;max-width:320px;box-shadow:0 8px 30px rgba(0,0,0,.2)}
 #confirm::backdrop{background:rgba(0,0,0,.35)}
 #confirm p{margin:0 0 1rem;font-size:1.05rem}
 #confirm menu{display:flex;justify-content:flex-end;gap:.5rem;margin:0;padding:0}
 #confirm .danger{background:#b33;color:#fff;border:1px solid #922;border-radius:4px}
</style></head><body>
<header>
  <h2>__HOTEL__ - Reservation Assistant</h2>
  <button id="clear" type="button" title="Clear conversation" aria-label="Clear conversation">&#128465; Clear</button>
</header>
<div id="log" aria-live="polite"></div>
<div id="editbar"><span>&#9998; Editing your message. Sending it replaces the original and everything after it.</span><button id="cancel-edit" type="button">Cancel</button></div>
<form id="f" autocomplete="off">
  <input id="m" autocomplete="off" placeholder="Ask about the hotel or manage your reservation..." />
  <button id="send" type="submit">Send</button>
</form>
<dialog id="confirm" aria-labelledby="confirm-text">
  <p id="confirm-text">Clear this conversation?</p>
  <menu><button id="confirm-cancel" type="button">Cancel</button><button id="confirm-ok" type="button" class="danger">Clear</button></menu>
</dialog>
<script>
(() => {
  const DEFAULT_PLACEHOLDER = 'Ask about the hotel or manage your reservation...';
  const $ = (id) => document.getElementById(id);
  const logEl = $('log'), form = $('f'), input = $('m'), sendBtn = $('send'), editBar = $('editbar'), dialog = $('confirm');
  const isTouch = window.matchMedia('(pointer: coarse)').matches;   // phones/tablets: don't pop the keyboard automatically

  // Single source of truth for the page. `messages` mirrors the backend session transcript.
  const state = {
    sid: sessionStorage.sid || (sessionStorage.sid = crypto.randomUUID()),
    messages: [],          // [{turn_id, role: 'user'|'assistant', content}]
    isLoading: false,      // true from Send until the complete reply has been added (or failed)
    editingTurnId: null,   // turn_id of the user message being edited, if any
    error: null,
    requestSeq: 0,         // only the response of the latest request may touch the state
    controller: null,      // AbortController of the in-flight request
  };
  window.__chatState = state;   // handy for debugging / automated tests

  // ---- rendering -----------------------------------------------------------
  function render() {
    logEl.replaceChildren();
    if (!state.messages.length && !state.isLoading && !state.error) {
      const e = document.createElement('div'); e.className = 'empty';
      e.textContent = 'Ask a question about the hotel or manage your reservation.'; logEl.appendChild(e);
    }
    for (const m of state.messages) {
      const d = document.createElement('div');
      d.className = 'msg ' + (m.role === 'user' ? 'u' : 'a');
      d.dataset.turn = m.turn_id;
      const who = document.createElement('span'); who.className = 'who';
      who.textContent = m.role === 'user' ? 'You:' : 'Assistant:';
      const body = document.createElement('span'); body.className = 'text'; body.textContent = m.content;
      d.append(who, body);
      if (m.role === 'user') {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'edit'; b.title = 'Edit message'; b.setAttribute('aria-label', 'Edit message');
        b.textContent = '✎';
        b.disabled = state.isLoading;
        b.addEventListener('click', (ev) => { ev.stopPropagation(); startEdit(m); });
        d.appendChild(b);
        attachLongPress(d);
      }
      logEl.appendChild(d);
    }
    if (state.isLoading) {
      const t = document.createElement('div'); t.className = 'typing'; t.id = 'typing';
      t.append('Assistant is typing');
      for (let i = 0; i < 3; i++) { const s = document.createElement('span'); s.className = 'dot'; t.appendChild(s); }
      logEl.appendChild(t);
    }
    if (state.error) {
      const e = document.createElement('div'); e.className = 'error'; e.id = 'error'; e.textContent = state.error; logEl.appendChild(e);
    }
    logEl.scrollTop = logEl.scrollHeight;

    input.disabled = state.isLoading;
    sendBtn.disabled = state.isLoading;
    input.placeholder = state.editingTurnId ? 'Edit your message and press Send...' : DEFAULT_PLACEHOLDER;
    editBar.classList.toggle('on', !!state.editingTurnId);
  }

  // ---- mobile: long-press a message bubble to reveal its Edit action --------
  function attachLongPress(el) {
    let timer = null;
    const cancel = () => { if (timer) { clearTimeout(timer); timer = null; } };
    el.addEventListener('touchstart', () => { timer = setTimeout(() => el.classList.add('show-actions'), 450); }, { passive: true });
    el.addEventListener('touchend', cancel); el.addEventListener('touchmove', cancel); el.addEventListener('touchcancel', cancel);
  }
  document.addEventListener('touchstart', (ev) => {
    for (const el of logEl.querySelectorAll('.show-actions')) if (!el.contains(ev.target)) el.classList.remove('show-actions');
  }, { passive: true });

  // ---- edit ----------------------------------------------------------------
  function startEdit(m) {
    if (state.isLoading) return;                 // never edit while a reply is being generated
    state.editingTurnId = m.turn_id;
    state.error = null;
    input.value = m.content;
    render();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
  function cancelEdit() { state.editingTurnId = null; input.value = ''; render(); input.focus(); }
  $('cancel-edit').addEventListener('click', cancelEdit);
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && state.editingTurnId) { ev.preventDefault(); cancelEdit(); }
    if (ev.key === 'Enter' && state.isLoading) ev.preventDefault();     // input is disabled anyway; belt and braces
  });

  // ---- send (new message or edited message) --------------------------------
  form.addEventListener('submit', (ev) => { ev.preventDefault(); send(); });

  async function send() {
    if (state.isLoading) return;                 // blocks double-click, Enter spam and queued sends
    const text = input.value.trim();
    if (!text) return;
    const editId = state.editingTurnId;
    const previous = state.messages;             // restored if the request fails
    const seq = ++state.requestSeq;
    const controller = new AbortController();
    state.controller = controller;
    state.isLoading = true;                      // set synchronously, before any await
    state.error = null;
    state.editingTurnId = null;
    input.value = '';

    // Optimistic view: an edited message replaces the original and drops everything after it.
    const idx = editId ? previous.findIndex((m) => m.turn_id === editId) : -1;
    const base = idx >= 0 ? previous.slice(0, idx) : previous;
    state.messages = base.concat([{ turn_id: 'pending', role: 'user', content: text }]);
    render();

    try {
      const r = await fetch('/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ message: text, session_id: state.sid, edit_turn_id: editId }),
      });
      const data = await r.json().catch(() => ({}));
      if (seq !== state.requestSeq) return;      // superseded (cleared meanwhile): never touch the state
      if (r.ok && Array.isArray(data.messages)) {
        state.messages = data.messages;          // backend transcript is the source of truth
      } else {
        state.messages = Array.isArray(data.messages) ? data.messages : previous;
        state.error = typeof data.detail === 'string' ? data.detail : 'The assistant is temporarily unavailable. Please try again.';
        input.value = text;                      // let the user retry or fix the message
        if (editId && state.messages.some((m) => m.turn_id === editId)) state.editingTurnId = editId;
      }
    } catch (err) {
      if (seq !== state.requestSeq) return;      // aborted on purpose by Clear
      state.messages = previous;
      state.error = 'Could not reach the assistant. Check your connection and try again.';
      input.value = text;
      if (editId) state.editingTurnId = editId;
    } finally {
      if (seq === state.requestSeq) {            // only the current request may unlock the UI
        state.isLoading = false;
        state.controller = null;
        render();
        if (!isTouch) input.focus();
      }
    }
  }

  // ---- clear ---------------------------------------------------------------
  $('clear').addEventListener('click', () => dialog.showModal());
  $('confirm-cancel').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (ev) => { if (ev.target === dialog) dialog.close(); });   // click on backdrop = cancel
  $('confirm-ok').addEventListener('click', () => { dialog.close(); clearConversation(); });

  async function clearConversation() {
    state.requestSeq++;                          // any in-flight reply is now stale and will be ignored
    if (state.controller) { state.controller.abort(); state.controller = null; }
    const oldSid = state.sid;
    state.messages = []; state.isLoading = false; state.editingTurnId = null; state.error = null;
    input.value = '';
    state.sid = sessionStorage.sid = crypto.randomUUID();   // fresh backend session for whatever comes next
    render();
    input.focus();
    try {                                        // and reset the old backend session so its pending reply is dropped
      await fetch('/session/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                      body: JSON.stringify({ session_id: oldSid }) });
    } catch (_) { /* the old session is never used again either way */ }
  }

  // ---- restore this session's transcript on load ----------------------------
  render();
  (async () => {
    try {
      const r = await fetch('/history?session_id=' + encodeURIComponent(state.sid));
      const data = r.ok ? await r.json() : {};
      if (Array.isArray(data.messages) && !state.isLoading) { state.messages = data.messages; render(); }
    } catch (_) { /* start empty */ }
  })();
})();
</script></body></html>"""
_PAGE = _PAGE_TEMPLATE.replace("__HOTEL__", config.HOTEL_NAME)


# --------------------------------------------------------------------------- cli
def run_cli() -> None:
    agent = get_agent()
    print(f"{config.HOTEL_NAME} assistant. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        print(f"Assistant: {agent.handle(text).reply}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hotel reservation assistant")
    parser.add_argument("--serve", action="store_true", help="Run the web UI instead of the CLI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.serve:
        import uvicorn

        get_agent()  # fail fast (missing index / provider) before accepting requests
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
