"""
Assignment 11 - User Interface (Gradio) for the chat-based RAG system.

Dark three-column layout matching project/Interface_Mockup:
    [ Chat ]   [ Top-K retrieval : Articles | Videos ]   [ System inspection ]

Tasks:
  1  Chat interface + state management  (gr.Chatbot, system-prompt display, context
     display, text input, send via button + Enter, gr.State history; a turn updates the
     history, clears the input, updates the context + token indicator).
  2  System inspection: read-only system prompt, total token usage (budget bar), injected
     context, retrieval settings, latency -- auto-updated.
  3  Video panel: a video player + one retrieved clip per instruction step (in step order);
     clicking a segment loads its clip, seeks to the timestamp and autoplays. Video content
     is never in the prompt.

Run:  python app.py    
"""

import warnings
from pathlib import Path
from urllib.parse import quote
from html import escape

from starlette.exceptions import StarletteDeprecationWarning

# gradio's routes.py still reads the old (renamed) starlette status constant on every request;
# harmless, but noisy -- silence just that one warning class rather than all DeprecationWarnings.
warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)

import gradio as gr

from rag_backend import RagBackend

backend = RagBackend()

GREETING = ('Hello! I can answer how-to questions by retrieving the most relevant articles '
            'and video clips from the knowledge base. Try asking something like '
            '"How do I change a car tire?" or "How do I fix a leaky faucet?"')


# --------------------------------------------------------------------------- helpers
def fmt_time(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def pct_cls(p):
    return "teal" if p >= 85 else ("orange" if p >= 65 else "muted")


def render_articles(articles):
    if not articles:
        return "<div class='rhead'>TOP-5 · COSINE SIMILARITY</div><div class='empty'>Ask a question to retrieve articles.</div>"
    rows = ["<div class='rhead'>TOP-5 · COSINE SIMILARITY</div>"]
    for a in articles:
        p = round(a["score"] * 100)
        cls = pct_cls(p)
        rows.append(
            f"<div class='rcard'><div class='rnum'>{a['rank']}</div><div class='rbody'>"
            f"<div class='rtop'><span class='rtitle'>{escape(a['title'])}</span>"
            f"<span class='rpct {cls}'>{p}%</span></div>"
            f"<div class='rid'>id: {escape(str(a['id']))}</div>"
            f"<div class='rbar'><div class='rfill {cls}' style='width:{p}%'></div></div>"
            f"</div></div>")
    return f"<div class='rlist'>{''.join(rows)}</div>"


def render_videos(videos):
    if not videos:
        return "<div class='rhead'>ONE CLIP PER INSTRUCTION STEP</div><div class='empty'>Ask a question to retrieve video segments.</div>"
    rows = ["<div class='rhead'>ONE CLIP PER INSTRUCTION STEP</div>"]
    for v in videos:
        p = round(v["score"] * 100)
        cls = pct_cls(p)
        rows.append(
            f"<div class='rcard vcard' data-vidx='{v['rank'] - 1}' title='Click to play'>"
            f"<div class='rnum'>{v['rank']}</div>"
            f"<div class='vthumb'>▶<span class='vdur'>{fmt_time(v['end_time'])}</span></div>"
            f"<div class='rbody'>"
            f"<div class='rstep'>Step {v['rank']}: {escape(v.get('step', ''))}</div>"
            f"<div class='rtitle'>{escape(v['title'])}</div>"
            f"<div class='rid'>id: {escape(str(v['video_id']))}</div>"
            f"<div class='vtime'>{fmt_time(v['start_time'])}–{fmt_time(v['end_time'])}</div>"
            f"<div class='rbar'><div class='rfill {cls}' style='width:{p}%'></div></div>"
            f"<div class='rpct {cls} vpct'>{p}%</div>"
            f"</div></div>")
    return f"<div class='rlist'>{''.join(rows)}</div>"


def render_tabs(active):
    """middle-column header: two clickable tab labels (custom, so all 3 headers align)."""
    a = "active" if active == "articles" else ""
    v = "active" if active == "videos" else ""
    return (f"<div class='col-head tabhead'>"
            f"<span class='tab {a}' data-tab='articles'>Articles</span>"
            f"<span class='tab {v}' data-tab='videos'>Videos</span></div>")


def render_inspection(articles, videos, prompt_tokens, latency):
    used, total = prompt_tokens, backend.max_context_tokens
    ratio = min(1.0, used / total)
    bcls = "teal" if ratio < 0.75 else ("orange" if ratio < 0.95 else "red")

    art_items = "".join(
        f"<div class='inj'><div class='inj-t'><span>› {escape(a['title'])}</span>"
        f"<span class='tok'>{a['tokens']}t</span></div><div class='inj-id'>id: {escape(str(a['id']))}</div></div>"
        for a in articles) or "<div class='inj-empty'>—</div>"
    seen_ids, uniq_videos = set(), []
    for v in videos:
        if v["video_id"] not in seen_ids:
            seen_ids.add(v["video_id"])
            uniq_videos.append(v)
    vid_items = "".join(
        f"<div class='inj'><div class='inj-t'><span>› {escape(v['title'])}</span>"
        f"<span class='tok'>{backend.count_tokens(v['title'])}t</span></div>"
        f"<div class='inj-id'>id: {escape(str(v['video_id']))}</div></div>"
        for v in uniq_videos) or "<div class='inj-empty'>—</div>"

    settings = "".join(
        f"<div class='kv'><span>{escape(k)}</span><span class='val'>{escape(str(v))}</span></div>"
        for k, v in backend.settings.items())

    lat = []
    for k, v in latency.items():
        val = f"{v*1000:.0f}ms" if v < 1 else f"{v:.2f}s"
        lat.append(f"<div class='kv'><span>{escape(k)}</span><span class='val'>{val}</span></div>")

    return f"""
    <div class='insp'>
      <div class='sec'>CONTEXT SIZE</div>
      <div class='kv'><span>Token budget</span><span class='val'><b>{used:,}</b> / {total:,}</span></div>
      <div class='rbar big'><div class='rfill {bcls}' style='width:{ratio*100:.1f}%'></div></div>
      <div class='budget-sub'><span>0</span><span>{round(ratio*100)}% used</span><span>{total:,}</span></div>

      <div class='sec'>INJECTED INTO CONTEXT</div>
      <div class='sub'>Articles</div>{art_items}
      <div class='sub'>Videos</div>{vid_items}

      <div class='sec'>RETRIEVAL SETTINGS</div>{settings}

      <div class='sec'>LATENCY BREAKDOWN</div>{''.join(lat)}
    </div>"""


def video_player(video_id, start, title):
    path = Path(backend.video_dir) / f"{video_id}.mp4"
    cap = f"<div class='vp-cap'>▶ {escape(title)} · id {escape(str(video_id))} · seek {fmt_time(start)}</div>"
    if path.exists():
        url = f"/gradio_api/file={quote(str(path.resolve()))}#t={start:.1f}"
        return (f"<div class='vp'><video src='{url}' width='100%' controls autoplay muted "
                f"playsinline></video>{cap}</div>")
    return (f"<div class='vp vp-empty'><div class='vp-box'><div class='vp-play'>▶</div>"
            f"<div>Clip <code>{escape(str(video_id))}.mp4</code> loads here, seeks to "
            f"<b>{fmt_time(start)}</b> and autoplays.</div></div>{cap}</div>")


# --------------------------------------------------------------------------- handlers
def on_user(message, history):
    """Step 1: append the user's message and clear the input right away."""
    message = (message or "").strip()
    if not message:
        return gr.update(), gr.update(), history
    history = history + [{"role": "user", "content": message}]
    return "", history, history            # clear input, show the message in the chat


def on_bot(history):
    """Step 2: generate the grounded answer and refresh the retrieval + inspection panels."""
    if not history or history[-1].get("role") != "user":
        return (gr.update(),) * 6
    out = backend.answer(history[:-1], history[-1]["content"])
    history = history + [{"role": "assistant", "content": out["answer"]}]
    return (history, history,
            render_articles(out["articles"]),
            render_videos(out["videos"]),
            render_inspection(out["articles"], out["videos"], out["prompt_tokens"], out["latency"]),
            out["videos"])


def on_select(idx, videos):
    try:
        v = videos[int(str(idx).split(":")[0])]   # bridge sends "index:counter"
    except (ValueError, TypeError, IndexError):
        return gr.update()
    return video_player(v["video_id"], v["start_time"], v["title"])


def on_tab(sel):
    """switch the middle column between the Articles and Videos panels."""
    active = str(sel).split(":")[0]
    if active not in ("articles", "videos"):
        return gr.update(), gr.update(), gr.update()
    return (render_tabs(active),
            gr.update(visible=active == "articles"),
            gr.update(visible=active == "videos"))


# JS on load: force dark mode + delegate video-card clicks to the hidden bridge textbox.
BRIDGE_JS = """
() => {
  ['.gradio-container', 'gradio-app', 'body', 'html'].forEach(s => {
    const el = document.querySelector(s); if (el) el.classList.add('dark');
  });
  let n = 0;
  const setNative = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  };
  document.addEventListener('click', (e) => {
    const card = e.target.closest('[data-vidx]');
    if (card) {
      const ta = document.querySelector('#vsel textarea');
      if (ta) setNative(ta, card.getAttribute('data-vidx') + ':' + (n++));
      return;
    }
    const tab = e.target.closest('[data-tab]');
    if (tab) {
      const ts = document.querySelector('#tsel textarea');
      if (ts) setNative(ts, tab.getAttribute('data-tab') + ':' + (n++));
    }
  });
}
"""

CSS = f"""
:root{{--bg:#0a0e14;--panel:#0d131c;--panel2:#0f1620;--bd:#1c2634;--teal:#2dd4bf;
--orange:#f59e0b;--red:#ef4444;--tx:#dbe3ec;--mut:#8291a3;--mut2:#5f6e80;}}
.gradio-container,.gradio-container .main,body,gradio-app{{background:var(--bg)!important;}}
.gradio-container{{max-width:100%!important;color:var(--tx)!important;font-size:13px;padding:0 8px;}}
/* three-column grid: flush columns split by vertical divider lines */
#mainrow{{gap:0!important;flex-wrap:nowrap!important;align-items:stretch!important;}}
#chatcol,#mid,#inspcol{{padding:0 20px!important;box-sizing:border-box;
  align-self:stretch;justify-content:flex-start;}}
#chatcol,#mid{{border-right:1px solid var(--bd);}}
/* keep the chat window a fixed height so messages don't get pushed down by column stretch */
#chatbox{{flex:0 0 auto!important;}}
.col-head{{color:var(--teal);font-weight:600;letter-spacing:.08em;font-size:13px;text-transform:uppercase;
  height:48px;display:flex;align-items:center;padding:0 20px;border-bottom:1px solid var(--bd);
  margin:0 -20px 14px;box-sizing:border-box;}}
.hidden,#vsel,#tsel{{display:none!important;}}
.form:has(#vsel),.form:has(#tsel){{display:none!important;}}  /* kill the bridges' form line */
/* chat bubbles */
#chatbox{{background:transparent!important;border:none!important;}}
#chatbox .message{{background:var(--panel)!important;border:1px solid var(--bd)!important;
  color:var(--tx)!important;border-radius:12px!important;box-shadow:none!important;
  font-size:13px!important;line-height:1.55!important;padding:12px 15px!important;}}
#chatbox .message .message{{background:transparent!important;border:none!important;
  border-radius:0!important;padding:0!important;}}  /* kill Gradio's nested double box */
#chatbox .user-row .message,#chatbox .user{{border-color:#1f3b39!important;}}
/* hide Gradio's chatbot chrome (hover toolbar, per-message copy/like buttons) */
#chatbox .icon-button-wrapper,#chatbox .message-buttons,#chatbox .message-buttons-bot,
#chatbox .message-buttons-user,#chatbox button[title],#chatbox .copy-button,
#chatbox .like-dislike{{display:none!important;}}
/* input bar: one rounded box with an inline paper-plane */
#inputbar{{background:transparent!important;border:none!important;padding:0!important;
  margin-top:12px;gap:10px;align-items:center;flex-wrap:nowrap!important;}}
#inputbar #msg{{flex:1;min-width:0;}}
#inputbar input,#inputbar textarea{{background:var(--panel2)!important;color:var(--tx)!important;
  border:1px solid var(--bd)!important;border-radius:12px!important;box-shadow:none!important;
  resize:none;font-size:14px!important;padding:13px 14px!important;height:48px!important;}}
#sendbtn{{height:48px!important;width:52px!important;min-width:52px!important;flex:none;
  padding:0!important;border:none!important;border-radius:11px!important;font-size:18px;line-height:1;
  box-shadow:0 3px 12px rgba(45,212,191,.30)!important;transition:filter .15s,transform .15s;}}
#sendbtn:hover{{filter:brightness(1.08);transform:translateY(-1px);}}
#sendbtn:active{{transform:translateY(0);}}
#sysbox textarea{{background:var(--panel2)!important;color:var(--mut)!important;border:1px solid var(--bd)!important;
  font-size:11px;line-height:1.4;}}
/* middle-column custom tab header -- identical height/baseline to the .col-head cells */
.tabhead{{justify-content:center;gap:34px;padding:0;}}
.tabhead .tab{{height:48px;display:flex;align-items:center;cursor:pointer;color:var(--mut);
  font-weight:600;letter-spacing:.08em;text-transform:uppercase;font-size:13px;
  transition:color .15s;}}
.tabhead .tab:hover{{color:var(--tx);}}
.tabhead .tab.active{{color:var(--teal);box-shadow:inset 0 -2px 0 var(--teal);}}
/* retrieval cards */
.rlist{{display:flex;flex-direction:column;}}
.rhead{{color:var(--mut2);font-size:11px;letter-spacing:.12em;padding:2px 0 10px;}}
.rcard{{display:flex;gap:11px;background:transparent;padding:13px 0;border-bottom:1px solid #131b25;}}
.vcard{{cursor:pointer;border-radius:8px;}}
.vcard:hover{{background:var(--panel);}}
.rnum{{color:var(--mut2);font-size:13px;min-width:14px;}}
.rbody{{flex:1;min-width:0;}}
.rtop{{display:flex;justify-content:space-between;gap:8px;align-items:baseline;}}
.rstep{{color:var(--teal);font-size:11px;font-weight:600;margin-bottom:3px;}}
.rtitle{{color:var(--tx);font-weight:600;font-size:13px;}}
.rid{{color:var(--mut2);font-size:11px;font-family:ui-monospace,monospace;margin:3px 0 9px;}}
.rpct{{font-size:12px;font-weight:600;white-space:nowrap;}}
.vpct{{display:inline-block;margin-top:5px;}}
.rpct.teal,.rfill.teal{{color:var(--teal);}}
.rpct.orange,.rfill.orange{{color:var(--orange);}}
.rpct.muted{{color:var(--mut);}}
.rbar{{height:3px;background:#141c27;border-radius:3px;overflow:hidden;}}
.rbar.big{{height:7px;margin:8px 0 4px;}}
.rfill{{height:100%;border-radius:3px;background:var(--mut2);}}
.rfill.teal{{background:var(--teal);}}
.rfill.orange{{background:var(--orange);}}
.rfill.red{{background:var(--red);}}
.vthumb{{position:relative;width:76px;height:50px;flex:none;border-radius:6px;
  background:linear-gradient(135deg,#1b2836,#0e1620);display:flex;align-items:center;
  justify-content:center;color:var(--tx);font-size:14px;border:1px solid var(--bd);}}
.vdur{{position:absolute;right:3px;bottom:3px;background:#000a;color:#fff;font-size:9px;
  padding:1px 4px;border-radius:3px;}}
.vtime{{color:var(--mut);font-size:11px;margin-bottom:7px;}}
/* inspection */
.insp{{display:flex;flex-direction:column;gap:2px;}}
.sec{{color:var(--mut2);font-size:11px;letter-spacing:.12em;margin:16px 0 8px;}}
.sec:first-child{{margin-top:2px;}}
.kv{{display:flex;justify-content:space-between;gap:8px;padding:4px 0;color:var(--mut);font-size:12px;}}
.kv .val{{color:var(--tx);font-family:ui-monospace,monospace;}}
.kv.total{{border-top:1px solid var(--bd);margin-top:4px;padding-top:8px;color:var(--tx);}}
.budget-sub{{display:flex;justify-content:space-between;color:var(--mut2);font-size:10px;margin-bottom:2px;}}
.sub{{color:var(--mut);font-size:11px;margin:10px 0 5px;}}
.inj{{background:var(--panel);border:1px solid var(--bd);border-radius:8px;padding:9px 11px;margin-bottom:7px;}}
.inj-t{{display:flex;justify-content:space-between;gap:8px;color:var(--tx);font-size:12px;}}
.inj-t .tok{{color:var(--mut2);font-size:11px;font-family:ui-monospace,monospace;}}
.inj-id{{color:var(--mut2);font-size:10px;font-family:ui-monospace,monospace;margin-top:3px;}}
.inj-empty{{color:var(--mut2);padding:4px 2px;}}
/* video player */
.vp{{margin-bottom:14px;}}
.vp video{{display:block;border-radius:8px;background:#000;max-height:230px;}}
.vp-cap{{color:var(--mut);font-size:11px;margin-top:6px;}}
.vp-empty .vp-box{{border:1px dashed var(--bd);border-radius:8px;padding:18px 14px;text-align:center;
  color:var(--mut);display:flex;flex-direction:column;gap:8px;align-items:center;background:var(--panel);}}
.vp-play{{width:42px;height:42px;border-radius:50%;background:var(--panel2);border:1px solid var(--bd);
  display:flex;align-items:center;justify-content:center;color:var(--teal);font-size:16px;}}
.empty{{color:var(--mut2);padding:16px 4px;font-size:12px;}}
/* never show Gradio's "processing | Xs" overlay on top of the panels */
.progress-text,.meta-text,.meta-text-center{{display:none!important;}}
footer{{display:none!important;}}
"""

with gr.Blocks(title="How-To RAG Assistant") as demo:
    history = gr.State([{"role": "assistant", "content": GREETING}])
    videos_state = gr.State([])
    bridge = gr.Textbox(elem_id="vsel", elem_classes="hidden")   # video-card click -> Python
    tab_bridge = gr.Textbox(elem_id="tsel", elem_classes="hidden")  # tab click -> Python

    with gr.Row(elem_id="mainrow"):
        # ---- Chat (Task 1) ----
        with gr.Column(scale=7, elem_id="chatcol"):
            gr.HTML("<div class='col-head'>Chat</div>")
            chatbot = gr.Chatbot(value=history.value, elem_id="chatbox", height=520,
                                 show_label=False)
            with gr.Row(elem_id="inputbar"):
                # lines=max_lines=1 -> a true single-line box so Enter submits (a multi-line
                # textarea would insert a newline on Enter instead of firing .submit)
                msg = gr.Textbox(placeholder="Ask a how-to question…", show_label=False,
                                 elem_id="msg", container=False, lines=1, max_lines=1,
                                 autofocus=True, scale=1)
                send = gr.Button("➤", elem_id="sendbtn", variant="primary", scale=0, min_width=52)
            with gr.Accordion("Active system prompt (read-only)", open=False):
                gr.Textbox(backend.system_prompt, show_label=False, interactive=False,
                           lines=4, elem_id="sysbox", container=False)

        # ---- Top-K retrieval (Task 3 playlist) ----
        with gr.Column(scale=4, elem_id="mid", min_width=0):
            tabhead = gr.HTML(render_tabs("articles"))
            with gr.Column(elem_id="pane-articles") as articles_pane:
                articles_html = gr.HTML(render_articles([]))
            with gr.Column(elem_id="pane-videos", visible=False) as videos_pane:
                player = gr.HTML(video_player("—", 0.0, "no clip selected"))
                videos_html = gr.HTML(render_videos([]))

        # ---- System inspection (Task 2) ----
        with gr.Column(scale=5, elem_id="inspcol", min_width=0):
            gr.HTML("<div class='col-head'>System inspection</div>")
            inspection_html = gr.HTML(render_inspection(
                [], [], 0, {"LLM generation": 0}))

    # Two-step submit: on_user clears the input + shows the message instantly, then on_bot
    # generates. show_progress="minimal" avoids Gradio's "processing | Xs" overlay on the panels.
    bot_out = [chatbot, history, articles_html, videos_html, inspection_html, videos_state]
    for trigger in (msg.submit, send.click):                                  # Enter key + button
        trigger(on_user, [msg, history], [msg, chatbot, history], show_progress="hidden").then(
            on_bot, history, bot_out, show_progress="minimal")
    bridge.change(on_select, [bridge, videos_state], [player], show_progress="hidden")
    tab_bridge.change(on_tab, [tab_bridge], [tabhead, articles_pane, videos_pane], show_progress="hidden")
    demo.load(None, None, None, js=BRIDGE_JS)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue=gr.themes.colors.teal), css=CSS,
                allowed_paths=[str(Path(backend.video_dir).resolve())])
