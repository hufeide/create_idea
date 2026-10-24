"""FastAPI server: serves the canvas UI, exposes a WebSocket control channel,
and REST endpoints for inspecting each egg's worktree (review code)."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.responses import FileResponse

from .graph import Canvas
from .egg import reasoning_capture_state

FRONTEND = Path(__file__).resolve().parent.parent / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    canvas = Canvas()
    await canvas.start()
    app.state.canvas = canvas
    yield
    await canvas.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(FRONTEND)


@app.get("/api/egg/{egg_id}/files")
async def list_files(egg_id: str):
    canvas: Canvas = app.state.canvas
    egg = canvas.eggs.get(egg_id)
    if not egg:
        raise HTTPException(404)
    return {"files": egg.list_files()}


@app.get("/api/egg/{egg_id}/file")
async def read_file(egg_id: str, path: str = Query(...)):
    canvas: Canvas = app.state.canvas
    egg = canvas.eggs.get(egg_id)
    if not egg:
        raise HTTPException(404)
    try:
        content = egg.read_file(path)
    except ValueError:
        raise HTTPException(400, "invalid path")
    if content is None:
        raise HTTPException(404)
    return {"path": path, "content": content}


@app.get("/api/trace/commander")
async def trace_commander():
    """Chronological coordinator trace for the Commander.

    Declared BEFORE the parameterized route so 'commander' is not captured as an
    egg id by /api/trace/{egg_id}.
    """
    canvas: Canvas = app.state.canvas
    return {
        "connected": canvas.commander._agent is not None,
        "transcript": getattr(canvas.commander, "transcript", []),
    }


@app.get("/api/trace/{egg_id}")
async def trace_egg(egg_id: str):
    """Chronological conversation trace for one egg (user/model/tool records)."""
    canvas: Canvas = app.state.canvas
    egg = canvas.eggs.get(egg_id)
    if not egg:
        raise HTTPException(404)
    return {
        "eggId": egg_id,
        "title": egg.title,
        "connected": egg._agent is not None,
        "transcript": getattr(egg, "transcript", []),
    }


@app.post("/api/interrupt")
async def interrupt_egg(egg_id: str = Query(...)):
    """Interrupt a running egg's in-flight model turn (the stop button).

    Uses HTTP (not the WS channel) because the WS read-loop is busy streaming
    while an egg runs, so an interrupt must arrive on a separate request.
    """
    canvas: Canvas = app.state.canvas
    egg = canvas.eggs.get(egg_id)
    if not egg:
        raise HTTPException(404)
    egg.interrupt()
    return {"ok": True, "eggId": egg_id, "busy": egg.busy}


@app.get("/api/sessions")
async def list_sessions():
    """List all saved canvas sessions (newest first)."""
    return {"sessions": Canvas.list_sessions()}


async def _stream_events(ws: WebSocket, agen):
    """Translate orchestration events into WS messages.

    Shared by the commander (advice/control) and the one-click run:
      say      -> stream into the commander chat (schedule / narration)
      instruct -> push an instruction into an egg's chat
      egg      -> stream an egg's reply tokens
      end_egg  -> finalize that egg's stream
      end      -> finalize the commander stream
    """
    async for ev in agen:
        k = ev.get("k")
        if k == "say":
            await ws.send_json({"type": "token", "eggId": "commander", "text": ev["text"]})
        elif k == "instruct":
            await ws.send_json({"type": "msg", "eggId": ev["egg"], "role": "user",
                                "text": ev["text"]})
        elif k == "egg":
            await ws.send_json({"type": "token", "eggId": ev["egg"], "text": ev["text"]})
        elif k == "status":
            await ws.send_json({"type": "status", "eggId": ev["egg"],
                                "status": ev["status"]})
        elif k == "end_egg":
            await ws.send_json({"type": "done", "eggId": ev["egg"]})
        elif k == "end":
            await ws.send_json({"type": "done", "eggId": "commander"})


async def _dispatch(msg: dict, ws: WebSocket, canvas: Canvas):
    t = msg.get("type")

    if t == "sync":
        await ws.send_json({"type": "eggs", **canvas.view()})

    elif t == "generate":
        created = await canvas.generate(msg.get("goal", ""))
        await ws.send_json({"type": "eggs", **canvas.view()})
        for e in created:
            await ws.send_json(
                {"type": "system", "eggId": e.id,
                 "text": "（已根据目标由 LLM 生成该蛋的提示词）"}
            )
        # The generate opened a NEW dated session; refresh the picker + mark active.
        await ws.send_json({"type": "sessions", "sessions": Canvas.list_sessions()})
        await ws.send_json({"type": "session", "sessionId": canvas.session_id})

    elif t == "addEgg":
        await canvas.add_egg(msg.get("title", "新蛋"), msg.get("prompt", ""))
        await ws.send_json({"type": "eggs", **canvas.view()})

    elif t == "command":
        eid = msg["eggId"]
        egg = canvas.eggs.get(eid)
        if not egg:
            return

        async def produce():
            await ws.send_json({"type": "status", "eggId": eid, "status": "running"})
            gen = egg.run(msg["text"])
            try:
                async for tok in gen:
                    await ws.send_json({"type": "token", "eggId": eid, "text": tok})
            except asyncio.CancelledError:
                try:
                    await ws.send_json({"type": "system", "eggId": eid,
                                        "text": "⏹ 已打断运行。"})
                except Exception:  # pragma: no cover
                    pass
            finally:
                # aclose releases the egg's locks (box + per-egg) immediately.
                try:
                    await gen.aclose()
                except Exception:  # pragma: no cover
                    pass
                try:
                    await ws.send_json({"type": "status", "eggId": eid, "status": "idle"})
                    await ws.send_json({"type": "done", "eggId": eid})
                    canvas.save()
                except Exception:  # pragma: no cover
                    pass

        # Run in a dedicated task so the stop button (egg.interrupt) cancels only
        # this run, never the websocket connection itself.
        await asyncio.create_task(produce())

    elif t == "interrupt":
        egg = canvas.eggs.get(msg.get("eggId"))
        if egg:
            egg.interrupt()

    elif t == "listSessions":
        await ws.send_json({"type": "sessions", "sessions": Canvas.list_sessions()})

    elif t == "loadSession":
        sid = msg.get("sessionId")
        if sid:
            try:
                view = await canvas.load(sid)
                await ws.send_json({"type": "eggs", **view})
                # Replay each egg's (and the commander's) stored transcript so the
                # UI shows the prior conversation. The SDK resumes the LLM session
                # on the next message via the saved conversation_id + save_dir.
                for e in canvas.eggs.values():
                    await ws.send_json({"type": "history", "eggId": e.id,
                                        "transcript": e.transcript})
                await ws.send_json({"type": "history", "eggId": "commander",
                                    "transcript": canvas.commander.transcript})
            except Exception as ex:
                await ws.send_json({"type": "error", "message": f"load failed: {ex}"})

    elif t == "newSession":
        # Persist the current session (if any) then open a fresh empty session.
        try:
            canvas.save()
        except Exception:
            pass
        for e in list(canvas.eggs.values()):
            await e.stop()
        canvas.eggs.clear()
        canvas.edges.clear()
        sid = canvas._new_session()
        await ws.send_json({"type": "eggs", **canvas.view()})
        await ws.send_json({"type": "sessions", "sessions": Canvas.list_sessions()})
        await ws.send_json({"type": "session", "sessionId": sid})

    elif t == "compress":
        egg = canvas.eggs.get(msg["eggId"])
        if not egg:
            return

        async def do_compress():
            await ws.send_json({"type": "status", "eggId": egg.id, "status": "running"})
            try:
                summary = await egg.compress()
                await ws.send_json({"type": "compressed", "eggId": egg.id, "summary": summary})
            except asyncio.CancelledError:
                try:
                    await ws.send_json({"type": "system", "eggId": egg.id,
                                        "text": "⏹ 已取消压缩。"})
                except Exception:  # pragma: no cover
                    pass
            finally:
                try:
                    await ws.send_json({"type": "status", "eggId": egg.id, "status": "idle"})
                    canvas.save()
                except Exception:  # pragma: no cover
                    pass

        # Run in a child task so the stop button cancels only this compress,
        # never the websocket connection.
        await asyncio.create_task(do_compress())

    elif t == "combine":
        new = await canvas.combine(msg.get("ids", []))
        await ws.send_json({"type": "eggs", **canvas.view()})
        if new:
            await ws.send_json(
                {"type": "system", "eggId": new.id,
                 "text": "（已将所选蛋组合为新蛋）"}
            )

    elif t == "deleteEgg":
        await canvas.delete_egg(msg["id"])
        await ws.send_json({"type": "eggs", **canvas.view()})

    elif t == "deleteEdge":
        await canvas.delete_edge(msg["id"])
        await ws.send_json({"type": "eggs", **canvas.view()})

    elif t == "addEdge":
        await canvas.add_edge(msg["from"], msg["to"])
        await ws.send_json({"type": "eggs", **canvas.view()})

    elif t == "move":
        await canvas.move(msg["id"], msg["x"], msg["y"])

    elif t == "commander":
        targets = msg.get("targets") or (
            [msg.get("target")] if msg.get("target") else []
        )
        await _stream_events(ws, canvas.commander_run(msg.get("text", ""), targets))
        try:
            canvas.save()
        except Exception:
            pass

    elif t == "run":
        await _stream_events(ws, canvas.run_selected(msg.get("targets") or []))
        try:
            canvas.save()   # persist any recovered conversation ids, etc.
        except Exception:
            pass

    elif t == "setHistoryMode":
        mode = msg.get("mode")
        if mode in ("brief", "full"):
            canvas.history_mode = mode
            await ws.send_json({"type": "historyMode", "mode": canvas.history_mode})

    elif t == "setReasoning":
        from .egg import set_reasoning_capture
        set_reasoning_capture(bool(msg.get("on")))
        await ws.send_json({"type": "reasoning", **reasoning_capture_state()})

    else:
        await ws.send_json({"type": "error", "message": f"unknown type: {t}"})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    canvas: Canvas = app.state.canvas
    try:
        while True:
            msg = await websocket.receive_json()
            try:
                await _dispatch(msg, websocket, canvas)
            except Exception as ex:  # keep the socket alive on per-message errors
                await websocket.send_json({"type": "error", "message": str(ex)})
    except Exception as ex:  # client disconnected
        print(f"[ws] closed: {ex}")
