"""
Chess Opening Trainer - FastAPI Backend
Single file serving static frontend + JSON APIs
"""
import json
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="Chess Opening Trainer")

# Paths
DB_PATH = Path("studies.db")
WEAKNESSES_PATH = Path("weaknesses.json")
STATS_PATH = Path("stats.json")


def get_db():
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# API Routes
# =============================================================================

@app.get("/api/studies")
async def get_studies(fen: str = Query(..., description="FEN position (piece placement only)")):
    """Get study annotations for a position."""
    if not DB_PATH.exists():
        return {"moves": [], "error": "Studies database not found"}

    # Normalize FEN to just piece placement
    fen_prefix = fen.split()[0] if " " in fen else fen

    conn = get_db()
    rows = conn.execute("""
        SELECT p.move_san, p.comment_text, p.arrows, p.squares, p.full_line, p.ply,
               p.is_sideline, p.is_mainline, p.has_annotation,
               s.name as study_name, s.url as study_url, p.chapter_name
        FROM positions p
        JOIN studies s ON p.study_id = s.id
        WHERE p.fen = ?
        ORDER BY p.is_mainline DESC, p.has_annotation DESC, LENGTH(p.comment_text) DESC
    """, (fen_prefix,)).fetchall()
    conn.close()

    moves = []
    for r in rows:
        moves.append({
            "move": r["move_san"],
            "comment": r["comment_text"] or "",
            "arrows": json.loads(r["arrows"]) if r["arrows"] else [],
            "squares": json.loads(r["squares"]) if r["squares"] else [],
            "line": r["full_line"],
            "ply": r["ply"],
            "study": r["study_name"],
            "study_url": r["study_url"],
            "chapter": r["chapter_name"],
            "is_sideline": bool(r["is_sideline"]),
            "is_mainline": bool(r["is_mainline"]),
            "has_annotation": bool(r["has_annotation"])
        })

    return {"moves": moves, "count": len(moves)}


@app.get("/api/weaknesses")
async def get_weaknesses(color: Optional[str] = Query(None, description="Filter by 'white' or 'black'")):
    """Get precomputed weakness analysis."""
    if not WEAKNESSES_PATH.exists():
        return {"weaknesses": [], "error": "Weaknesses file not found. Run analyze.py first."}

    data = json.loads(WEAKNESSES_PATH.read_text())
    weaknesses = data.get("weaknesses", [])

    # Filter by color if specified
    if color:
        is_white = color.lower() == "white"
        weaknesses = [w for w in weaknesses if w.get("isWhite") == is_white]

    return {"weaknesses": weaknesses, "count": len(weaknesses)}


@app.get("/api/stats")
async def get_stats():
    """Get aggregated game statistics."""
    if not STATS_PATH.exists():
        return {"error": "Stats file not found"}

    return json.loads(STATS_PATH.read_text())


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "studies_db": DB_PATH.exists(),
        "weaknesses": WEAKNESSES_PATH.exists(),
        "stats": STATS_PATH.exists()
    }


# =============================================================================
# Legacy JSON file routes (for backward compatibility during migration)
# =============================================================================

@app.get("/weaknesses.json")
async def weaknesses_json():
    """Serve weaknesses.json for backward compatibility."""
    if not WEAKNESSES_PATH.exists():
        return JSONResponse({"weaknesses": []}, status_code=404)
    return JSONResponse(json.loads(WEAKNESSES_PATH.read_text()))


@app.get("/stats.json")
async def stats_json():
    """Serve stats.json for backward compatibility."""
    if not STATS_PATH.exists():
        return JSONResponse({}, status_code=404)
    return JSONResponse(json.loads(STATS_PATH.read_text()))


@app.get("/study_moves.json")
async def study_moves_json():
    """Serve study_moves.json for backward compatibility."""
    path = Path("study_moves.json")
    if not path.exists():
        return JSONResponse({"positions": {}, "total_positions": 0, "total_annotations": 0})
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/study/{study_id}/pgn")
async def get_study_pgn(study_id: str):
    """Get raw PGN for a study (for tree navigation)."""
    if not DB_PATH.exists():
        return JSONResponse({"error": "Database not found"}, status_code=404)

    conn = get_db()
    row = conn.execute(
        "SELECT name, url, raw_pgn FROM studies WHERE id = ?",
        (study_id,)
    ).fetchone()
    conn.close()

    if not row:
        return JSONResponse({"error": "Study not found"}, status_code=404)

    return {
        "id": study_id,
        "name": row["name"],
        "url": row["url"],
        "pgn": row["raw_pgn"]
    }


# =============================================================================
# Static Files (must be last - catch-all)
# =============================================================================

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# =============================================================================
# Run with: uvicorn app:app --reload --port 8000
# =============================================================================
