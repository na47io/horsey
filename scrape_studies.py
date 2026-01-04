#!/usr/bin/env python3
"""
Scrape Lichess studies for opening explanations.
Downloads annotated PGNs and indexes them by position for RAG.

Usage:
    uv run python scrape_studies.py              # Download and parse studies
    uv run python scrape_studies.py --reprocess  # Re-parse stored PGNs (no download)

The --reprocess flag is useful when you've changed the parsing logic and want to
re-extract positions from already-downloaded PGNs. It's much faster since it skips
the network requests and just processes the raw_pgn column in the database.

Data flow:
    1. Search Lichess for studies matching your weakness openings
    2. Download full PGNs with annotations (stored in studies.raw_pgn)
    3. Parse all moves (mainline + sidelines) into positions table
    4. Export to study_moves.json for frontend consumption
"""

import json
import re
import sqlite3
import time
from pathlib import Path

import chess
import chess.pgn
import requests
from io import StringIO
from tqdm import tqdm

DB_PATH = Path("studies.db")
HEADERS = {"User-Agent": "ChessOpeningTrainer/1.0 (educational project)"}
RATE_LIMIT = 1.0  # seconds between requests

def init_db():
    """Initialize SQLite database for study index."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS studies (
            id TEXT PRIMARY KEY,
            name TEXT,
            owner TEXT,
            url TEXT,
            opening_query TEXT,
            raw_pgn TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY,
            fen TEXT,
            full_fen TEXT,
            study_id TEXT,
            chapter_name TEXT,
            move_san TEXT,
            comment_text TEXT,
            arrows TEXT,
            squares TEXT,
            full_line TEXT,
            ply INTEGER,
            is_sideline INTEGER DEFAULT 0,
            is_mainline INTEGER DEFAULT 0,
            has_annotation INTEGER DEFAULT 0,
            FOREIGN KEY (study_id) REFERENCES studies(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fen ON positions(fen)")
    conn.commit()
    return conn


def search_studies(query: str, max_pages: int = 3) -> list[dict]:
    """Search Lichess studies by query (scraping search page)."""
    studies = []
    seen_ids = set()

    for page in range(1, max_pages + 1):
        # URL encode the query
        encoded_query = query.replace(' ', '+')
        url = f"https://lichess.org/study/search?q={encoded_query}&page={page}"

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Connection error on page {page}: {e}")
            break

        if r.status_code != 200:
            print(f"Failed to fetch page {page}: {r.status_code}")
            break

        # Find all study IDs (8 char alphanumeric after /study/)
        study_ids = re.findall(r'/study/([a-zA-Z0-9]{8})', r.text)

        # Get unique IDs from this page
        new_ids = [sid for sid in study_ids if sid not in seen_ids]

        if not new_ids:
            break

        for study_id in new_ids:
            seen_ids.add(study_id)
            studies.append({
                'id': study_id,
                'name': f'Study {study_id}',  # Name fetched later from PGN headers
                'url': f'https://lichess.org/study/{study_id}'
            })

        time.sleep(RATE_LIMIT)

    return studies


def download_study_pgn(study_id: str, retries: int = 3) -> str:
    """Download full PGN of a study with annotations."""
    url = f"https://lichess.org/api/study/{study_id}.pgn?comments=true&variations=true"

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            # 403 = private study, 404 = deleted - silently skip
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            return None
        except Exception:
            return None
    return None


def parse_annotations(comment: str) -> dict:
    """Parse Lichess visual annotations from comment."""
    import re

    result = {
        'text': '',
        'arrows': [],  # [(from_sq, to_sq, color), ...]
        'squares': [], # [(square, color), ...]
    }

    # Extract colored arrows: [%cal Gf7f5,Rf2f4] -> green f7-f5, red f2-f4
    arrow_match = re.findall(r'\[%cal ([^\]]+)\]', comment)
    for match in arrow_match:
        for arrow in match.split(','):
            if len(arrow) >= 5:
                color = {'G': 'green', 'R': 'red', 'Y': 'yellow', 'B': 'blue'}.get(arrow[0], 'green')
                from_sq = arrow[1:3]
                to_sq = arrow[3:5]
                result['arrows'].append({'from': from_sq, 'to': to_sq, 'color': color})

    # Extract colored squares: [%csl Gf5,Re4] -> green f5, red e4
    square_match = re.findall(r'\[%csl ([^\]]+)\]', comment)
    for match in square_match:
        for sq in match.split(','):
            if len(sq) >= 3:
                color = {'G': 'green', 'R': 'red', 'Y': 'yellow', 'B': 'blue'}.get(sq[0], 'green')
                square = sq[1:3]
                result['squares'].append({'square': square, 'color': color})

    # Extract clock times: [%clk 0:10:00]
    # (ignore for now)

    # Clean text comment (remove all annotations)
    text = re.sub(r'\[%[^\]]+\]', '', comment).strip()
    result['text'] = text

    return result


def extract_positions_from_pgn(pgn_text: str, study_id: str) -> list[dict]:
    """Extract positions with comments from PGN."""
    positions = []
    pgn_io = StringIO(pgn_text)

    study_name = None

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        chapter_name = game.headers.get("Event", "Unknown")
        if not study_name:
            study_name = game.headers.get("Event", "Unknown Study")

        # Walk through the game tree
        def walk_node(node, board, line, ply):
            for child in node.variations:
                move = child.move
                san = board.san(move)

                # Get FEN BEFORE the move (position where you make the decision)
                fen_before = board.fen()

                board.push(move)
                current_line = line + [san]

                # Extract and parse comment if present
                raw_comment = child.comment.strip() if child.comment else ""
                annotations = parse_annotations(raw_comment) if raw_comment else {'text': '', 'arrows': [], 'squares': []}

                # Check if this is a sideline (not the main variation)
                is_sideline = len(node.variations) > 1 and child != node.variations[0]
                has_content = annotations['text'] or annotations['arrows'] or annotations['squares']
                is_mainline = len(node.variations) >= 1 and child == node.variations[0]

                # Store ALL moves from studies to allow following study lines without gaps
                # - Main line moves: always store (to allow following the study)
                # - Sideline moves: always store (they're recommendations)
                # - Annotated moves: always store (they have explanations)
                positions.append({
                    'fen': fen_before.split(' ')[0],  # Position part only (before move)
                    'full_fen': fen_before,
                    'study_id': study_id,
                    'chapter_name': chapter_name,
                    'move_san': san,
                    'comment_text': annotations['text'],
                    'arrows': annotations['arrows'],
                    'squares': annotations['squares'],
                    'full_line': ' '.join(current_line),
                    'ply': ply,
                    'is_sideline': is_sideline,
                    'is_mainline': is_mainline,
                    'has_annotation': has_content
                })

                # Recurse into variations
                walk_node(child, board.copy(), current_line, ply + 1)
                board.pop()

        walk_node(game, game.board(), [], 0)

    return positions, study_name


def index_opening(conn: sqlite3.Connection, opening_name: str, max_studies: int = 20):
    """Search and index studies for an opening."""
    print(f"\nSearching for: {opening_name}")
    studies = search_studies(opening_name)[:max_studies]
    print(f"Found {len(studies)} studies")

    for study in tqdm(studies, desc="Downloading"):
        # Check if already indexed
        existing = conn.execute(
            "SELECT id FROM studies WHERE id = ?",
            (study['id'],)
        ).fetchone()

        if existing:
            continue

        # Download and parse
        pgn = download_study_pgn(study['id'])
        if not pgn:
            continue

        # Extract positions
        positions, study_name = extract_positions_from_pgn(pgn, study['id'])

        # Store study metadata with actual name from PGN + raw PGN for re-processing
        conn.execute(
            "INSERT OR REPLACE INTO studies VALUES (?, ?, ?, ?, ?, ?)",
            (study['id'], study_name or study['name'], '', study['url'], opening_name, pgn)
        )

        # Store positions
        for pos in positions:
            conn.execute(
                """INSERT INTO positions
                   (fen, full_fen, study_id, chapter_name, move_san, comment_text, arrows, squares, full_line, ply, is_sideline, is_mainline, has_annotation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pos['fen'],
                    pos['full_fen'],
                    pos['study_id'],
                    pos['chapter_name'],
                    pos['move_san'],
                    pos['comment_text'],
                    json.dumps(pos['arrows']),
                    json.dumps(pos['squares']),
                    pos['full_line'],
                    pos['ply'],
                    1 if pos.get('is_sideline') else 0,
                    1 if pos.get('is_mainline') else 0,
                    1 if pos.get('has_annotation') else 0
                )
            )

        conn.commit()
        time.sleep(RATE_LIMIT)

    count = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE study_id IN (SELECT id FROM studies WHERE opening_query = ?)",
        (opening_name,)
    ).fetchone()[0]
    print(f"Indexed {count} annotated positions for {opening_name}")


def query_position(conn: sqlite3.Connection, fen: str, limit: int = 5) -> list[dict]:
    """Query annotations for a position."""
    # Normalize FEN (just piece placement)
    fen_prefix = fen.split(' ')[0]

    rows = conn.execute("""
        SELECT p.comment, p.move_san, p.full_line, p.chapter_name, s.name, s.url
        FROM positions p
        JOIN studies s ON p.study_id = s.id
        WHERE p.fen = ?
        ORDER BY LENGTH(p.comment) DESC
        LIMIT ?
    """, (fen_prefix, limit)).fetchall()

    return [
        {
            'comment': row[0],
            'move': row[1],
            'line': row[2],
            'chapter': row[3],
            'study_name': row[4],
            'url': row[5]
        }
        for row in rows
    ]


def query_similar_positions(conn: sqlite3.Connection, moves: list[str], limit: int = 5) -> list[dict]:
    """Query annotations for positions in a line."""
    board = chess.Board()
    results = []

    for move in moves:
        try:
            board.push_san(move)
            fen_prefix = board.fen().split(' ')[0]

            rows = conn.execute("""
                SELECT p.comment, p.move_san, p.full_line, s.name
                FROM positions p
                JOIN studies s ON p.study_id = s.id
                WHERE p.fen = ?
                AND LENGTH(p.comment) > 50
                LIMIT 3
            """, (fen_prefix,)).fetchall()

            for row in rows:
                results.append({
                    'after_move': move,
                    'comment': row[0],
                    'suggested_move': row[1],
                    'line': row[2],
                    'source': row[3]
                })
        except:
            continue

    return results[:limit]


def export_to_json(conn: sqlite3.Connection, output_path: Path = Path("study_moves.json")):
    """Export all study moves to JSON for frontend consumption."""
    rows = conn.execute("""
        SELECT p.fen, p.move_san, p.comment_text, p.arrows, p.squares, p.full_line, p.ply,
               s.name as study_name, s.url as study_url, p.chapter_name,
               p.is_sideline, p.is_mainline, p.has_annotation, p.study_id
        FROM positions p
        JOIN studies s ON p.study_id = s.id
        ORDER BY p.fen, p.is_mainline DESC, p.has_annotation DESC, LENGTH(p.comment_text) DESC
    """).fetchall()

    # Group by FEN for efficient lookup
    by_fen = {}
    for row in rows:
        fen = row[0]
        if fen not in by_fen:
            by_fen[fen] = []

        by_fen[fen].append({
            'move': row[1],
            'comment': row[2],
            'arrows': json.loads(row[3]) if row[3] else [],
            'squares': json.loads(row[4]) if row[4] else [],
            'line': row[5],
            'ply': row[6],
            'study': row[7],
            'study_url': row[8],
            'chapter': row[9],
            'is_sideline': bool(row[10]),
            'is_mainline': bool(row[11]),
            'has_annotation': bool(row[12]),
            'study_id': row[13]
        })

    # Compute continuation lengths for each move
    print("Computing continuation lengths...")
    for fen, moves in tqdm(by_fen.items(), desc="Processing positions"):
        for move_data in moves:
            move_data['continuation'] = count_continuation(
                fen, move_data['move'], move_data['study'], by_fen
            )

    output = {
        'positions': by_fen,
        'total_positions': len(by_fen),
        'total_annotations': len(rows)
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nExported {len(rows)} annotations for {len(by_fen)} positions to {output_path}")


def count_continuation(start_fen: str, first_move: str, study_name: str, by_fen: dict) -> int:
    """Count how many moves remain in a study line from given position."""
    try:
        # Need full FEN for chess library - reconstruct with default values
        board = chess.Board(start_fen + " w - - 0 1")
    except:
        try:
            board = chess.Board(start_fen + " b - - 0 1")
        except:
            return 0

    try:
        board.push_san(first_move)
    except:
        return 0

    count = 0
    for _ in range(50):  # Max 50 moves
        next_fen = board.fen().split(' ')[0]
        next_moves = by_fen.get(next_fen, [])
        next_move = next((m for m in next_moves if m['study'] == study_name), None)
        if not next_move:
            break

        count += 1
        try:
            board.push_san(next_move['move'])
        except:
            break

    return count


def get_openings_from_weaknesses(min_games: int = 3, max_openings: int = 15) -> list[str]:
    """Extract opening names from weaknesses.json, sorted by games played.

    Args:
        min_games: Minimum games to include an opening
        max_openings: Maximum number of openings to return
    """
    try:
        data = json.loads(Path("weaknesses.json").read_text())

        # Count games per opening
        opening_games = {}
        for w in data.get("weaknesses", []):
            opening = w.get("opening", "")
            games = w.get("games", 0)
            if opening:
                # Simplify opening name for search
                # "Queen's Pawn Game: Accelerated London System" -> "London System"
                if ":" in opening:
                    opening = opening.split(":")[-1].strip()
                opening_games[opening] = opening_games.get(opening, 0) + games

        # Sort by games played (descending) and filter
        sorted_openings = sorted(opening_games.items(), key=lambda x: -x[1])
        filtered = [(name, count) for name, count in sorted_openings if count >= min_games]

        print(f"Opening priorities (by games played):")
        for name, count in filtered[:max_openings]:
            print(f"  {count:4d} games: {name}")

        if len(filtered) > max_openings:
            print(f"  ... and {len(filtered) - max_openings} more (skipped)")

        return [name for name, _ in filtered[:max_openings]]
    except Exception as e:
        print(f"Error reading weaknesses: {e}")
        return []


def reprocess_from_stored_pgns(conn: sqlite3.Connection):
    """Re-process all stored PGNs without re-downloading. Fast iteration on parsing logic."""
    print("Re-processing from stored PGNs...")

    # Clear existing positions (but keep studies with raw_pgn)
    conn.execute("DELETE FROM positions")
    conn.commit()

    # Get all stored studies with PGNs
    studies = conn.execute("""
        SELECT id, name, raw_pgn FROM studies WHERE raw_pgn IS NOT NULL
    """).fetchall()

    print(f"Found {len(studies)} studies with stored PGNs")

    for study_id, study_name, pgn in tqdm(studies, desc="Processing"):
        if not pgn:
            continue

        positions, _ = extract_positions_from_pgn(pgn, study_id)

        for pos in positions:
            conn.execute(
                """INSERT INTO positions
                   (fen, full_fen, study_id, chapter_name, move_san, comment_text, arrows, squares, full_line, ply, is_sideline, is_mainline, has_annotation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pos['fen'],
                    pos['full_fen'],
                    pos['study_id'],
                    pos['chapter_name'],
                    pos['move_san'],
                    pos['comment_text'],
                    json.dumps(pos['arrows']),
                    json.dumps(pos['squares']),
                    pos['full_line'],
                    pos['ply'],
                    1 if pos.get('is_sideline') else 0,
                    1 if pos.get('is_mainline') else 0,
                    1 if pos.get('has_annotation') else 0
                )
            )

        conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    print(f"Re-processed {total} positions")


def main():
    import sys

    # Resume from existing DB if present
    conn = init_db()

    # Check for --reprocess flag
    if '--reprocess' in sys.argv:
        reprocess_from_stored_pgns(conn)
        export_to_json(conn)
        conn.close()
        return

    # Get openings from your weaknesses
    openings = get_openings_from_weaknesses()

    if not openings:
        # Default openings to index
        openings = [
            "London System",
            "Italian Game",
            "Sicilian Defense",
            "Caro-Kann",
            "French Defense",
            "Queen's Gambit",
        ]

    print(f"Will index studies for: {openings}")

    for opening in openings:
        index_opening(conn, opening, max_studies=15)

    # Show stats
    total_positions = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    total_studies = conn.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
    print(f"\n=== INDEX COMPLETE ===")
    print(f"Studies: {total_studies}")
    print(f"Annotated positions: {total_positions}")

    # Export to JSON for frontend
    export_to_json(conn)

    conn.close()


if __name__ == "__main__":
    main()
