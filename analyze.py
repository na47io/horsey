#!/usr/bin/env python3
"""
Precompute opening weaknesses using native Stockfish.
Outputs a JSON file that the browser app loads.
"""

import json
import chess
import chess.engine
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

# Config
DEPTH = 15
MIN_GAMES = 3
EVAL_THRESHOLD = 0.3  # pawns

# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def analyze_position_stats(board: chess.Board, color: chess.Color) -> dict:
    """Extract detailed positional statistics."""
    stats = {}

    # Material count
    material = {chess.WHITE: 0, chess.BLACK: 0}
    for sq, piece in board.piece_map().items():
        material[piece.color] += PIECE_VALUES[piece.piece_type]
    stats["material_balance"] = material[color] - material[not color]

    # Development (pieces off back rank)
    back_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    developed = 0
    total_pieces = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(piece_type, color):
            total_pieces += 1
            if not (chess.BB_SQUARES[sq] & back_rank):
                developed += 1
    stats["development"] = developed
    stats["development_pct"] = developed / max(total_pieces, 1)

    # Castling
    stats["can_castle_kingside"] = board.has_kingside_castling_rights(color)
    stats["can_castle_queenside"] = board.has_queenside_castling_rights(color)
    stats["has_castled"] = not (stats["can_castle_kingside"] or stats["can_castle_queenside"]) or _king_has_moved_from_start(board, color)

    # King safety - pawn shield
    king_sq = board.king(color)
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)

        # Count pawns near king
        pawn_shield = 0
        for f in range(max(0, king_file - 1), min(8, king_file + 2)):
            for r in range(king_rank, king_rank + 2 if color == chess.WHITE else king_rank - 1, 1 if color == chess.WHITE else -1):
                if 0 <= r < 8:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_shield += 1
        stats["king_pawn_shield"] = pawn_shield

        # King exposure (attackers near king)
        king_zone = chess.BB_SQUARES[king_sq]
        for adj in chess.SQUARES:
            if chess.square_distance(king_sq, adj) <= 2:
                king_zone |= chess.BB_SQUARES[adj]
        stats["king_attackers"] = len(board.attackers(not color, king_sq))
    else:
        stats["king_pawn_shield"] = 0
        stats["king_attackers"] = 0

    # Center control (e4, d4, e5, d5)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_control = 0
    for sq in center_squares:
        # Piece on center
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            center_control += 2
        # Attacks on center
        if board.is_attacked_by(color, sq):
            center_control += 1
    stats["center_control"] = center_control

    # Pawn structure
    pawns = list(board.pieces(chess.PAWN, color))
    pawn_files = [chess.square_file(sq) for sq in pawns]

    # Doubled pawns
    from collections import Counter
    file_counts = Counter(pawn_files)
    stats["doubled_pawns"] = sum(1 for c in file_counts.values() if c > 1)

    # Isolated pawns (no friendly pawns on adjacent files)
    isolated = 0
    for sq in pawns:
        f = chess.square_file(sq)
        has_neighbor = any(
            chess.square_file(p) in [f - 1, f + 1]
            for p in pawns if p != sq
        )
        if not has_neighbor:
            isolated += 1
    stats["isolated_pawns"] = isolated

    # Passed pawns (no enemy pawns ahead on same or adjacent files)
    passed = 0
    enemy_pawns = list(board.pieces(chess.PAWN, not color))
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True
        for ep in enemy_pawns:
            ef = chess.square_file(ep)
            er = chess.square_rank(ep)
            if abs(ef - f) <= 1:
                if color == chess.WHITE and er > r:
                    is_passed = False
                    break
                elif color == chess.BLACK and er < r:
                    is_passed = False
                    break
        if is_passed:
            passed += 1
    stats["passed_pawns"] = passed

    # Mobility (total legal moves)
    # We need to check if it's our turn
    if board.turn == color:
        stats["mobility"] = len(list(board.legal_moves))
    else:
        # Make a null move conceptually - just count attacks
        stats["mobility"] = sum(
            len(board.attacks(sq))
            for sq in board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color) |
               board.pieces(chess.ROOK, color) | board.pieces(chess.QUEEN, color)
        )

    # Hanging pieces (our pieces attacked but not defended)
    hanging = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            if board.is_attacked_by(not color, sq):
                defenders = len(board.attackers(color, sq))
                attackers = len(board.attackers(not color, sq))
                if defenders < attackers:
                    hanging += 1
    stats["hanging_pieces"] = hanging

    # Bishop pair
    bishops = list(board.pieces(chess.BISHOP, color))
    stats["has_bishop_pair"] = len(bishops) >= 2

    return stats


def _king_has_moved_from_start(board: chess.Board, color: chess.Color) -> bool:
    """Check if king is not on starting square (proxy for has castled)."""
    king_sq = board.king(color)
    start_sq = chess.E1 if color == chess.WHITE else chess.E8
    return king_sq != start_sq


def classify_mistake_type(board_before: chess.Board, user_move: str, best_move: str, pv_line: list) -> str:
    """Classify what type of mistake was made."""
    board = board_before.copy()
    color = board.turn

    # Check if best line is tactical (starts with checks/captures)
    tactical_moves = 0
    test_board = board.copy()
    for move in pv_line[:4]:
        if test_board.is_capture(move) or test_board.gives_check(move):
            tactical_moves += 1
        test_board.push(move)

    if tactical_moves >= 2:
        return "tactical"

    # Check if we hung a piece
    try:
        user_move_obj = board.parse_san(user_move)
        board.push(user_move_obj)
        stats_after = analyze_position_stats(board, color)
        board.pop()
        stats_before = analyze_position_stats(board, color)

        if stats_after["hanging_pieces"] > stats_before["hanging_pieces"]:
            return "hanging_piece"
    except:
        pass

    # Check development
    try:
        user_move_obj = board.parse_san(user_move)
        best_move_obj = board.parse_san(best_move)

        board.push(user_move_obj)
        dev_after_user = analyze_position_stats(board, color)["development"]
        board.pop()

        board.push(best_move_obj)
        dev_after_best = analyze_position_stats(board, color)["development"]
        board.pop()

        if dev_after_best > dev_after_user:
            return "development"
    except:
        pass

    # Check pawn structure
    try:
        user_move_obj = board.parse_san(user_move)
        board.push(user_move_obj)
        stats = analyze_position_stats(board, color)
        board.pop()

        if stats["doubled_pawns"] > 0 or stats["isolated_pawns"] > 0:
            return "pawn_structure"
    except:
        pass

    return "positional"

def find_stockfish() -> str:
    """Find stockfish binary."""
    import shutil
    import subprocess

    # Check mise first
    try:
        result = subprocess.run(["mise", "which", "stockfish"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except:
        pass

    # Check PATH
    path = shutil.which("stockfish")
    if path:
        return path

    # Common locations
    for p in ["/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish", "/usr/bin/stockfish"]:
        if Path(p).exists():
            return p

    raise RuntimeError("Stockfish not found. Install with: brew install stockfish")

def load_tree(cache_path: Path) -> dict:
    """Load the opening tree from IndexedDB export or generate from scratch."""
    import requests
    from datetime import datetime, timedelta
    import re

    username = "nikkyrush"
    time_class = "rapid"

    headers = {
        "User-Agent": "ChessOpeningsTrainer/1.0 (personal use)"
    }

    print(f"Fetching games for {username}...")

    # Get archives
    r = requests.get(f"https://api.chess.com/pub/player/{username}/games/archives", headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch archives: {r.status_code} {r.text}")
    all_archives = r.json().get("archives", [])

    # Filter to last year only
    one_year_ago = datetime.now() - timedelta(days=365)
    archives = []
    for url in all_archives:
        # Extract year/month from URL like .../games/2024/01
        match = re.search(r'/games/(\d{4})/(\d{2})$', url)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            archive_date = datetime(year, month, 1)
            if archive_date >= one_year_ago:
                archives.append(url)

    print(f"Filtering to last year: {len(archives)}/{len(all_archives)} archives")

    games = []
    for url in tqdm(archives, desc="Fetching archives"):
        r = requests.get(url, headers=headers)
        data = r.json()
        for g in data.get("games", []):
            if g.get("time_class") == time_class:
                games.append(g)

    print(f"Found {len(games)} {time_class} games")

    # Build trees
    tree_white = {"moves": {}, "stats": {"w": 0, "d": 0, "l": 0}}
    tree_black = {"moves": {}, "stats": {"w": 0, "d": 0, "l": 0}}

    for g in games:
        is_white = g["white"]["username"].lower() == username.lower()
        pgn = g.get("pgn", "")
        if not pgn:
            continue

        # Determine result
        if g["white"]["result"] == "win":
            result = "w" if is_white else "l"
        elif g["black"]["result"] == "win":
            result = "l" if is_white else "w"
        else:
            result = "d"

        moves = parse_pgn(pgn)
        if not moves:
            continue

        tree = tree_white if is_white else tree_black
        add_to_tree(tree, moves, result, is_white)

    return tree_white, tree_black


def parse_pgn(pgn: str) -> list:
    """Extract moves from PGN."""
    import re
    # Remove headers and comments
    clean = re.sub(r'\[.*?\]\s*', '', pgn)
    clean = re.sub(r'\{[^}]*\}', '', clean)
    clean = re.sub(r'\([^)]*\)', '', clean)
    clean = re.sub(r'\d+\.\.\.', '', clean)
    clean = re.sub(r'1-0|0-1|1/2-1/2|\*', '', clean)

    moves = []
    for m in re.finditer(r'\d+\.\s*(\S+)(?:\s+(\S+))?', clean):
        if m.group(1) and '.' not in m.group(1):
            moves.append(m.group(1))
        if m.group(2) and '.' not in m.group(2):
            moves.append(m.group(2))
    return moves


def add_to_tree(tree: dict, moves: list, result: str, is_white: bool):
    """Add a game to the opening tree."""
    tree["stats"][result] += 1
    node = tree
    for i, mv in enumerate(moves):
        mine = (is_white and i % 2 == 0) or (not is_white and i % 2 == 1)
        if mv not in node["moves"]:
            node["moves"][mv] = {"moves": {}, "stats": {"w": 0, "d": 0, "l": 0}, "mine": mine}
        node = node["moves"][mv]
        node["stats"][result] += 1


def collect_positions(tree: dict, is_white: bool) -> list:
    """Collect positions where it's user's turn."""
    positions = []

    def walk(node, moves, depth):
        if depth > 20:  # Limit to opening phase
            return

        is_user_turn = (is_white and depth % 2 == 0) or (not is_white and depth % 2 == 1)

        if is_user_turn and node["moves"]:
            # Get user's most played move
            user_moves = [
                (san, data) for san, data in node["moves"].items()
                if data.get("mine", False)
            ]
            if user_moves:
                user_moves.sort(key=lambda x: sum(x[1]["stats"].values()), reverse=True)
                top_move, top_data = user_moves[0]
                games = sum(top_data["stats"].values())

                if games >= MIN_GAMES:
                    positions.append({
                        "moves": list(moves),
                        "user_move": top_move,
                        "games": games,
                        "is_white": is_white
                    })

        for san, child in node["moves"].items():
            walk(child, moves + [san], depth + 1)

    walk(tree, [], 0)
    return positions


def get_opening_name(moves: list, eco_db: dict) -> str:
    """Get opening name from ECO database."""
    for i in range(len(moves), 0, -1):
        line = " ".join(moves[:i])
        if line in eco_db:
            return eco_db[line]

    if not moves:
        return "Starting Position"
    if moves[0] == "e4":
        return "King's Pawn"
    if moves[0] == "d4":
        return "Queen's Pawn"
    if moves[0] == "c4":
        return "English Opening"
    if moves[0] == "Nf3":
        return "Reti Opening"
    return "Opening"


def load_eco_database() -> dict:
    """Load ECO database from lichess."""
    import requests
    import re

    eco_db = {}
    for letter in "abcde":
        url = f"https://raw.githubusercontent.com/lichess-org/chess-openings/master/{letter}.tsv"
        r = requests.get(url)
        for line in r.text.strip().split("\n")[1:]:  # skip header
            parts = line.split("\t")
            if len(parts) >= 3:
                eco, name, pgn = parts[0], parts[1], parts[2]
                # Normalize PGN
                normalized = re.sub(r'\d+\.\s*', '', pgn).strip()
                eco_db[normalized] = name

    print(f"Loaded {len(eco_db)} ECO entries")
    return eco_db


def analyze_single_position(args):
    """Analyze a single position - designed for multiprocessing."""
    pos, stockfish_path, eco_db = args

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        # Build position
        board = chess.Board()
        for m in pos["moves"]:
            try:
                board.push_san(m)
            except:
                engine.quit()
                return None

        color = chess.WHITE if pos["is_white"] else chess.BLACK
        user_move = pos["user_move"]

        # Collect position stats BEFORE the move
        position_stats = analyze_position_stats(board, color)

        # Get multipv analysis
        info = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=5)
        engine.quit()

        if not info:
            return None

        best_info = info[0]
        best_move = best_info["pv"][0]
        best_pv = best_info.get("pv", [])
        best_score = best_info["score"].white().score(mate_score=10000)

        if best_score is None:
            return None

        # Find user move in results
        try:
            user_move_obj = board.parse_san(user_move)
        except:
            return None

        user_score = None
        for pv_info in info:
            if pv_info["pv"][0] == user_move_obj:
                user_score = pv_info["score"].white().score(mate_score=10000)
                break

        # If user move not in top 5, assume it's worse
        if user_score is None:
            user_score = best_score - 200  # Assume 2 pawns worse

        # Adjust for side to move
        if board.turn == chess.BLACK:
            best_score = -best_score
            user_score = -user_score

        eval_loss = (best_score - user_score) / 100  # Convert to pawns
        best_san = board.san(best_move)

        # Always return stats for aggregation
        result = {
            "opening": get_opening_name(pos["moves"], eco_db),
            "line": pos["moves"],
            "move_number": (len(pos["moves"]) // 2) + 1,
            "played": user_move,
            "best": best_san,
            "eval_loss": round(eval_loss, 2),
            "games": pos["games"],
            "is_white": pos["is_white"],
            "position_stats": position_stats,
            "is_weakness": eval_loss >= EVAL_THRESHOLD and best_san != user_move,
        }

        # Classify mistake type if it's a weakness
        if result["is_weakness"]:
            result["mistake_type"] = classify_mistake_type(board, user_move, best_san, best_pv)
        else:
            result["mistake_type"] = None

        return result
    except Exception as e:
        return None


def aggregate_mistake_types(weaknesses: list) -> dict:
    """Count mistake types across all weaknesses."""
    counts = defaultdict(int)
    for w in weaknesses:
        mtype = w.get("mistake_type") or "unknown"
        counts[mtype] += 1
    return dict(counts)


def aggregate_opening_stats(results: list) -> dict:
    """Aggregate stats per opening."""
    from collections import defaultdict
    import statistics

    openings = defaultdict(lambda: {
        "positions": [],
        "weaknesses": [],
        "mistake_types": defaultdict(int),
    })

    for r in results:
        opening = r["opening"]
        openings[opening]["positions"].append(r)
        if r["is_weakness"]:
            openings[opening]["weaknesses"].append(r)
            if r["mistake_type"]:
                openings[opening]["mistake_types"][r["mistake_type"]] += 1

    # Compute aggregates
    aggregated = {}
    for opening, data in openings.items():
        positions = data["positions"]
        if not positions:
            continue

        # Extract stats arrays
        stats_list = [p["position_stats"] for p in positions if p.get("position_stats")]
        if not stats_list:
            continue

        def safe_mean(values):
            return round(statistics.mean(values), 2) if values else 0

        def safe_pct(values):
            return round(sum(values) / len(values) * 100, 1) if values else 0

        aggregated[opening] = {
            "total_positions": len(positions),
            "weakness_count": len(data["weaknesses"]),
            "weakness_rate": round(len(data["weaknesses"]) / len(positions) * 100, 1),

            # Mistake type breakdown
            "mistake_types": dict(data["mistake_types"]),

            # Average position stats
            "avg_development": safe_mean([s["development"] for s in stats_list]),
            "avg_center_control": safe_mean([s["center_control"] for s in stats_list]),
            "avg_mobility": safe_mean([s["mobility"] for s in stats_list]),
            "avg_king_safety": safe_mean([s["king_pawn_shield"] for s in stats_list]),

            # Pawn structure issues
            "doubled_pawn_rate": safe_pct([1 if s["doubled_pawns"] > 0 else 0 for s in stats_list]),
            "isolated_pawn_rate": safe_pct([1 if s["isolated_pawns"] > 0 else 0 for s in stats_list]),

            # Castling
            "castled_rate": safe_pct([1 if s["has_castled"] else 0 for s in stats_list]),

            # Hanging pieces
            "hanging_piece_rate": safe_pct([1 if s["hanging_pieces"] > 0 else 0 for s in stats_list]),

            # Bishop pair retention
            "bishop_pair_rate": safe_pct([1 if s["has_bishop_pair"] else 0 for s in stats_list]),

            # Average eval loss
            "avg_eval_loss": safe_mean([p["eval_loss"] for p in positions]),

            # Top weaknesses for this opening
            "top_weaknesses": [
                {"line": w["line"], "played": w["played"], "best": w["best"], "eval_loss": w["eval_loss"], "type": w["mistake_type"]}
                for w in sorted(data["weaknesses"], key=lambda x: -x["eval_loss"])[:3]
            ]
        }

    return aggregated


def main():
    print("Loading ECO database...")
    eco_db = load_eco_database()

    print("\nLoading games from chess.com...")
    tree_white, tree_black = load_tree(Path("cache.json"))

    print("\nCollecting positions...")
    white_positions = collect_positions(tree_white, True)
    black_positions = collect_positions(tree_black, False)
    all_positions = white_positions + black_positions
    print(f"Found {len(all_positions)} positions to analyze")

    stockfish_path = find_stockfish()
    num_workers = multiprocessing.cpu_count()
    print(f"\nStarting Stockfish analysis (depth {DEPTH})...")
    print(f"Using: {stockfish_path}")
    print(f"Parallel workers: {num_workers}")

    # Prepare args for parallel processing
    args_list = [(pos, stockfish_path, eco_db) for pos in all_positions]

    all_results = []
    weaknesses = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_single_position, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing positions"):
            result = future.result()
            if result:
                all_results.append(result)
                if result["is_weakness"]:
                    weaknesses.append(result)

    # Sort weaknesses by eval loss
    weaknesses.sort(key=lambda x: -x["eval_loss"])

    # Aggregate stats per opening
    print("\nAggregating statistics...")
    opening_stats = aggregate_opening_stats(all_results)

    # Clean weaknesses for output (remove position_stats to reduce size)
    clean_weaknesses = [
        {
            "opening": w["opening"],
            "line": w["line"],
            "played": w["played"],
            "best": w["best"],
            "eval_loss": w["eval_loss"],
            "games": w["games"],
            "is_white": w["is_white"],
            "mistake_type": w["mistake_type"]
        }
        for w in weaknesses
    ]

    # Save weaknesses.json
    weaknesses_output = {
        "weaknesses": clean_weaknesses,
        "analyzed": len(all_positions),
        "depth": DEPTH
    }
    Path("weaknesses.json").write_text(json.dumps(weaknesses_output, indent=2))
    print(f"\nSaved {len(weaknesses)} weaknesses to weaknesses.json")

    # Save stats.json
    stats_output = {
        "openings": opening_stats,
        "summary": {
            "total_positions": len(all_results),
            "total_weaknesses": len(weaknesses),
            "overall_weakness_rate": round(len(weaknesses) / max(len(all_results), 1) * 100, 1),
            "mistake_breakdown": aggregate_mistake_types(weaknesses),
            "worst_openings": sorted(
                [(name, data["weakness_rate"]) for name, data in opening_stats.items()],
                key=lambda x: -x[1]
            )[:5],
            "best_openings": sorted(
                [(name, data["weakness_rate"]) for name, data in opening_stats.items()],
                key=lambda x: x[1]
            )[:5],
        }
    }
    Path("stats.json").write_text(json.dumps(stats_output, indent=2))
    print(f"Saved opening stats to stats.json")

    # Summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)

    print(f"\nPositions analyzed: {len(all_results)}")
    print(f"Weaknesses found: {len(weaknesses)} ({stats_output['summary']['overall_weakness_rate']}%)")

    print("\n--- MISTAKE TYPES ---")
    for mtype, count in stats_output['summary']['mistake_breakdown'].items():
        print(f"  {mtype}: {count}")

    print("\n--- WORST OPENINGS (by weakness rate) ---")
    for name, rate in stats_output['summary']['worst_openings']:
        print(f"  {name}: {rate}%")

    print("\n--- BEST OPENINGS (by weakness rate) ---")
    for name, rate in stats_output['summary']['best_openings']:
        print(f"  {name}: {rate}%")

    print("\n--- TOP 10 WEAKNESSES ---")
    for w in weaknesses[:10]:
        print(f"  [{w['mistake_type'] or 'unknown'}] {w['opening']}: {w['played']} -> {w['best']} (-{w['eval_loss']})")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
