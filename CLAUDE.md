# Chess Opening Trainer

Personal opening trainer that analyzes your chess.com games.

## Setup

```bash
cd ~/code/chess-openings
mise install   # installs python, uv
mise run setup # installs deps + stockfish via brew
```

## Usage

### Run the web app
```bash
mise run serve
# Open http://localhost:8000
```
Uses FastAPI + uvicorn with hot reload. Serves `static/index.html` and JSON APIs.

### Analyze your openings for weaknesses
```bash
mise run analyze
```
Runs Stockfish analysis in parallel. Outputs `weaknesses.json`.

### Scrape Lichess studies for opening explanations
```bash
mise run scrape-studies           # download from Lichess
uv run python scrape_studies.py --reprocess  # re-parse stored PGNs (fast)
```
Stores raw PGNs in `studies.db`, exports to `study_moves.json`.

## Architecture

```
chess-openings/
├── app.py              # FastAPI backend
├── static/
│   └── index.html      # Single-file frontend
├── studies.db          # Lichess study annotations (SQLite)
├── study_moves.json    # Exported for frontend
├── weaknesses.json     # Stockfish weakness analysis
└── stats.json          # Aggregated game stats
```

### API Endpoints
- `GET /api/studies?fen=...` - Study annotations for position
- `GET /api/weaknesses?color=white` - Weakness analysis
- `GET /api/stats` - Game statistics
- `GET /api/health` - Health check

## Features

- **Explore tab**: Browse your opening repertoire with win/draw/loss stats
- **Weaknesses tab**: Positions where you consistently play suboptimal moves
- **Study moves**: Lichess study annotations with arrows/squares
- **Stockfish eval bar**: Real-time engine evaluation
- **Move quality badges**: Best/Good/Inaccuracy/Mistake/Blunder
- **Arrow key navigation**: Left/Right to move through positions

## Config

In `analyze.py`:
- `DEPTH = 15` - Stockfish search depth
- `MIN_GAMES = 3` - Minimum games to consider a position
- `EVAL_THRESHOLD = 0.3` - Minimum eval loss to flag as weakness

## Data

- Games cached in IndexedDB (browser) with 24h TTL
- `studies.db` - Lichess study PGNs and parsed positions
- `weaknesses.json` - precomputed weakness analysis
