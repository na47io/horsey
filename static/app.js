import { Chessground } from 'https://esm.sh/chessground@9.1.1';

// =============================================================================
// Config & State
// =============================================================================

const CONFIG = {
    username: 'nikkyrush',
    timeClass: 'rapid',
    cacheKey: 'openings_nikkyrush_rapid_v3',
    cacheTTL: 24 * 60 * 60 * 1000
};

const state = {
    // Game trees
    treeWhite: { moves: {}, stats: { w: 0, d: 0, l: 0 } },
    treeBlack: { moves: {}, stats: { w: 0, d: 0, l: 0 } },
    tree: null,
    node: null,

    // Chess state
    game: null,
    cg: null,
    history: [],
    forwardStack: [],
    color: 'white',

    // Engine
    stockfish: null,
    currentEval: 0,
    bestMove: null,
    multiPvEvals: [],
    moveQualities: {},
    moveEvals: {},
    engineBestSan: null,

    // Data
    ecoDb: {},
    studyData: null,
    weaknesses: [],

    // UI state
    currentWeakness: null,
    activeStudyLine: null  // { study: string, moves: string[] } when following a study
};

// =============================================================================
// Utilities
// =============================================================================

const $ = id => document.getElementById(id);

function getLegalMoves() {
    const dests = new Map();
    for (const m of state.game.moves({ verbose: true })) {
        if (!dests.has(m.from)) dests.set(m.from, []);
        dests.get(m.from).push(m.to);
    }
    return dests;
}

function getArrows() {
    const arrows = [];
    const testGame = new Chess(state.game.fen());

    // Study arrows first (they're recommendations)
    const studyMoves = getStudyMoves();
    const studyWithArrows = studyMoves.find(m => m.arrows?.length || m.squares?.length);
    if (studyWithArrows) {
        for (const a of (studyWithArrows.arrows || [])) {
            arrows.push({ orig: a.from, dest: a.to, brush: a.color || 'blue' });
        }
    }

    // Your move arrows
    const moves = Object.entries(state.node?.moves || {})
        .map(([san, data]) => {
            const move = testGame.move(san);
            if (move) {
                testGame.undo();
                return { san, from: move.from, to: move.to, total: data.stats.w + data.stats.d + data.stats.l, mine: data.mine };
            }
            return null;
        })
        .filter(Boolean)
        .sort((a, b) => b.total - a.total)
        .slice(0, 5);

    for (const m of moves) {
        arrows.push({ orig: m.from, dest: m.to, brush: m.mine ? 'green' : 'yellow' });
    }
    return arrows;
}

function updateBoard() {
    state.cg.set({
        fen: state.game.fen(),
        turnColor: state.game.turn() === 'w' ? 'white' : 'black',
        movable: { dests: getLegalMoves() },
        drawable: { autoShapes: getArrows() }
    });
}

// =============================================================================
// ECO Database
// =============================================================================

async function loadEcoDatabase() {
    try {
        for (const letter of ['a', 'b', 'c', 'd', 'e']) {
            const r = await fetch(`https://raw.githubusercontent.com/lichess-org/chess-openings/master/${letter}.tsv`);
            const text = await r.text();
            for (const line of text.trim().split('\n').slice(1)) {
                const [eco, name, pgn] = line.split('\t');
                if (pgn) state.ecoDb[pgn.replace(/\d+\.\s*/g, '').trim()] = { eco, name };
            }
        }
    } catch (e) { console.error('Failed to load ECO:', e); }
}

function getOpening(moves) {
    if (!moves.length) return 'Starting Position';
    for (let i = moves.length; i > 0; i--) {
        const entry = state.ecoDb[moves.slice(0, i).join(' ')];
        if (entry) return entry.name;
    }
    const fallbacks = { e4: "King's Pawn", d4: "Queen's Pawn", c4: 'English', Nf3: 'Reti' };
    return fallbacks[moves[0]] || 'Opening';
}

// =============================================================================
// Cache (IndexedDB)
// =============================================================================

function openDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open('chess-openings', 1);
        req.onerror = () => reject(req.error);
        req.onsuccess = () => resolve(req.result);
        req.onupgradeneeded = e => e.target.result.createObjectStore('cache');
    });
}

async function loadCache() {
    try {
        const db = await openDB();
        return new Promise(resolve => {
            const req = db.transaction('cache', 'readonly').objectStore('cache').get(CONFIG.cacheKey);
            req.onsuccess = () => {
                const c = req.result;
                resolve(c && Date.now() - c.ts < CONFIG.cacheTTL ? c : null);
            };
            req.onerror = () => resolve(null);
        });
    } catch { return null; }
}

async function saveCache() {
    try {
        const db = await openDB();
        const tx = db.transaction('cache', 'readwrite');
        tx.objectStore('cache').put({ ts: Date.now(), w: state.treeWhite, b: state.treeBlack }, CONFIG.cacheKey);
    } catch {}
}

// =============================================================================
// Game Fetching & Processing
// =============================================================================

async function fetchGames() {
    loadEcoDatabase();

    const cached = await loadCache();
    if (cached) {
        state.treeWhite = cached.w;
        state.treeBlack = cached.b;
        return init();
    }

    $('progress').textContent = 'Fetching archives...';
    const res = await fetch(`https://api.chess.com/pub/player/${CONFIG.username}/games/archives`);
    const { archives: allArchives } = await res.json();

    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
    const archives = allArchives.filter(url => {
        const m = url.match(/\/games\/(\d{4})\/(\d{2})$/);
        return m && new Date(+m[1], +m[2] - 1) >= oneYearAgo;
    });

    const games = [];
    for (let i = 0; i < archives.length; i++) {
        $('progress').textContent = `Fetching ${i + 1}/${archives.length}...`;
        try {
            const d = await (await fetch(archives[i])).json();
            if (d.games) games.push(...d.games.filter(g => g.time_class === CONFIG.timeClass));
        } catch {}
    }

    $('progress').textContent = `Processing ${games.length} games...`;
    processGames(games);
    await saveCache();
    init();
}

function processGames(games) {
    for (const g of games) {
        const isW = g.white.username.toLowerCase() === CONFIG.username.toLowerCase();
        if (!g.pgn) continue;

        let result = g.white.result === 'win' ? (isW ? 'w' : 'l')
                   : g.black.result === 'win' ? (isW ? 'l' : 'w') : 'd';

        const moves = parsePGN(g.pgn);
        if (moves.length) addToTree(isW ? state.treeWhite : state.treeBlack, moves, result, isW);
    }
}

function parsePGN(pgn) {
    const clean = pgn.replace(/\[.*?\]\s*/g, '').replace(/\{[^}]*\}/g, '')
        .replace(/\([^)]*\)/g, '').replace(/\d+\.\.\./g, '')
        .replace(/1-0|0-1|1\/2-1\/2|\*/g, '');
    const moves = [];
    let m;
    const re = /\d+\.\s*(\S+)(?:\s+(\S+))?/g;
    while ((m = re.exec(clean))) {
        if (m[1] && !m[1].includes('.')) moves.push(m[1]);
        if (m[2] && !m[2].includes('.')) moves.push(m[2]);
    }
    return moves;
}

function addToTree(tree, moves, result, isW) {
    tree.stats[result]++;
    let n = tree;
    for (let i = 0; i < moves.length; i++) {
        const mv = moves[i];
        const mine = isW ? (i % 2 === 0) : (i % 2 === 1);
        if (!n.moves[mv]) n.moves[mv] = { moves: {}, stats: { w: 0, d: 0, l: 0 }, mine };
        n = n.moves[mv];
        n.stats[result]++;
    }
}

// =============================================================================
// Stockfish Engine
// =============================================================================

async function initStockfish() {
    try {
        const res = await fetch('https://unpkg.com/stockfish.js@10.0.2/stockfish.js');
        const blob = new Blob([await res.text()], { type: 'application/javascript' });
        state.stockfish = new Worker(URL.createObjectURL(blob));

        state.stockfish.onmessage = e => {
            const line = e.data;
            const flip = state.game.turn() === 'b' ? -1 : 1;

            if (line.startsWith('info') && line.includes(' pv ')) {
                const depth = +(line.match(/depth (\d+)/)?.[1] || 0);
                if (depth >= 10) {
                    const pvNum = +(line.match(/multipv (\d+)/)?.[1] || 1);
                    const uci = line.match(/ pv (\S+)/)?.[1];
                    const cpMatch = line.match(/score cp (-?\d+)/);
                    const mateMatch = line.match(/score mate (-?\d+)/);

                    if (uci) {
                        let cp = cpMatch ? +cpMatch[1] * flip : mateMatch ? (+mateMatch[1] > 0 ? 9900 : -9900) * flip : 0;
                        if (pvNum === 1) {
                            state.multiPvEvals = [];
                            state.currentEval = cp / 100;
                            state.bestMove = uci;
                            updateEval(state.currentEval, mateMatch ? +mateMatch[1] * flip : null);
                        }
                        state.multiPvEvals.push({ uci, cp });
                    }
                }
            } else if (line.startsWith('bestmove')) {
                evaluateMoves();
            }
            if (state.multiPvEvals.length >= 5) evaluateMoves();
        };

        state.stockfish.postMessage('uci');
        state.stockfish.postMessage('setoption name MultiPV value 10');
        state.stockfish.postMessage('isready');
    } catch (e) {
        console.error('Stockfish init failed:', e);
        $('evalText').textContent = '—';
    }
}

function analyzePosition() {
    if (!state.stockfish) return;
    state.multiPvEvals = [];
    state.stockfish.postMessage('stop');
    state.stockfish.postMessage('position fen ' + state.game.fen());
    state.stockfish.postMessage('go depth 12');
}

function updateEval(score, mate = null) {
    const pct = Math.min(95, Math.max(5, 50 - (score * 10)));
    $('evalFill').style.height = pct + '%';
    $('evalText').textContent = mate !== null ? `${mate > 0 ? '+' : ''}M${Math.abs(mate)}`
                                               : `${score > 0 ? '+' : ''}${score.toFixed(1)}`;
}

function uciToSan(uci) {
    if (!uci || uci.length < 4) return null;
    const move = state.game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
    if (move) { const san = move.san; state.game.undo(); return san; }
    return null;
}

function classifyMove(cpLoss, isBest) {
    if (isBest) return { cls: 'best', label: 'Best' };
    if (cpLoss <= 50) return { cls: 'good', label: 'Good' };
    if (cpLoss <= 100) return { cls: 'inaccuracy', label: 'Inaccuracy' };
    if (cpLoss <= 200) return { cls: 'mistake', label: 'Mistake' };
    return { cls: 'blunder', label: 'Blunder' };
}

function evaluateMoves() {
    state.moveQualities = {};
    state.moveEvals = {};
    state.engineBestSan = null;
    if (!state.multiPvEvals.length) return;

    const bestCp = state.multiPvEvals[0].cp;
    const bestUci = state.multiPvEvals[0].uci;

    state.engineBestSan = uciToSan(bestUci);
    if (state.engineBestSan) state.moveEvals[state.engineBestSan] = bestCp / 100;

    const uciToCp = Object.fromEntries(state.multiPvEvals.map(e => [e.uci, e.cp]));
    const candidates = new Set([...Object.keys(state.node?.moves || {}), ...getStudyMoves().map(m => m.move)]);

    for (const san of candidates) {
        const move = state.game.move(san);
        if (move) {
            const uci = move.from + move.to + (move.promotion || '');
            state.game.undo();
            if (uciToCp[uci] !== undefined) {
                state.moveQualities[san] = classifyMove(Math.max(0, bestCp - uciToCp[uci]), uci === bestUci);
                state.moveEvals[san] = uciToCp[uci] / 100;
            }
        }
    }
    renderMoves();
}

// =============================================================================
// Study Data
// =============================================================================

async function loadStudyMoves() {
    try {
        const res = await fetch('study_moves.json');
        if (res.ok) state.studyData = await res.json();
    } catch {}
}

function getStudyMoves() {
    if (!state.studyData) return [];
    return state.studyData.positions[state.game.fen().split(' ')[0]] || [];
}

function getBestStudyAnnotation() {
    const moves = getStudyMoves();
    if (!moves.length) return null;

    // If following a study line, prioritize moves from that study
    if (state.activeStudyLine) {
        const fromActiveLine = moves.find(m => m.study === state.activeStudyLine.study);
        if (fromActiveLine) return fromActiveLine;
        // Line ended, clear active study
        state.activeStudyLine = null;
    }

    // Find move with best annotation (longest comment, or has continuation)
    const withComment = moves.filter(m => m.comment && m.comment.length > 20);
    if (withComment.length) {
        // Prefer ones that aren't just game results
        const explanatory = withComment.filter(m =>
            !m.comment.match(/wins by|resigns|was best|draw|checkmate is now/i)
        );
        if (explanatory.length) {
            return explanatory.sort((a, b) => (b.continuation || 0) - (a.continuation || 0))[0];
        }
        return withComment[0];
    }

    // No good comment, but maybe has continuation (a line to follow)
    const withContinuation = moves.filter(m => m.continuation > 3);
    if (withContinuation.length) {
        return withContinuation.sort((a, b) => b.continuation - a.continuation)[0];
    }

    return null;
}

function renderStudyPanel() {
    const panel = $('studyPanel');
    const annotation = getBestStudyAnnotation();

    if (!annotation) {
        panel.style.display = 'none';
        return;
    }

    panel.style.display = 'block';
    $('studySource').textContent = annotation.study;

    // Show comment or indicate this is part of a line
    if (annotation.comment) {
        $('studyComment').textContent = annotation.comment;
        $('studyComment').style.display = 'block';
    } else if (annotation.continuation > 0) {
        $('studyComment').textContent = `Study line continues for ${annotation.continuation} moves...`;
        $('studyComment').style.display = 'block';
    } else {
        $('studyComment').style.display = 'none';
    }

    // Actions: play this move, follow line
    let actions = '';
    actions += `<button onclick="window.playStudyMove('${annotation.move}', '${annotation.study}')">Play ${annotation.move}</button>`;

    if (annotation.continuation > 2) {
        actions += `<button class="secondary" onclick="window.followStudyLine('${annotation.study}')">Follow line (${annotation.continuation} moves)</button>`;
    }

    $('studyActions').innerHTML = actions;
}

window.playStudyMove = function(san, studyName) {
    // Set active study line if not already following one
    if (!state.activeStudyLine || state.activeStudyLine.study !== studyName) {
        state.activeStudyLine = { study: studyName, moves: [...state.history] };
    }
    playMove(san);
};

window.followStudyLine = function(studyName) {
    state.activeStudyLine = { study: studyName, moves: [...state.history] };

    // Find the next move from this study
    const moves = getStudyMoves();
    const studyMove = moves.find(m => m.study === studyName);
    if (studyMove) {
        playMove(studyMove.move);
    }
};

window.closeStudyPanel = function() {
    $('studyPanel').style.display = 'none';
    state.activeStudyLine = null;
};

// =============================================================================
// Navigation
// =============================================================================

function playMove(san) {
    const move = state.game.move(san);
    if (!move) return;

    state.forwardStack = [];
    state.history.push(san);
    state.node = state.node?.moves?.[san] || { moves: {}, stats: { w: 0, d: 0, l: 0 } };
    state.currentWeakness = null;
    $('errorInfo').style.display = 'none';

    updateBoard();
    render();
    analyzePosition();
}

function back() {
    if (!state.history.length) return;
    state.forwardStack.push(state.history.pop());

    state.game = new Chess();
    state.node = state.tree;
    for (const m of state.history) {
        state.game.move(m);
        if (state.node.moves[m]) state.node = state.node.moves[m];
    }

    state.currentWeakness = null;
    $('errorInfo').style.display = 'none';

    updateBoard();
    render();
    analyzePosition();
}

function forward() {
    if (!state.forwardStack.length) return;
    const san = state.forwardStack.pop();

    const move = state.game.move(san);
    if (!move) return;

    state.history.push(san);
    state.node = state.node?.moves?.[san] || { moves: {}, stats: { w: 0, d: 0, l: 0 } };

    updateBoard();
    render();
    analyzePosition();
}

function reset() {
    state.game = new Chess();
    state.history = [];
    state.forwardStack = [];
    state.node = state.tree;
    state.currentWeakness = null;
    state.activeStudyLine = null;
    $('errorInfo').style.display = 'none';

    state.cg.set({
        fen: state.game.fen(),
        orientation: state.color,
        turnColor: 'white',
        movable: { dests: getLegalMoves() },
        drawable: { autoShapes: getArrows() }
    });

    render();
    analyzePosition();
}

function setColor(c) {
    state.color = c;
    state.tree = c === 'white' ? state.treeWhite : state.treeBlack;
    $('whiteBtn').classList.toggle('active', c === 'white');
    $('blackBtn').classList.toggle('active', c === 'black');
    reset();
    if (state.weaknesses.length) renderWeaknesses();
}

// =============================================================================
// Rendering
// =============================================================================

function render() {
    $('openingName').textContent = getOpening(state.history);

    if (!state.history.length) {
        $('moveLine').innerHTML = 'Click a move or drag pieces to explore';
    } else {
        $('moveLine').innerHTML = state.history.map((m, i) =>
            `${i % 2 === 0 ? `<span class="num">${Math.floor(i/2)+1}.</span>` : ''}<span class="move">${m}</span> `
        ).join('');
    }

    const s = state.node?.stats || { w: 0, d: 0, l: 0 };
    const t = s.w + s.d + s.l;
    $('wins').textContent = s.w;
    $('draws').textContent = s.d;
    $('losses').textContent = s.l;

    const wp = t ? (s.w / t * 100) : 0;
    const dp = t ? (s.d / t * 100) : 0;
    $('bar').innerHTML = `<div class="win" style="width:${wp}%"></div><div class="draw" style="width:${dp}%"></div>`;

    state.moveQualities = {};
    state.moveEvals = {};
    state.engineBestSan = null;
    renderStudyPanel();
    renderMoves();
}

function renderMoves() {
    const moves = Object.entries(state.node?.moves || {})
        .map(([san, n]) => ({ san, ...n.stats, total: n.stats.w + n.stats.d + n.stats.l, mine: n.mine }))
        .sort((a, b) => b.total - a.total);

    const studyMoves = getStudyMoves();
    const userSans = new Set(moves.map(m => m.san));
    let html = '';

    // Engine best move (if not in user moves)
    if (state.engineBestSan && !userSans.has(state.engineBestSan)) {
        const ev = state.moveEvals[state.engineBestSan];
        html += `<div class="move-row engine-suggestion" onclick="window.playMove('${state.engineBestSan}')">
            <div class="move-san">${state.engineBestSan}</div>
            <span class="move-quality best">Best</span>
            <div class="move-info"><div class="move-games">Engine recommendation</div></div>
            ${ev !== undefined ? `<div class="move-eval">${ev > 0 ? '+' : ''}${ev.toFixed(1)}</div>` : ''}
        </div>`;
    }

    // Study-only moves
    const shownStudy = new Set();
    for (const sm of studyMoves) {
        if (userSans.has(sm.move) || shownStudy.has(sm.move)) continue;
        shownStudy.add(sm.move);
        const q = state.moveQualities[sm.move];
        const ev = state.moveEvals[sm.move];
        html += `<div class="move-row study-move" onclick="window.playMove('${sm.move}')">
            <div class="move-san">${sm.move}</div>
            <span class="move-quality study">📖</span>
            ${q ? `<span class="move-quality ${q.cls}">${q.label}</span>` : ''}
            <div class="move-info"><div class="move-games study-source">${sm.study}</div></div>
            ${ev !== undefined ? `<div class="move-eval">${ev > 0 ? '+' : ''}${ev.toFixed(1)}</div>` : ''}
        </div>`;
    }

    // User moves
    const t = (state.node?.stats?.w || 0) + (state.node?.stats?.d || 0) + (state.node?.stats?.l || 0);
    for (const m of moves) {
        const wp = (m.w / m.total * 100).toFixed(0);
        const dp = (m.d / m.total * 100).toFixed(0);
        const lp = (m.l / m.total * 100).toFixed(0);
        const freq = t > 0 ? (m.total / t * 100).toFixed(0) : 0;
        const q = state.moveQualities[m.san];
        const ev = state.moveEvals[m.san];
        const hasStudy = studyMoves.some(sm => sm.move === m.san);

        html += `<div class="${hasStudy ? 'move-row has-study' : 'move-row'}" onclick="window.playMove('${m.san}')">
            <div class="move-san">${m.san}</div>
            ${hasStudy ? '<span class="move-quality study">📖</span>' : ''}
            ${q ? `<span class="move-quality ${q.cls}">${q.label}</span>` : ''}
            <div class="move-info">
                <div class="move-games">${m.total} games · ${freq}% (${m.mine ? 'you' : 'opp'})</div>
                <div class="move-bar"><div class="win" style="width:${wp}%"></div><div class="draw" style="width:${dp}%"></div></div>
            </div>
            ${ev !== undefined ? `<div class="move-eval">${ev > 0 ? '+' : ''}${ev.toFixed(1)}</div>` : ''}
            <div class="move-pct"><span class="w">${wp}</span>/<span class="d">${dp}</span>/<span class="l">${lp}</span></div>
        </div>`;
    }

    $('moves').innerHTML = html || '<div class="empty">No games from this position</div>';
}

// =============================================================================
// Weaknesses
// =============================================================================

async function loadWeaknesses() {
    $('weaknessContent').innerHTML = '<div class="analyzing-status"><div class="spinner"></div><div class="analyzing-text">Loading...</div></div>';

    try {
        const r = await fetch('weaknesses.json');
        if (!r.ok) throw new Error();
        const data = await r.json();

        state.weaknesses = data.weaknesses.map(w => ({
            opening: w.opening, line: w.line.join(' '), lineArr: w.line,
            playedMove: w.played, bestMove: w.best, evalLoss: w.eval_loss,
            games: w.games, isWhite: w.is_white, mistakeType: w.mistake_type
        }));
        renderWeaknesses();
    } catch {
        $('weaknessContent').innerHTML = `<div class="empty"><div style="color:var(--yellow)">No analysis found</div>
            <div style="font-size:11px;margin-top:8px">Run <code>mise run analyze</code> to generate.</div></div>`;
    }
}

function renderWeaknesses() {
    const filtered = state.weaknesses.filter(w => w.isWhite === (state.color === 'white'))
        .sort((a, b) => b.games - a.games || b.evalLoss - a.evalLoss);

    if (!filtered.length) {
        $('weaknessCount').style.display = 'none';
        $('weaknessContent').innerHTML = `<div class="empty"><div style="color:var(--light-green)">✓ No weaknesses as ${state.color}!</div></div>`;
        return;
    }

    $('weaknessCount').textContent = `${filtered.length} issue${filtered.length > 1 ? 's' : ''}`;
    $('weaknessCount').style.display = 'block';

    const grouped = {};
    for (const w of filtered) {
        (grouped[w.opening] ||= []).push(w);
    }

    const types = { tactical: 'TAC', positional: 'POS', development: 'DEV', hanging_piece: 'HANG', pawn_structure: 'PAWN' };

    let html = '<div class="weakness-list">';
    for (const [opening, items] of Object.entries(grouped)) {
        html += `<div class="weakness-group"><div class="weakness-opening">${opening}</div>`;
        for (const w of items) {
            html += `<div class="weakness-item" onclick="window.goToWeakness(${JSON.stringify(w.lineArr).replace(/"/g, '&quot;')}, ${w.isWhite}, '${w.playedMove}', '${w.bestMove}', ${w.evalLoss})">
                <div class="weakness-moves">
                    <span class="weakness-played">${w.playedMove}</span>
                    <span class="weakness-arrow">→</span>
                    <span class="weakness-best">${w.bestMove}</span>
                </div>
                <div class="weakness-info">
                    <div class="weakness-line">...${w.lineArr.slice(-4).join(' ')}</div>
                    <div class="weakness-stats">${types[w.mistakeType] || ''}</div>
                </div>
                <div class="weakness-games">${w.games}×</div>
                <div class="weakness-eval">−${w.evalLoss.toFixed(1)}</div>
            </div>`;
        }
        html += '</div>';
    }
    html += '</div>';

    $('weaknessContent').innerHTML = html;
}

window.goToWeakness = function(lineArr, isWhite, playedMove, bestMove, evalLoss) {
    if ((isWhite && state.color !== 'white') || (!isWhite && state.color !== 'black')) {
        setColor(isWhite ? 'white' : 'black');
    }

    state.game = new Chess();
    state.history = [];
    state.node = state.tree;

    for (const m of lineArr) {
        state.game.move(m);
        state.history.push(m);
        state.node = state.node?.moves?.[m] || { moves: {}, stats: { w: 0, d: 0, l: 0 } };
    }

    state.cg.set({
        fen: state.game.fen(),
        orientation: isWhite ? 'white' : 'black',
        turnColor: state.game.turn() === 'w' ? 'white' : 'black',
        movable: { dests: getLegalMoves() },
        drawable: { autoShapes: getArrows() }
    });

    state.forwardStack = [];
    state.currentWeakness = { playedMove, bestMove, evalLoss };

    $('errorInfo').style.display = 'block';
    $('errorInfo').className = 'error-info';
    $('errorInfo').innerHTML = `
        <div class="error-info-header">
            <span class="error-info-title">Weakness Found</span>
            <button class="error-info-close" onclick="$('errorInfo').style.display='none'">×</button>
        </div>
        <div class="error-info-moves">
            <span>You played:</span> <span class="error-info-played">${playedMove}</span>
            <span class="error-info-arrow">→</span>
            <span>Best:</span> <span class="error-info-best">${bestMove}</span>
        </div>
        <div class="error-info-eval">Eval loss: <span>−${evalLoss.toFixed(1)}</span> pawns</div>
    `;

    render();
    analyzePosition();

    // Switch to explore tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector('[data-tab="explore"]').classList.add('active');
    $('exploreTab').classList.add('active');
};

// =============================================================================
// Init
// =============================================================================

function init() {
    $('loading').style.display = 'none';
    $('app').style.display = 'flex';

    const total = state.treeWhite.stats.w + state.treeWhite.stats.d + state.treeWhite.stats.l +
                  state.treeBlack.stats.w + state.treeBlack.stats.d + state.treeBlack.stats.l;
    $('meta').textContent = `${total} games · ${CONFIG.username}`;

    state.tree = state.treeWhite;
    state.node = state.tree;
    state.game = new Chess();

    state.cg = Chessground($('board'), {
        fen: state.game.fen(),
        orientation: 'white',
        movable: { free: false, color: 'both', dests: getLegalMoves() },
        draggable: { enabled: true },
        events: { move: (from, to) => {
            const move = state.game.move({ from, to, promotion: 'q' });
            if (move) { state.history.push(move.san); state.node = state.node?.moves?.[move.san] || { moves: {}, stats: { w: 0, d: 0, l: 0 } }; state.forwardStack = []; updateBoard(); render(); analyzePosition(); }
        }},
        drawable: { enabled: true, autoShapes: getArrows() }
    });

    $('resetBtn').onclick = reset;
    $('backBtn').onclick = back;
    $('whiteBtn').onclick = () => setColor('white');
    $('blackBtn').onclick = () => setColor('black');
    $('refreshBtn').onclick = async () => {
        const db = await openDB();
        const tx = db.transaction('cache', 'readwrite');
        tx.objectStore('cache').delete(CONFIG.cacheKey);
        tx.oncomplete = () => location.reload();
    };

    document.addEventListener('keydown', e => {
        if (e.key === 'ArrowLeft') { e.preventDefault(); back(); }
        if (e.key === 'ArrowRight') { e.preventDefault(); forward(); }
    });

    document.querySelectorAll('.tab').forEach(tab => {
        tab.onclick = () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            $(tab.dataset.tab + 'Tab').classList.add('active');
        };
    });

    render();
    initStockfish().then(() => analyzePosition());
    loadWeaknesses();
    loadStudyMoves();
}

window.playMove = playMove;
window.toggleWeaknessHelp = function() {
    const help = $('weaknessHelp');
    const content = help.querySelector('.weakness-help-content');
    help.classList.toggle('open');
    content.style.display = help.classList.contains('open') ? 'block' : 'none';
};

fetchGames().catch(e => { $('progress').textContent = 'Error: ' + e.message; });
