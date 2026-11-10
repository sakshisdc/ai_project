/**
 * Adaptive Minimax Interactive Dashboard — app.js
 * Handles all interactivity for the Connect Four research visualization.
 *
 * Architecture:
 *   - gameState  : canonical game state (moves list, last response data)
 *   - renderBoard: draws the 6x7 grid from API data
 *   - API helpers: fetchState, postMove, postAnalyze, fetchResults
 *   - Tab logic  : Arena | Inspector | Comparator | Benchmarks
 */

'use strict';

/* ──────────────────────────────────────────
   1.  GLOBAL STATE
   ────────────────────────────────────────── */
const gameState = {
  moves: [],          // [int] — column indices played so far
  board: null,        // last board object from API
  features: null,     // last features object from API
  details: null,      // last agent details from /api/move
  info: null,         // last search info from /api/move
  currentBudget: 300,
  autoPlayInterval: null,
  isThinking: false,
};

/* ──────────────────────────────────────────
   1b. MATCH MODE + AUTOMATIC AI REPLY
   Human vs AI / AI vs Human  -> the AI moves by itself whenever it is its turn.
   Human vs Human / AI vs AI  -> unchanged (manual clicks / Step / Auto-Play).
   ────────────────────────────────────────── */
let autoAiToken = 0;   // bumped to cancel a pending automatic AI move (reset, undo, new request)
const sleep = ms => new Promise(r => setTimeout(r, ms));

function getPlayerType(color) {
  return document.getElementById(color === 1 ? 'p1-select' : 'p2-select')?.value ?? 'human';
}
function getCurrentPlayer() {
  return gameState.board?.current_player ?? ((gameState.moves.length % 2 === 0) ? 1 : 2);
}
function getMatchMode() {
  const h1 = getPlayerType(1) === 'human';
  const h2 = getPlayerType(2) === 'human';
  if (h1 && h2) return 'human-vs-human';
  if (!h1 && !h2) return 'ai-vs-ai';
  return h1 ? 'human-vs-ai' : 'ai-vs-human';
}
function isMixedMode() {
  const m = getMatchMode();
  return m === 'human-vs-ai' || m === 'ai-vs-human';
}
function updateModeBadge() {
  const el = document.getElementById('mode-badge');
  if (!el) return;
  const text = {
    'human-vs-human': '👥 Human vs Human — take turns clicking a column.',
    'ai-vs-ai':       '🤖 AI vs AI — use “Step AI Move” or “Auto-Play Match”.',
    'human-vs-ai':    '🎮 Human vs AI — the AI replies automatically after every move you make.',
    'ai-vs-human':    '🎮 AI vs Human — the AI opens the game and replies automatically after every move you make.',
  }[getMatchMode()];
  el.textContent = text;
  el.dataset.mode = getMatchMode();
}
async function maybeAutoAI(delay = 350) {
  if (!isMixedMode()) return;
  const token = ++autoAiToken;
  await sleep(delay);                                   // short pause so the human disc is visible
  if (token !== autoAiToken) return;                    // cancelled (reset / undo / newer request)
  if (gameState.isThinking || gameState.board?.is_terminal) return;
  if (!isMixedMode()) return;                           // user changed players meanwhile
  if (getPlayerType(getCurrentPlayer()) === 'human') return;
  await triggerAIMove();
}

// Figure captions map
const FIGURE_CAPTIONS = {
  '03_budget_curves.png':
    'Figure 3: Decision regret vs node budget (left), agreement with depth-7 reference (center), and regret efficiency per node actually used (right). Agent F achieves the lowest regret at all budgets above 300 nodes.',
  '04_tournament.png':
    'Figure 4: Head-to-head tournament scores (60 games × 2 colors per matchup). Each cell shows Agent-row win fraction vs Agent-col. Agent F beats all baselines; Agent E ties Agent F closely.',
  '05_allocation.png':
    'Figure 5: Search allocation — nodes actually used and deepest ply reached by position type (early/mid/late game). VOC agents search deeper in tactical positions while respecting the node cap.',
  '06_ablation.png':
    'Figure 6: Ablation study of Agent F components at 1000 nodes. Removing the learned error model (−σ) or VOC scheduling (+vanilla) degrades performance incrementally, validating each component.',
  '01_classifier_comparison.png':
    'Figure 1: Group-CV vs held-out test AUC for tactical classifier candidates. The gradient boosting model (GBM) achieves the best generalization without overfitting to the training set.',
  '02_error_models.png':
    'Figure 2: Gaussian NLL and 90% coverage calibration for the three error models. The learned heteroscedastic GBM model achieves the best NLL while maintaining calibrated coverage bands.',
  '07_sigma_calibration.png':
    'Figure 7: Predicted σ(x) vs observed residual standard deviation, binned by phase and absolute eval. Well-calibrated predictions fall on the diagonal; the GBM model tracks it closely throughout the game.',
};

/* ──────────────────────────────────────────
   2.  API HELPERS
   ────────────────────────────────────────── */
async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} — ${res.statusText}`);
  return res.json();
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { const j = await res.json(); msg = j.error || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

async function fetchState(moves) {
  const qs = moves.length ? moves.join(',') : '';
  return apiGet(`/api/state?moves=${qs}`);
}

async function postMove(moves, agent, budget) {
  return apiPost('/api/move', { moves, agent, budget });
}

async function postAnalyze(moves, budget) {
  return apiPost('/api/analyze', { moves, budget });
}

async function fetchResults() {
  return apiGet('/api/results');
}

/* ──────────────────────────────────────────
   3.  BOARD RENDERING
   ────────────────────────────────────────── */
function renderBoard(boardData) {
  const container = document.getElementById('connect4-board');
  if (!container || !boardData) return;

  const grid = boardData.grid; // 6 rows × 7 cols, row 0 = top
  const legal = new Set(boardData.legal_moves || []);

  // Build / reuse cells — only rebuild if count is wrong
  if (container.children.length !== 42) {
    container.innerHTML = '';
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 7; c++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.row = r;
        cell.dataset.col = c;
        container.appendChild(cell);
      }
    }
  }

  const cells = container.querySelectorAll('.cell');
  cells.forEach(cell => {
    const r = parseInt(cell.dataset.row);
    const c = parseInt(cell.dataset.col);
    const val = grid[r][c];
    cell.className = 'cell' + (val === 1 ? ' red' : val === 2 ? ' yellow' : '');
  });

  // Update drop buttons
  const btns = document.querySelectorAll('.col-drop-btn');
  btns.forEach(btn => {
    const col = parseInt(btn.dataset.col);
    btn.disabled = boardData.is_terminal || !legal.has(col) || gameState.isThinking;
  });

  // Update column hover hints
  container.querySelectorAll('.cell').forEach(cell => {
    cell.addEventListener('mouseenter', onCellHover, { passive: true });
    cell.addEventListener('mouseleave', onCellUnhover, { passive: true });
  });
}

function onCellHover(e) {
  const col = parseInt(e.currentTarget.dataset.col);
  if (gameState.board && (gameState.board.is_terminal || gameState.isThinking)) return;
  document.querySelectorAll(`.cell[data-col="${col}"]`).forEach(c => {
    if (!c.classList.contains('red') && !c.classList.contains('yellow')) {
      c.classList.add('hover-hint');
    }
  });
}

function onCellUnhover() {
  document.querySelectorAll('.cell.hover-hint').forEach(c => c.classList.remove('hover-hint'));
}

function animateDrop(col) {
  // find the lowest empty cell in col
  const rows = [5, 4, 3, 2, 1, 0]; // bottom to top (row 5 = bottom visually)
  for (const r of rows) {
    const cell = document.querySelector(`.cell[data-row="${r}"][data-col="${col}"]`);
    if (cell && !cell.classList.contains('red') && !cell.classList.contains('yellow')) {
      cell.classList.add('drop-anim');
      cell.addEventListener('animationend', () => cell.classList.remove('drop-anim'), { once: true });
      break;
    }
  }
}

/* ──────────────────────────────────────────
   4.  TURN INDICATOR & STATUS
   ────────────────────────────────────────── */
function updateTurnIndicator(boardData) {
  const indicator = document.getElementById('turn-indicator');
  const statusText = document.getElementById('game-status-text');
  if (!indicator || !boardData) return;

  if (boardData.is_terminal) {
    if (boardData.winner === 0) {
      indicator.className = 'player-turn-badge turn-done';
      indicator.innerHTML = '🤝 Draw!';
      statusText.textContent = 'Game over — it\'s a draw.';
    } else {
      const color = boardData.winner === 1 ? 'Red' : 'Yellow';
      const cls   = boardData.winner === 1 ? 'turn-red' : 'turn-yellow';
      indicator.className = `player-turn-badge ${cls}`;
      indicator.innerHTML = `<span class="disc-dot ${boardData.winner === 1 ? 'red' : 'yellow'}"></span> ${color} Wins! 🎉`;
      statusText.textContent = `Game over — ${color} player wins!`;
    }
    stopAutoPlay();
  } else {
    const player = boardData.current_player;
    const cls    = player === 1 ? 'turn-red' : 'turn-yellow';
    const color  = player === 1 ? 'Red' : 'Yellow';
    indicator.className = `player-turn-badge ${cls}`;
    indicator.innerHTML = `<span class="disc-dot ${player === 1 ? 'red' : 'yellow'}"></span> ${color}'s Turn`;
    if (isMixedMode()) {
      statusText.textContent = getPlayerType(player) === 'human'
        ? 'Your turn — click a column.'
        : 'AI is about to move…';
    } else if (getMatchMode() === 'human-vs-human') {
      statusText.textContent = 'Click a column to drop a disc.';
    } else {
      statusText.textContent = 'Use “Step AI Move” or “Auto-Play Match”.';
    }
  }
}

/* ──────────────────────────────────────────
   5.  MOVE STATS PANEL (Arena tab)
   ────────────────────────────────────────── */
function updateStatsPanel(data) {
  if (!data) return;
  const { info, details, move } = data;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val ?? '—'; };

  set('stat-agent',  details?.name ?? '—');
  set('stat-move',   move !== undefined ? `Col ${move + 1}` : '—');
  set('stat-nodes',  info?.nodes?.toLocaleString() ?? '—');
  set('stat-depth',  info?.depth ?? '—');
  set('stat-ply',    info?.max_ply ?? '—');
  set('stat-time',   info?.time !== undefined ? `${(info.time * 1000).toFixed(1)}ms` : '—');

  const note = document.getElementById('strategy-note');
  if (note) note.textContent = details?.description ?? '';
}

/* ──────────────────────────────────────────
   6.  MOVE HISTORY
   ────────────────────────────────────────── */
function appendMoveHistory(col, playerColor) {
  const scroll = document.getElementById('history-scroll');
  if (!scroll) return;

  // Clear empty-history message
  const empty = scroll.querySelector('.empty-history');
  if (empty) empty.remove();

  const chip = document.createElement('span');
  chip.className = `move-chip ${playerColor === 1 ? 'red-chip' : 'yellow-chip'}`;
  chip.innerHTML = `<span class="disc-dot ${playerColor === 1 ? 'red' : 'yellow'}"></span>${col + 1}`;
  scroll.appendChild(chip);
  scroll.scrollLeft = scroll.scrollWidth;
}

function rebuildMoveHistory(moves) {
  const scroll = document.getElementById('history-scroll');
  if (!scroll) return;
  scroll.innerHTML = '';
  if (!moves.length) {
    scroll.innerHTML = '<div class="empty-history">Game started. No moves yet.</div>';
  } else {
    moves.forEach((col, idx) => {
      const playerColor = (idx % 2 === 0) ? 1 : 2; // P1 red plays first (idx 0,2,4...)
      const chip = document.createElement('span');
      chip.className = `move-chip ${playerColor === 1 ? 'red-chip' : 'yellow-chip'}`;
      chip.innerHTML = `<span class="disc-dot ${playerColor === 1 ? 'red' : 'yellow'}"></span>${col + 1}`;
      scroll.appendChild(chip);
    });
    scroll.scrollLeft = scroll.scrollWidth;
  }
  const counter = document.getElementById('ply-counter');
  if (counter) counter.textContent = `${moves.length} ${moves.length === 1 ? 'ply' : 'plies'}`;
}

/* ──────────────────────────────────────────
   7.  HUMAN COLUMN DROP
   ────────────────────────────────────────── */
async function humanDrop(col) {
  if (gameState.isThinking) return;
  if (gameState.board?.is_terminal) return;

  // Determine whose turn it is
  const board = gameState.board;
  const playerColor = board ? board.current_player : ((gameState.moves.length % 2 === 0) ? 1 : 2);
  const selectId = playerColor === 1 ? 'p1-select' : 'p2-select';
  const agentChoice = document.getElementById(selectId)?.value ?? 'human';

  if (agentChoice !== 'human') {
    // AI is playing this turn — trigger AI move instead
    await triggerAIMove();
    return;
  }

  // Human move
  const prevColor = playerColor; // player who is moving
  gameState.moves.push(col);
  animateDrop(col);

  gameState.isThinking = true;
  setButtonsDisabled(true);
  let moved = false;
  try {
    const data = await fetchState(gameState.moves);
    gameState.board = data.board;
    gameState.features = data.features;
    renderBoard(data.board);
    updateTurnIndicator(data.board);
    appendMoveHistory(col, prevColor);
    rebuildMoveHistory(gameState.moves);
    updateInspectorFromFeatures(data.features, null, null);
    moved = true;
  } catch (e) {
    gameState.moves.pop();
    showError(`Move failed: ${e.message}`);
  } finally {
    gameState.isThinking = false;
    setButtonsDisabled(false);
  }
  // Human vs AI / AI vs Human: the AI answers by itself (no "Step AI Move" click needed).
  if (moved) maybeAutoAI();
}

/* ──────────────────────────────────────────
   8.  AI MOVE (Step)
   ────────────────────────────────────────── */
async function triggerAIMove() {
  if (gameState.isThinking) return;
  if (gameState.board?.is_terminal) {
    showError('Game is over. Start a new game to continue.');
    return;
  }

  const board = gameState.board || { current_player: 1 };
  const playerColor = board.current_player ?? ((gameState.moves.length % 2 === 0) ? 1 : 2);
  const selectId = playerColor === 1 ? 'p1-select' : 'p2-select';
  const agentName = document.getElementById(selectId)?.value ?? 'F';

  if (agentName === 'human') {
    showError('Current player is set to Human. Click a column to drop a disc.');
    return;
  }

  gameState.isThinking = true;
  setButtonsDisabled(true);

  const statusText = document.getElementById('game-status-text');
  if (statusText) statusText.textContent = `Agent ${agentName} is thinking…`;

  try {
    const data = await postMove(gameState.moves, agentName, gameState.currentBudget);

    const col = data.move;
    animateDrop(col);

    gameState.moves = data.moves;
    gameState.board = data.next_board;
    gameState.features = data.features;
    gameState.details = data.details;
    gameState.info = data.info;

    renderBoard(data.next_board);
    updateTurnIndicator(data.next_board);
    updateStatsPanel(data);
    rebuildMoveHistory(gameState.moves);
    updateInspectorFromFeatures(data.features, data.details, data.info);
  } catch (e) {
    showError(`AI move failed: ${e.message}`);
  } finally {
    gameState.isThinking = false;
    setButtonsDisabled(false);
  }
}

/* ──────────────────────────────────────────
   9.  AUTO-PLAY
   ────────────────────────────────────────── */
function getAutoPlayDelay() {
  const slider = document.getElementById('speed-slider');
  return slider ? parseInt(slider.value) : 700;
}

function startAutoPlay() {
  if (gameState.autoPlayInterval) return;
  const btn = document.getElementById('btn-ai-autoplay');
  if (btn) { btn.textContent = '⏹ Stop Auto-Play'; btn.classList.add('btn-secondary'); btn.classList.remove('btn-outline'); }

  async function step() {
    if (gameState.board?.is_terminal) { stopAutoPlay(); return; }

    // Check if current player is human → stop
    const board = gameState.board || { current_player: 1 };
    const playerColor = board.current_player ?? 1;
    const selectId = playerColor === 1 ? 'p1-select' : 'p2-select';
    const agentName = document.getElementById(selectId)?.value ?? 'human';
    if (agentName === 'human') { stopAutoPlay(); return; }

    await triggerAIMove();
    if (!gameState.board?.is_terminal) {
      gameState.autoPlayInterval = setTimeout(step, getAutoPlayDelay());
    } else {
      stopAutoPlay();
    }
  }

  gameState.autoPlayInterval = setTimeout(step, 150);
}

function stopAutoPlay() {
  if (gameState.autoPlayInterval) {
    clearTimeout(gameState.autoPlayInterval);
    gameState.autoPlayInterval = null;
  }
  const btn = document.getElementById('btn-ai-autoplay');
  if (btn) { btn.textContent = '▶▶ Auto-Play Match'; btn.classList.remove('btn-secondary'); btn.classList.add('btn-outline'); }
}

/* ──────────────────────────────────────────
   10. UNDO & RESET
   ────────────────────────────────────────── */
async function undoMove() {
  if (gameState.isThinking || !gameState.moves.length) return;
  stopAutoPlay();
  autoAiToken++;                                   // cancel any pending automatic AI move
  gameState.moves.pop();
  if (isMixedMode()) {
    // Undo a whole turn: keep popping until it is the human's turn again (or the board is empty).
    while (gameState.moves.length &&
           getPlayerType(gameState.moves.length % 2 === 0 ? 1 : 2) !== 'human') {
      gameState.moves.pop();
    }
  }

  try {
    const data = await fetchState(gameState.moves);
    gameState.board = data.board;
    gameState.features = data.features;
    gameState.details = null;
    gameState.info = null;
    renderBoard(data.board);
    updateTurnIndicator(data.board);
    rebuildMoveHistory(gameState.moves);
    updateInspectorFromFeatures(data.features, null, null);
  } catch (e) {
    showError(`Undo failed: ${e.message}`);
  }
}

async function resetGame() {
  stopAutoPlay();
  autoAiToken++;
  gameState.moves = [];
  gameState.details = null;
  gameState.info = null;

  try {
    const data = await fetchState([]);
    gameState.board = data.board;
    gameState.features = data.features;
    renderBoard(data.board);
    updateTurnIndicator(data.board);
    rebuildMoveHistory([]);
    updateStatsPanel(null);
    updateInspectorFromFeatures(data.features, null, null);
    clearError();
    updateModeBadge();
    updateTurnIndicator(data.board);
    maybeAutoAI(500);                  // AI vs Human: the AI plays the first move by itself
  } catch (e) {
    showError(`Reset failed: ${e.message}`);
  }
}

/* ──────────────────────────────────────────
   11. INSPECTOR TAB — FEATURES & DEEP DIVE
   ────────────────────────────────────────── */
function updateInspectorFromFeatures(features, details, info) {
  if (!features) return;

  // Feature rows
  const fmap = {
    'f-phase':     features.phase,
    'f-branching': features.branching,
    'f-eval':      features.static_eval,
    'f-abseval':   features.abs_eval,
    'f-mywin':     features.my_win_now,
    'f-oppwin':    features.opp_win_now,
    'f-mythrees':  features.my_open_threes,
    'f-oppthrees': features.opp_open_threes,
    'f-swing':     features.swing,
  };
  for (const [id, val] of Object.entries(fmap)) {
    const el = document.getElementById(id);
    if (el) el.textContent = val !== undefined && val !== null ? Number(val).toFixed(4) : '—';
  }

  // Eval meter — maps static_eval from [-0.8, +0.8] to [0, 100]%
  const evalFill = document.getElementById('eval-meter-fill');
  if (evalFill && features.static_eval !== undefined) {
    const ev = Math.max(-0.8, Math.min(0.8, features.static_eval));
    const pct = ((ev + 0.8) / 1.6) * 100;
    // Show relative to center (50%)
    if (pct >= 50) {
      evalFill.style.left  = '50%';
      evalFill.style.width = `${pct - 50}%`;
      evalFill.style.background = 'linear-gradient(90deg, var(--blue), var(--red))';
    } else {
      evalFill.style.left  = `${pct}%`;
      evalFill.style.width = `${50 - pct}%`;
      evalFill.style.background = 'linear-gradient(90deg, var(--yellow), var(--blue))';
    }
  }

  // Deep dive panel
  updateDeepDive(details, info);
}

function updateDeepDive(details, info) {
  const tagEl  = document.getElementById('deep-agent-tag');
  const descEl = document.getElementById('deep-agent-desc');
  const panel  = document.getElementById('strategy-specific-panel');
  const compBadge = document.getElementById('compliance-badge');
  const compBar   = document.getElementById('compliance-bar-fill');
  const nodesCap  = document.getElementById('nodes-caption');

  if (!details) {
    if (tagEl)  tagEl.textContent  = 'No move yet';
    if (descEl) descEl.textContent = 'Make a move to see agent reasoning.';
    if (panel)  panel.innerHTML    = '';
    return;
  }

  if (tagEl)  tagEl.textContent  = `Agent ${details.name ?? '?'}`;
  if (descEl) descEl.textContent = details.label ?? '';

  // Strategy-specific panel HTML
  let html = '';
  const row = (label, val, extra = '') =>
    `<div class="ssp-row"><span class="ssp-label">${label}</span><span class="ssp-val">${val}</span></div>${extra}`;

  const agentName = details.name ?? '';

  if (agentName === 'A') {
    html  = row('Strategy', 'Fixed-Depth Alpha-Beta');
    html += row('Calibrated Depth', details.calibrated_depth ?? '—');
    html += `<div class="ssp-desc">${details.description ?? ''}</div>`;
  } else if (agentName === 'B') {
    html  = row('Strategy', 'Iterative Deepening AB');
    html += row('Max Depth Reached', info?.depth ?? '—');
    html += `<div class="ssp-desc">${details.description ?? ''}</div>`;
  } else if (agentName === 'C') {
    const extended = details.horizon_extended ? '✅ Yes' : '❌ No';
    html  = row('Strategy', 'Rule-Based Adaptive Cutoff');
    html += row('Swing Threshold', details.swing_threshold ?? '—');
    html += row('Observed Eval Swing', details.observed_swing ?? '—');
    html += row('Tactical Triggers (wins)', details.tactical_triggers ?? '—');
    html += row('Horizon Extended?', extended);
    html += `<div class="ssp-desc">${details.description ?? ''}</div>`;
  } else if (agentName === 'D') {
    const prob = details.ml_probability !== undefined ? (details.ml_probability * 100).toFixed(1) + '%' : '—';
    const thr  = details.threshold !== undefined ? (details.threshold * 100).toFixed(1) + '%' : '—';
    const ext  = details.horizon_extended ? '✅ Extend search' : '❌ Stop at budget';
    // Mini probability bar
    const barPct = details.ml_probability !== undefined ? (details.ml_probability * 100).toFixed(1) : 0;
    const barColor = details.horizon_extended ? 'var(--green)' : 'var(--text-dim)';
    html  = row('Strategy', 'ML Adaptive Cutoff (GBM)');
    html += row('P(extend search)', prob);
    html += row('Decision Threshold', thr);
    html += `<div class="ssp-row"><span class="ssp-label">Decision</span><span class="ssp-val">${ext}</span></div>`;
    html += `<div style="margin-top:0.5rem;">
               <div style="height:6px;background:var(--bg);border-radius:3px;overflow:hidden;border:1px solid var(--border-soft)">
                 <div style="width:${barPct}%;height:100%;background:${barColor};border-radius:3px;transition:width 0.4s"></div>
               </div>
               <div style="font-size:0.68rem;color:var(--text-dim);margin-top:2px">Probability of beneficial extension (threshold at ${thr})</div>
             </div>`;
    html += `<div class="ssp-desc">${details.description ?? ''}</div>`;
  } else if (agentName === 'E' || agentName === 'F') {
    const mu    = details.mu    !== undefined ? details.mu.toFixed(4)    : '—';
    const sigma = details.sigma !== undefined ? details.sigma.toFixed(4) : '—';
    html  = row('Strategy', `VOC Best-First Search`);
    html += row('Model Kind', details.model_kind ?? '—');
    html += row('Predicted μ (residual)', mu);
    html += row('Predicted σ (uncertainty)', sigma);
    html += row('k_root (exploration)', details.k_root ?? '—');
    html += row('λ (discount)', details.lam ?? '—');
    html += `<div class="ssp-desc">${details.description ?? ''}</div>`;
  } else {
    html = `<div class="ssp-desc">${details.description ?? ''}</div>`;
  }

  if (panel) panel.innerHTML = html;

  // Budget compliance
  if (info && compBar && compBadge && nodesCap) {
    const used   = info.nodes ?? 0;
    const budget = gameState.currentBudget;
    const pct    = Math.min(100, (used / budget) * 100);
    compBar.style.width = `${pct.toFixed(1)}%`;
    const compliant = info.within_budget !== false;
    compBadge.className = `badge ${compliant ? 'badge-green' : 'badge-red'}`;
    compBadge.textContent = compliant ? '✓ Within Budget' : '⚠ Over Budget';
    nodesCap.textContent  = `${used.toLocaleString()} nodes used / ${budget.toLocaleString()} budget cap`;
  }
}

/* ──────────────────────────────────────────
   12. COMPARATOR TAB
   ────────────────────────────────────────── */
async function runComparison() {
  const btn = document.getElementById('btn-run-comparison');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Running…'; }

  try {
    const data = await postAnalyze(gameState.moves, gameState.currentBudget);
    renderComparatorResults(data);
  } catch (e) {
    const grid = document.getElementById('comparison-cards-grid');
    if (grid) grid.innerHTML = `<div class="error-toast" style="grid-column:1/-1">Error: ${e.message}</div>`;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '⚡ Compare All 6 Agents'; }
  }
}

function renderComparatorResults(data) {
  const grid    = document.getElementById('comparison-cards-grid');
  const banner  = document.getElementById('consensus-banner');
  const cTitle  = document.getElementById('consensus-title');
  const cDetail = document.getElementById('consensus-detail');

  if (!grid) return;
  grid.innerHTML = '';

  const agentOrder = ['A', 'B', 'C', 'D', 'E', 'F'];
  const consensusMove = data.consensus_move;
  const agreement     = data.consensus_agreement;

  // Consensus banner
  if (banner && cTitle && cDetail) {
    banner.style.display = 'flex';
    cTitle.textContent   = `Consensus Move: Column ${consensusMove + 1}`;
    cDetail.textContent  = `${agreement} agree on column ${consensusMove + 1}. ${
      agreement.startsWith('6') ? 'Perfect consensus — all agents agree!' :
      agreement.startsWith('5') ? 'Strong consensus — minority dissent.' :
      'Split decision — agents disagree.'
    }`;
  }

  for (const key of agentOrder) {
    const res = data.agents?.[key];
    if (!res) continue;

    const { move, info, details } = res;
    const isConsensus = (move === consensusMove);

    const card = document.createElement('div');
    card.className = `comparison-card ${isConsensus ? 'is-consensus' : 'is-dissent'}`;

    // Strategy-specific key metric
    let keyMetric = '';
    const d = details || {};
    if (key === 'A') keyMetric = `Depth: ${d.calibrated_depth ?? '—'}`;
    else if (key === 'B') keyMetric = `Depth: ${info?.depth ?? '—'}`;
    else if (key === 'C') keyMetric = `Extended: ${d.horizon_extended ? 'Yes' : 'No'}`;
    else if (key === 'D') keyMetric = `P(ext): ${d.ml_probability !== undefined ? (d.ml_probability * 100).toFixed(0) + '%' : '—'}`;
    else if (key === 'E') keyMetric = `σ: ${d.sigma ?? '—'}`;
    else if (key === 'F') keyMetric = `σ: ${d.sigma ?? '—'}`;

    card.innerHTML = `
      <div class="cc-header">
        <span class="cc-agent-name">Agent ${key}</span>
        ${isConsensus ? '<span class="badge badge-green">✓ Consensus</span>' : '<span class="badge badge-outline">Dissent</span>'}
      </div>
      <div class="cc-agent-label">${d.label ?? ''}</div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.35rem">
        <span style="font-size:0.75rem;color:var(--text-muted)">Chosen Column</span>
        <span class="cc-move-badge">${move + 1}</span>
      </div>
      <div class="cc-stats">
        <div class="cc-stat-row"><span>Nodes Used</span><span class="cc-stat-val">${(info?.nodes ?? 0).toLocaleString()}</span></div>
        <div class="cc-stat-row"><span>Search Depth</span><span class="cc-stat-val">${info?.depth ?? '—'}</span></div>
        <div class="cc-stat-row"><span>Max Ply</span><span class="cc-stat-val">${info?.max_ply ?? '—'}</span></div>
        <div class="cc-stat-row"><span>Time</span><span class="cc-stat-val">${info?.time !== undefined ? (info.time * 1000).toFixed(1) + 'ms' : '—'}</span></div>
        <div class="cc-stat-row"><span>${key}</span><span class="cc-stat-val">${keyMetric}</span></div>
      </div>
    `;
    grid.appendChild(card);
  }
}

/* ──────────────────────────────────────────
   13. BENCHMARKS TAB
   ────────────────────────────────────────── */
let benchmarksLoaded = false;

async function loadBenchmarks() {
  if (benchmarksLoaded) return;
  benchmarksLoaded = true;
  // Figures are statically listed in HTML; just ensure the default image loads
  const img = document.getElementById('active-figure-img');
  if (img) {
    img.onerror = () => {
      img.alt = 'Figure not found. Run scripts/05_make_plots.py to generate figures.';
    };
  }
}

function switchFigure(figName) {
  const img     = document.getElementById('active-figure-img');
  const caption = document.getElementById('active-figure-caption');
  if (img)     img.src         = `/figures/${figName}`;
  if (caption) caption.textContent = FIGURE_CAPTIONS[figName] ?? figName;

  document.querySelectorAll('.fig-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.fig === figName);
  });
}

/* ──────────────────────────────────────────
   14. UTILITY — DISABLE CONTROLS
   ────────────────────────────────────────── */
function setButtonsDisabled(disabled) {
  const ids = ['btn-undo', 'btn-reset', 'btn-ai-step', 'btn-ai-autoplay', 'btn-run-comparison'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
  });
  document.querySelectorAll('.col-drop-btn').forEach(btn => {
    if (disabled) {
      btn.disabled = true;
    } else {
      // Re-enable only legal columns
      const col = parseInt(btn.dataset.col);
      const legal = new Set(gameState.board?.legal_moves ?? []);
      btn.disabled = gameState.board?.is_terminal || !legal.has(col);
    }
  });
}

/* ──────────────────────────────────────────
   15. ERROR DISPLAY
   ────────────────────────────────────────── */
function showError(msg) {
  const status = document.getElementById('game-status-text');
  if (status) {
    status.textContent = `⚠ ${msg}`;
    status.style.color = 'var(--red)';
    setTimeout(() => { status.style.color = ''; }, 4000);
  }
  console.warn('[Dashboard]', msg);
}

function clearError() {
  const status = document.getElementById('game-status-text');
  if (status) { status.style.color = ''; }
}

/* ──────────────────────────────────────────
   16. TAB SWITCHING
   ────────────────────────────────────────── */
function initTabs() {
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
      btn.classList.add('active');
      const section = document.getElementById(`tab-${tab}`);
      if (section) section.classList.add('active');

      if (tab === 'benchmarks') loadBenchmarks();
      if (tab === 'inspector')  updateInspectorFromFeatures(gameState.features, gameState.details, gameState.info);
    });
  });
}

/* ──────────────────────────────────────────
   17. SPEED SLIDER
   ────────────────────────────────────────── */
function initSpeedSlider() {
  const slider  = document.getElementById('speed-slider');
  const display = document.getElementById('speed-display');
  if (!slider || !display) return;

  const updateDisplay = () => {
    const ms = parseInt(slider.value);
    display.textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
  };
  slider.addEventListener('input', updateDisplay);
  updateDisplay();
}

/* ──────────────────────────────────────────
   18. BUDGET PILLS
   ────────────────────────────────────────── */
function initBudgetPills() {
  document.querySelectorAll('.budget-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.budget-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      gameState.currentBudget = parseInt(pill.dataset.budget);
    });
  });
}

/* ──────────────────────────────────────────
   19. WIRE UP ALL BUTTONS
   ────────────────────────────────────────── */
function initArenaControls() {
  // Column drop buttons
  document.querySelectorAll('.col-drop-btn').forEach(btn => {
    btn.addEventListener('click', () => humanDrop(parseInt(btn.dataset.col)));
  });

  // Cell click (convenience — same as column button)
  document.getElementById('connect4-board')?.addEventListener('click', e => {
    const cell = e.target.closest('.cell');
    if (cell) humanDrop(parseInt(cell.dataset.col));
  });

  // Changing a player type updates the mode; if it is now the AI's turn in a Human/AI match it plays.
  ['p1-select', 'p2-select'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      updateModeBadge();
      updateTurnIndicator(gameState.board);
      maybeAutoAI(250);
    });
  });

  // Undo
  document.getElementById('btn-undo')?.addEventListener('click', undoMove);

  // Reset
  document.getElementById('btn-reset')?.addEventListener('click', resetGame);

  // AI step
  document.getElementById('btn-ai-step')?.addEventListener('click', triggerAIMove);

  // Auto-play toggle
  document.getElementById('btn-ai-autoplay')?.addEventListener('click', () => {
    if (gameState.autoPlayInterval) {
      stopAutoPlay();
    } else {
      startAutoPlay();
    }
  });
}

function initComparatorControls() {
  document.getElementById('btn-run-comparison')?.addEventListener('click', runComparison);
}

function initBenchmarkControls() {
  document.querySelectorAll('.fig-tab').forEach(btn => {
    btn.addEventListener('click', () => switchFigure(btn.dataset.fig));
  });
}

/* ──────────────────────────────────────────
   20. BOOTSTRAP — INITIALIZE EVERYTHING
   ────────────────────────────────────────── */
async function initBoard() {
  try {
    const data = await fetchState([]);
    gameState.board    = data.board;
    gameState.features = data.features;
    renderBoard(data.board);
    updateTurnIndicator(data.board);
    rebuildMoveHistory([]);
    updateInspectorFromFeatures(data.features, null, null);
    updateModeBadge();
    updateTurnIndicator(data.board);
    maybeAutoAI(600);
  } catch (e) {
    showError(`Failed to connect to server: ${e.message}. Is serve.py running?`);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSpeedSlider();
  initBudgetPills();
  initArenaControls();
  initComparatorControls();
  initBenchmarkControls();
  initBoard();
});
