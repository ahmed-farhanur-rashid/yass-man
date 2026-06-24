/**
 * YASS-MAN frontend — plain fetch + DOM, no framework.
 *
 * State machine:
 *   home  ←→  loading  →  results (with optional answer)
 *                      →  error
 *                      →  empty
 */

'use strict';

// ── DOM refs ─────────────────────────────────────────────────────────────────
const homeView        = document.getElementById('home-view');
const homeForm        = document.getElementById('home-form');
const homeQuery       = document.getElementById('home-query');

const resultsView     = document.getElementById('results-view');
const navForm         = document.getElementById('nav-form');
const navQuery        = document.getElementById('nav-query');
const navLogoBtn      = document.getElementById('nav-logo-btn');

const loadingState    = document.getElementById('loading-state');
const errorState      = document.getElementById('error-state');
const errorMessage    = document.getElementById('error-message');
const btnRetry        = document.getElementById('btn-retry');
const emptyState      = document.getElementById('empty-state');

const answerSection   = document.getElementById('answer-section');
const answerText      = document.getElementById('answer-text');
const citationsList   = document.getElementById('citations-list');

const expandedEl      = document.getElementById('expanded-queries');
const expandedCount   = document.getElementById('expanded-count');
const expandedList    = document.getElementById('expanded-list');

const resultsList     = document.getElementById('results-list');

const latencyFooter   = document.getElementById('latency-footer');
const latencyStages   = document.getElementById('latency-stages');

// ── State ─────────────────────────────────────────────────────────────────────
let currentQuery    = '';
let currentQueryId  = null;

// ── Stage animation progress ──────────────────────────────────────────────────
const LOADING_STAGES = ['router', 'expand', 'search', 'rank', 'synth'];
let stageTimer = null;

function startLoadingAnimation() {
  let idx = 0;
  const stages = loadingState.querySelectorAll('.stage');
  stages.forEach(s => s.classList.remove('active', 'done'));
  stages[0].classList.add('active');

  stageTimer = setInterval(() => {
    if (idx < stages.length - 1) {
      stages[idx].classList.remove('active');
      stages[idx].classList.add('done');
      idx++;
      stages[idx].classList.add('active');
    }
  }, 420);
}

function stopLoadingAnimation() {
  if (stageTimer) { clearInterval(stageTimer); stageTimer = null; }
  loadingState.querySelectorAll('.stage').forEach(s => {
    s.classList.remove('active');
    s.classList.add('done');
  });
}

// ── View switching ────────────────────────────────────────────────────────────
function showHome() {
  homeView.classList.remove('hidden');
  resultsView.classList.add('hidden');
  homeQuery.value = '';
  homeQuery.focus();
}

function showResults(query) {
  homeView.classList.add('hidden');
  resultsView.classList.remove('hidden');
  navQuery.value = query;

  // Reset result sections
  loadingState.classList.remove('hidden');
  errorState.classList.add('hidden');
  emptyState.classList.add('hidden');
  answerSection.classList.add('hidden');
  expandedEl.classList.add('hidden');
  resultsList.innerHTML = '';
  latencyFooter.classList.add('hidden');
  answerText.innerHTML = '';
  citationsList.innerHTML = '';
  expandedList.innerHTML = '';

  startLoadingAnimation();
}

function showError(msg) {
  stopLoadingAnimation();
  loadingState.classList.add('hidden');
  errorState.classList.remove('hidden');
  errorMessage.textContent = msg || 'Something went wrong. Please try again.';
}

// ── Search ────────────────────────────────────────────────────────────────────
async function doSearch(query, pushHistory = true) {
  query = query.trim();
  if (!query) return;

  currentQuery = query;

  // Keep the URL bar in sync so searches are bookmarkable and shareable
  if (pushHistory) {
    const url = new URL(window.location.href);
    url.search = '?q=' + encodeURIComponent(query);
    window.history.pushState({ q: query }, '', url.toString());
  }

  showResults(query);

  let data;
  try {
    const url = `/search?q=${encodeURIComponent(query)}`;
    const resp = await fetch(url);

    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const err = await resp.json();
        detail = err.detail || detail;
      } catch {}
      throw new Error(detail);
    }

    data = await resp.json();
  } catch (err) {
    showError(err.message);
    return;
  }

  stopLoadingAnimation();
  loadingState.classList.add('hidden');

  currentQueryId = data.query_id;

  // ── Empty ─────────────────────────────────────────────────────────────────
  if (!data.results || data.results.length === 0) {
    emptyState.classList.remove('hidden');
    return;
  }

  // ── Answer ────────────────────────────────────────────────────────────────
  if (data.answer) {
    renderAnswer(data.answer, data.citations || []);
    answerSection.classList.remove('hidden');
  }

  // ── Expanded queries ──────────────────────────────────────────────────────
  if (data.expanded_queries && data.expanded_queries.length > 1) {
    expandedCount.textContent = `${data.expanded_queries.length}`;
    expandedList.innerHTML = '';
    data.expanded_queries.forEach((q, i) => {
      const li = document.createElement('li');
      if (i === 0) li.classList.add('original');
      li.textContent = q;
      expandedList.appendChild(li);
    });
    expandedEl.classList.remove('hidden');
  }

  // ── Results ───────────────────────────────────────────────────────────────
  renderResults(data.results, data.citations || []);

  // ── Latency ───────────────────────────────────────────────────────────────
  if (data.latency_ms) {
    renderLatency(data.latency_ms);
    latencyFooter.classList.remove('hidden');
  }
}

// ── Answer rendering ──────────────────────────────────────────────────────────
function renderAnswer(raw, citations) {
  // Replace [N] with clickable superscript badges
  const citationMap = {};
  citations.forEach(c => { citationMap[c.index] = c; });

  const html = raw.replace(/\[(\d+)\]/g, (match, n) => {
    const idx = parseInt(n, 10);
    const cit = citationMap[idx];
    if (!cit) return match;
    return `<a class="cite-link" href="${escapeHtml(cit.url)}" target="_blank" rel="noopener" title="${escapeHtml(cit.title)}">${idx}</a>`;
  });

  answerText.innerHTML = html;

  // Citation list
  citationsList.innerHTML = '';
  citations.forEach(c => {
    const item = document.createElement('div');
    item.className = 'citation-item';
    item.innerHTML = `
      <span class="citation-num">${c.index}</span>
      <a class="citation-link" href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>
    `;
    citationsList.appendChild(item);
  });
}

// ── Results rendering ─────────────────────────────────────────────────────────
function renderResults(results, citations) {
  // Build a URL → citation index map for quick highlight
  const citedUrls = new Set((citations || []).map(c => c.url));

  resultsList.innerHTML = '';
  results.forEach((result, i) => {
    const card = buildResultCard(result, i, citedUrls.has(result.url));
    resultsList.appendChild(card);
  });
}

function buildResultCard(result, index, isCited) {
  const domain = extractDomain(result.url);
  const scoreNorm = normalizeScore(result.score);
  const scoreColor = scoreToColor(scoreNorm);
  const scoreDisplay = result.score > 0
    ? result.score.toFixed(3)
    : (scoreNorm * 100).toFixed(0) + '%';

  const card = document.createElement('div');
  card.className = 'result-card';
  if (isCited) card.style.borderColor = 'rgba(62,207,207,0.2)';

  card.innerHTML = `
    <div class="result-card-header">
      <a class="result-title-link"
         href="${escapeHtml(result.url)}"
         target="_blank"
         rel="noopener"
         data-url="${escapeHtml(result.url)}"
         data-qid=""
      >${escapeHtml(result.title || domain)}</a>
      <div class="result-meta">
        <span class="result-domain">${escapeHtml(domain)}</span>
        <div class="score-badge" title="Relevance score: ${result.score.toFixed(4)}">
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width:${scoreNorm * 100}%; background:${scoreColor};"></div>
          </div>
          <span class="score-val">${scoreDisplay}</span>
        </div>
      </div>
    </div>
    ${result.snippet ? `<p class="result-snippet">${escapeHtml(result.snippet)}</p>` : ''}
    <div class="result-footer">
      <span class="result-url-small">${escapeHtml(result.url)}</span>
      <div class="feedback-btns">
        <button class="feedback-btn" data-signal="up"   data-url="${escapeHtml(result.url)}" title="Relevant">👍</button>
        <button class="feedback-btn" data-signal="down" data-url="${escapeHtml(result.url)}" title="Not relevant">👎</button>
      </div>
    </div>
  `;

  // Click tracking
  const titleLink = card.querySelector('.result-title-link');
  titleLink.addEventListener('click', () => {
    if (currentQueryId) {
      postClick(currentQueryId, result.url);
    }
  });

  // Feedback buttons
  card.querySelectorAll('.feedback-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (!currentQueryId) return;
      const signal = btn.dataset.signal;
      const url    = btn.dataset.url;
      postFeedback(currentQueryId, url, signal);

      // Visual feedback
      card.querySelectorAll('.feedback-btn').forEach(b => {
        b.classList.remove('voted-up', 'voted-down');
      });
      btn.classList.add(signal === 'up' ? 'voted-up' : 'voted-down');
    });
  });

  return card;
}

// ── Latency rendering ─────────────────────────────────────────────────────────
const LATENCY_TARGETS = {
  router:      10,
  expansion:   20,
  search:    1200,
  aggregation: 20,
  embedding:  100,
  rerank:     150,
  synthesis:  800,
  total:     2500,
};

function renderLatency(lat) {
  latencyStages.innerHTML = '';

  const stages = ['router', 'expansion', 'search', 'aggregation', 'embedding', 'rerank', 'synthesis', 'total'];
  stages.forEach(stage => {
    if (lat[stage] === undefined) return;
    const val = lat[stage];
    const target = LATENCY_TARGETS[stage] || Infinity;
    const cls = val < target * 0.8 ? 'fast' : val < target ? 'ok' : 'slow';
    const isTotal = stage === 'total';

    const chip = document.createElement('div');
    chip.className = 'latency-chip';
    if (isTotal) chip.style.borderColor = 'rgba(255,255,255,0.12)';

    chip.innerHTML = `
      <span class="latency-chip-name">${stage}</span>
      <span class="latency-chip-val ${cls}">${val.toFixed(0)}ms</span>
    `;
    latencyStages.appendChild(chip);
  });
}

// ── API calls ─────────────────────────────────────────────────────────────────
async function postFeedback(queryId, url, signal) {
  try {
    await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id: queryId, result_url: url, signal }),
    });
  } catch { /* silent — feedback failure shouldn't break UX */ }
}

async function postClick(queryId, url) {
  try {
    await fetch('/click', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id: queryId, result_url: url }),
    });
  } catch {}
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function extractDomain(url) {
  try {
    const host = new URL(url).hostname;
    return host.startsWith('www.') ? host.slice(4) : host;
  } catch {
    return url;
  }
}

function normalizeScore(score) {
  // Cross-encoder scores can be large logits; sigmoid-normalise for display
  if (score >= 0 && score <= 1) return score;
  // sigmoid
  return 1 / (1 + Math.exp(-score));
}

function scoreToColor(norm) {
  if (norm >= 0.75) return '#3ecfcf';        // high — cyan
  if (norm >= 0.45) return '#f5a623';        // mid  — amber
  return '#e15959';                          // low  — red
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ── Event listeners ───────────────────────────────────────────────────────────
homeForm.addEventListener('submit', e => {
  e.preventDefault();
  const q = homeQuery.value.trim();
  if (q) doSearch(q);
});

navForm.addEventListener('submit', e => {
  e.preventDefault();
  const q = navQuery.value.trim();
  if (q) doSearch(q);
});

navLogoBtn.addEventListener('click', () => {
  showHome();
});

btnRetry.addEventListener('click', () => {
  if (currentQuery) doSearch(currentQuery);
});

// Handle browser back/forward
window.addEventListener('popstate', () => {
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  if (q) doSearch(q, false);  // history already has this entry
  else showHome();
});

// ── Init ──────────────────────────────────────────────────────────────────────
(function init() {
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  if (q) {
    doSearch(q);
  } else {
    homeQuery.focus();
  }
})();
