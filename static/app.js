/* ═══════════════════════════════════════════════════════════
   ML Feature Store — Dashboard JavaScript
   Auto-polls /health and /recent every 3 seconds to
   display live system metrics and transaction feed.
   ═══════════════════════════════════════════════════════════ */

const API = '';  // Same origin (served by FastAPI)
const POLL_INTERVAL = 3000;   // 3 seconds
const MAX_FEED_ROWS = 50;     // Max rows shown in feed table

let previousPredictions = null;
let lastSeenIds = new Set();

// ─── Startup ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateClock();
    setInterval(updateClock, 1000);
    pollHealth();
    pollRecent();
    setInterval(pollHealth, POLL_INTERVAL);
    setInterval(pollRecent, POLL_INTERVAL);
});


// ─── Clock ─────────────────────────────────────────────────
function updateClock() {
    const el = document.getElementById('clock');
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-US', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
}


// ─── Health Polling ────────────────────────────────────────
async function pollHealth() {
    try {
        const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderHealth(data);
        setConnectionStatus('online', 'Connected');
    } catch (err) {
        setConnectionStatus('offline', 'Disconnected');
        console.warn('Health poll failed:', err.message);
    }
}


function renderHealth(data) {
    // ── Metric cards ──
    const totalPred = data.total_predictions || 0;
    const totalFraud = data.total_frauds_detected || 0;
    const fraudPct = totalPred > 0 ? ((totalFraud / totalPred) * 100).toFixed(3) : '0.000';

    animateNumber('total-predictions', totalPred);
    animateNumber('total-frauds', totalFraud);
    document.getElementById('fraud-rate').textContent = `${fraudPct}% of all transactions`;

    // Rate calculation (rough per-second since last poll)
    if (previousPredictions !== null && totalPred > previousPredictions) {
        const delta = totalPred - previousPredictions;
        const rate = (delta / (POLL_INTERVAL / 1000)).toFixed(1);
        document.getElementById('predictions-rate').textContent = `${rate} predictions/sec`;
    }
    previousPredictions = totalPred;

    // ── Health dots ──
    setHealthDot('health-model',   data.model_loaded,      data.model_loaded ? 'Loaded' : 'Missing');
    setHealthDot('health-scaler',  data.scaler_loaded,     data.scaler_loaded ? 'Loaded' : 'Missing');
    setHealthDot('health-redis',   data.redis_connected,   data.redis_connected ? 'Connected' : 'Down');
    setHealthDot('health-postgres',data.postgres_connected, data.postgres_connected ? 'Connected' : 'Down');
    setHealthDot('health-api',     data.status === 'healthy', data.status);

    // DB count
    const dbCount = data.postgres_transaction_count || 0;
    document.getElementById('db-count').textContent =
        `${dbCount.toLocaleString()} total rows in PostgreSQL`;
}


function setHealthDot(itemId, isHealthy, statusText) {
    const item = document.getElementById(itemId);
    if (!item) return;
    const dot = item.querySelector('.health-dot');
    const label = item.querySelector('.health-status');
    dot.className = `health-dot ${isHealthy ? 'health-dot--healthy' : 'health-dot--unhealthy'}`;
    label.textContent = statusText;
}


// ─── Recent Transactions Polling ───────────────────────────
async function pollRecent() {
    try {
        const res = await fetch(`${API}/recent?limit=50`, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderFeed(data.transactions || []);
        renderLatencyAndCache(data.transactions || []);
    } catch (err) {
        // Silently fail — health poll will show disconnect
    }
}


function renderFeed(transactions) {
    const tbody = document.getElementById('feed-body');
    const countEl = document.getElementById('feed-count');

    if (transactions.length === 0) return;

    countEl.textContent = `${transactions.length} transactions`;

    // Build rows
    const rows = transactions.map(txn => {
        const isNew = !lastSeenIds.has(txn.transaction_id);
        const isFraud = txn.is_fraud;
        const rowClass = [
            isFraud ? 'feed-row--fraud' : '',
            isNew ? 'feed-row--new' : ''
        ].filter(Boolean).join(' ');

        const badge = isFraud
            ? '<span class="badge badge--fraud">FRAUD</span>'
            : '<span class="badge badge--legit">LEGIT</span>';

        const confidence = (txn.confidence * 100).toFixed(2) + '%';
        const latency = txn.latency_ms.toFixed(2) + 'ms';
        const amount = '$' + Number(txn.amount).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        const shortId = txn.transaction_id.substring(0, 12) + '…';
        const time = formatTime(txn.created_at);
        const source = txn.source || 'api';

        return `<tr class="${rowClass}">
            <td>${badge}</td>
            <td title="${txn.transaction_id}">${shortId}</td>
            <td>${amount}</td>
            <td>${confidence}</td>
            <td>${latency}</td>
            <td>${source}</td>
            <td>${time}</td>
        </tr>`;
    });

    tbody.innerHTML = rows.join('');

    // Update the seen-IDs set
    lastSeenIds = new Set(transactions.map(t => t.transaction_id));
}


function renderLatencyAndCache(transactions) {
    if (transactions.length === 0) return;

    // Average latency from visible transactions
    const totalLatency = transactions.reduce((sum, t) => sum + t.latency_ms, 0);
    const avgLatency = totalLatency / transactions.length;
    document.getElementById('avg-latency').textContent = avgLatency.toFixed(2) + 'ms';

    // Cache hit rate (from /health metrics — approximate from data)
    // The health endpoint has no cache_hit metric; we show "Redis Active" instead
    const cacheEl = document.getElementById('cache-rate');
    cacheEl.textContent = 'Active';
}


// ─── Benchmark ─────────────────────────────────────────────
async function runBenchmark() {
    const btn = document.getElementById('btn-benchmark');
    const content = document.getElementById('benchmark-content');
    const results = document.getElementById('benchmark-results');

    btn.disabled = true;
    btn.innerHTML = `
        <svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M21 12a9 9 0 11-6.22-8.56"/></svg>
        Running 1,000 ops…
    `;

    try {
        const res = await fetch(`${API}/benchmark`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Show results
        content.style.display = 'none';
        results.style.display = 'flex';

        document.getElementById('redis-ms').textContent = `${data.redis_avg_ms}ms avg (P95: ${data.redis_p95_ms}ms)`;
        document.getElementById('postgres-ms').textContent = `${data.postgres_avg_ms}ms avg (P95: ${data.postgres_p95_ms}ms)`;

        // Animate bars — scale relative to the slower one
        const maxMs = Math.max(data.redis_avg_ms, data.postgres_avg_ms);
        requestAnimationFrame(() => {
            document.getElementById('redis-bar').style.width =
                `${(data.redis_avg_ms / maxMs * 100).toFixed(1)}%`;
            document.getElementById('pg-bar').style.width =
                `${(data.postgres_avg_ms / maxMs * 100).toFixed(1)}%`;
        });

        document.getElementById('benchmark-verdict').textContent =
            `${data.conclusion}`;

    } catch (err) {
        content.style.display = 'flex';
        results.style.display = 'none';
        alert('Benchmark failed: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Run Again
        `;
    }
}


// ─── Helpers ───────────────────────────────────────────────
function setConnectionStatus(state, label) {
    const el = document.getElementById('connection-status');
    const dot = el.querySelector('.status-dot');
    const text = el.querySelector('.status-label');
    dot.className = `status-dot status-dot--${state}`;
    text.textContent = label;
}


function animateNumber(elementId, target) {
    const el = document.getElementById(elementId);
    const current = parseInt(el.textContent.replace(/[^0-9]/g, '')) || 0;
    if (current === target) {
        el.textContent = target.toLocaleString();
        return;
    }
    // Simple animation — jump in 8 steps
    const steps = 8;
    const diff = target - current;
    const increment = Math.ceil(diff / steps);
    let i = 0;
    const timer = setInterval(() => {
        i++;
        const val = i >= steps ? target : current + increment * i;
        el.textContent = val.toLocaleString();
        if (i >= steps) clearInterval(timer);
    }, 30);
}


function formatTime(isoString) {
    if (!isoString) return '—';
    try {
        const d = new Date(isoString);
        return d.toLocaleTimeString('en-US', {
            hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    } catch {
        return isoString;
    }
}
