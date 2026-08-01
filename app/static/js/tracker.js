/*
 * SmartReco behavioural tracker. No dependencies, nothing on the critical path.
 *
 * Design rules, in priority order:
 *   1. Never block rendering or interaction. Queue assembly runs in idle time, the
 *      network call is fire-and-forget, and no handler does layout-reading work.
 *   2. Never lose the end of a session. Page unload is exactly when the most valuable
 *      signal (dwell time) is known, and exactly when fetch() gets cancelled — so the
 *      final flush goes through sendBeacon, which the browser completes after the page
 *      is gone.
 *   3. Never send the same event twice. Every event carries a client-generated idem
 *      key; the server deduplicates, so retrying a failed batch is free.
 *   4. Degrade quietly. Tracking is not worth breaking a page over.
 */
(function () {
  "use strict";

  var ENDPOINT = "/api/events/batch";
  var MAX_BATCH = 20;
  var FLUSH_MS = 5000;
  var SCROLL_THROTTLE_MS = 500;
  var SEARCH_DEBOUNCE_MS = 400;
  var IMPRESSION_MS = 1000;
  var STORE_KEY = "sr_pending";
  var MAX_STORED = 200;

  var queue = [];
  var timer = null;
  var sessionId = getSessionId();

  function uid() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "");
    return (Date.now().toString(36) + Math.random().toString(36).slice(2, 12));
  }

  function getSessionId() {
    try {
      var id = sessionStorage.getItem("sr_session");
      if (!id) {
        id = uid();
        sessionStorage.setItem("sr_session", id);
      }
      return id;
    } catch (e) {
      return uid(); // private mode: a per-page session is still better than none
    }
  }

  // Idle time is a preference, not a promise. The timeout is mandatory: without it a
  // backgrounded tab can defer the callback indefinitely and the batch is never sent.
  // Never used on the unload path, where only sendBeacon is guaranteed to complete.
  var idle = window.requestIdleCallback
    ? function (fn) { return requestIdleCallback(fn, { timeout: 500 }); }
    : function (fn) { return setTimeout(fn, 1); };

  function track(type, payload) {
    var event = payload || {};
    event.type = type;
    event.ts = Date.now();
    event.session_id = sessionId;
    event.idem = uid();
    if (!event.path) event.path = location.pathname;

    queue.push(event);
    if (queue.length >= MAX_BATCH) {
      flush();
    } else if (!timer) {
      timer = setTimeout(flush, FLUSH_MS);
    }
  }

  function flush(useBeacon) {
    if (timer) { clearTimeout(timer); timer = null; }
    if (!queue.length) return;

    var batch = queue.splice(0, queue.length);
    var body = JSON.stringify({ events: batch });

    if (useBeacon && navigator.sendBeacon) {
      // Blob type matters: without it the browser sends text/plain and the server
      // rejects the body.
      var ok = navigator.sendBeacon(ENDPOINT, new Blob([body], { type: "application/json" }));
      if (!ok) stash(batch);
      return;
    }

    idle(function () {
      fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        keepalive: true,
        credentials: "same-origin",
      }).catch(function () {
        stash(batch); // retried on the next page load; idem keys make that safe
      });
    });
  }

  function stash(batch) {
    try {
      var kept = JSON.parse(localStorage.getItem(STORE_KEY) || "[]").concat(batch);
      localStorage.setItem(STORE_KEY, JSON.stringify(kept.slice(-MAX_STORED)));
    } catch (e) { /* quota or private mode: dropping telemetry is the right trade */ }
  }

  function drainStash() {
    try {
      var kept = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
      if (!kept.length) return;
      localStorage.removeItem(STORE_KEY);
      queue = kept.concat(queue);
      flush();
    } catch (e) { /* ignore */ }
  }

  function throttle(fn, ms) {
    var last = 0, pending = null;
    return function () {
      var now = Date.now();
      if (now - last >= ms) { last = now; fn(); return; }
      if (!pending) {
        pending = setTimeout(function () { pending = null; last = Date.now(); fn(); },
                             ms - (now - last));
      }
    };
  }

  function debounce(fn, ms) {
    var t = null;
    return function () {
      var args = arguments;
      clearTimeout(t);
      t = setTimeout(function () { fn.apply(null, args); }, ms);
    };
  }

  // ---- dwell -------------------------------------------------------------
  // Accumulated *visible* time, not wall clock: a tab left open in the background for
  // an hour is not an hour of interest, and treating it as such would swamp the
  // profile with a signal the user never actually gave.
  var visibleSince = document.visibilityState === "visible" ? Date.now() : 0;
  var dwellMs = 0;
  var dwellSent = false;

  function accumulate() {
    if (visibleSince) { dwellMs += Date.now() - visibleSince; visibleSince = 0; }
  }

  // Called when the page is going away. It must ALWAYS flush, whether or not a dwell
  // event is worth emitting — otherwise a navigation inside the 5s batch window drops
  // every queued event on the floor, which is most of a browsing session.
  function onPageHidden() {
    accumulate();
    if (!dwellSent && dwellMs >= 1000) {
      dwellSent = true;
      track("dwell", {
        dwell_ms: Math.min(dwellMs, 3600000),
        product_slug: document.body.dataset.productSlug || null,
        meta: { max_scroll: maxDepth },
      });
    }
    flush(true);
  }

  // ---- scroll depth ------------------------------------------------------
  var maxDepth = 0;
  var depthsSent = {};
  var onScroll = throttle(function () {
    var doc = document.documentElement;
    var scrollable = doc.scrollHeight - window.innerHeight;
    if (scrollable <= 0) return;
    var pct = Math.round(((window.scrollY || doc.scrollTop) / scrollable) * 100);
    if (pct <= maxDepth) return;
    maxDepth = pct;
    [25, 50, 75, 100].forEach(function (mark) {
      if (pct >= mark && !depthsSent[mark]) {
        depthsSent[mark] = true;
        track("scroll_depth", { position: mark });
      }
    });
  }, SCROLL_THROTTLE_MS);

  // ---- wiring ------------------------------------------------------------
  function init() {
    var body = document.body;
    drainStash();

    if (body.dataset.productSlug) {
      track("product_view", {
        product_slug: body.dataset.productSlug,
        meta: {
          category: body.dataset.productCategory || "",
          level: body.dataset.productLevel || "",
        },
      });
    } else {
      track("page_view", { meta: { title: document.title.slice(0, 120) } });
    }

    if (body.dataset.searchQuery) {
      track("search", { query: body.dataset.searchQuery, meta: { results: body.dataset.searchResults || "0" } });
    }

    // Delegated so cards rendered later are covered without rebinding.
    document.addEventListener("click", function (e) {
      var el = e.target.closest("[data-track-click]");
      if (!el) return;
      track(el.dataset.trackClick, {
        product_slug: el.dataset.productSlug || null,
        position: parseInt(el.dataset.position || "0", 10),
        meta: { source: el.dataset.source || "" },
      });
    }, { passive: true });

    var searchInput = document.querySelector("[data-track-search]");
    if (searchInput) {
      searchInput.addEventListener("input", debounce(function (e) {
        var q = e.target.value.trim();
        // Below 3 characters it is keystroke noise, not an intent signal.
        if (q.length >= 3) track("search", { query: q.slice(0, 200), meta: { partial: true } });
      }, SEARCH_DEBOUNCE_MS));
    }

    document.querySelectorAll("[data-track-filter]").forEach(function (el) {
      el.addEventListener("click", function () {
        track("filter", { meta: { facet: el.dataset.trackFilter, value: el.dataset.value || "" } });
      }, { passive: true });
    });

    window.addEventListener("scroll", onScroll, { passive: true });

    // Recommendation impressions: only counted once the card has actually been on
    // screen for a second, so "shown" means shown rather than "existed in the DOM".
    if (window.IntersectionObserver) {
      var seen = new WeakSet();
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          var el = entry.target;
          if (!entry.isIntersecting || seen.has(el)) return;
          el._srTimer = setTimeout(function () {
            if (seen.has(el)) return;
            seen.add(el);
            observer.unobserve(el);
            track("rec_impression", {
              product_slug: el.dataset.productSlug || null,
              position: parseInt(el.dataset.position || "0", 10),
            });
          }, IMPRESSION_MS);
        });
        entries.forEach(function (entry) {
          if (!entry.isIntersecting && entry.target._srTimer) clearTimeout(entry.target._srTimer);
        });
      }, { threshold: 0.5 });
      document.querySelectorAll("[data-track-impression]").forEach(function (el) {
        observer.observe(el);
      });
    }

    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") {
        onPageHidden();
      } else {
        visibleSince = Date.now();
        dwellSent = false; // returning to the tab starts a new dwell segment
      }
    });

    // pagehide fires on bfcache navigation where unload does not.
    window.addEventListener("pagehide", onPageHidden);
  }

  window.SmartReco = { track: track, flush: flush };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
