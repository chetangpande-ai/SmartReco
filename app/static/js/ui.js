/* Small interaction niceties. Deliberately separate from tracker.js: nothing here may
   interfere with tracking, and tracking must keep working if this file fails to load. */
(function () {
  "use strict";

  var stack = null;

  function toast(message) {
    if (!stack) {
      stack = document.createElement("div");
      stack.className = "toast-stack";
      stack.setAttribute("role", "status");
      stack.setAttribute("aria-live", "polite");
      document.body.appendChild(stack);
    }
    var el = document.createElement("div");
    el.className = "toast";
    el.textContent = message;
    stack.appendChild(el);
    setTimeout(function () { el.remove(); }, 2600);
  }

  // Listens on the same delegated pattern as tracker.js but independently, so an
  // exception raised here can never prevent an event from being queued.
  document.addEventListener("click", function (e) {
    var el = e.target.closest("[data-toast]");
    if (el) toast(el.dataset.toast);
  }, { passive: true });

  window.SmartRecoUI = { toast: toast };
})();
