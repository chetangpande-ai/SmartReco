/* Poll for a recommendation while a background agent run is in flight.
   Backs off and gives up rather than hammering the endpoint forever. */
(function () {
  "use strict";
  if (document.body.dataset.pollRecs !== "1" &&
      !new URLSearchParams(location.search).has("refreshing")) return;

  var attempts = 0;
  var MAX_ATTEMPTS = 12;
  var delay = 1500;

  function poll() {
    if (attempts++ >= MAX_ATTEMPTS) {
      var banner = document.getElementById("rec-pending");
      if (banner) banner.textContent = "Still working — refresh the page in a moment.";
      return;
    }
    fetch("/api/recommendations/current", { credentials: "same-origin" })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        // The agent may already have had a recommendation when the page rendered; only
        // reload when something newer actually arrived.
        var shown = document.getElementById("rec-block");
        if (data.ready && (!shown || shown.dataset.recId !== String(data.id))) {
          location.replace("/me");
          return;
        }
        delay = Math.min(delay * 1.4, 6000);
        setTimeout(poll, delay);
      })
      .catch(function () { setTimeout(poll, delay); });
  }

  setTimeout(poll, delay);
})();
