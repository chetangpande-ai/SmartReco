/* Arrow buttons for the home-page course rows.
 *
 * The rows are plain overflow-x scrollers, so swipe, trackpad and keyboard already work
 * with this file absent — all it adds is the affordance a mouse user expects, and the
 * disabled state that hides an arrow pointing at nothing. */
(function () {
  "use strict";

  document.querySelectorAll("[data-shelf]").forEach(function (shelf) {
    var track = shelf.querySelector("[data-shelf-track]");
    var prev = shelf.querySelector(".shelf-prev");
    var next = shelf.querySelector(".shelf-next");
    if (!track || !prev || !next) return;

    function update() {
      // 1px of slack: fractional scroll positions mean scrollLeft rarely lands exactly
      // on the maximum, which would leave the "next" arrow enabled at the end forever.
      var max = track.scrollWidth - track.clientWidth;
      prev.disabled = track.scrollLeft <= 1;
      next.disabled = track.scrollLeft >= max - 1;
    }

    function page(direction) {
      track.scrollBy({ left: direction * track.clientWidth * 0.85, behavior: "smooth" });
    }

    prev.addEventListener("click", function () { page(-1); });
    next.addEventListener("click", function () { page(1); });
    track.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  });
})();
