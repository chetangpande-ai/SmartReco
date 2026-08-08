"""Recapture the README screenshots against a running dev server.

Committed rather than done by hand because the screenshots are the first thing anyone
sees and they rot silently — the previous set showed a nav and a dashboard that no
longer exist, and nothing failed to tell us. Re-running this is the fix.

The script drives a real login and fills in the demo career profile first, so the
captures show the app with actual state in it rather than a set of empty-state
placeholders.

    make run                                  # in one terminal
    python scripts/screenshots.py             # in another

Needs playwright with a chromium build (`pip install playwright && playwright install
chromium`). Deliberately not a project dependency: it is a one-off authoring tool, not
something the app or the test suite needs.
"""

import sys
from pathlib import Path

BASE = "http://localhost:8000"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"

USER = ("learner@smartreco.dev", "learner12345")
ADMIN = ("admin@smartreco.dev", "admin12345")

# The QA -> AI Engineer scenario the whole platform is built to answer. Filled in before
# the captures so the career shots show a real plan instead of a blank form.
DEMO_PROFILE = {
    "current_role": "QA Engineer",
    "years_experience": "10",
    "skills": "Java, Selenium, API testing",
    "technologies": "Jenkins, Postman, Git",
    "target_role": "ai-engineer",
    "interests": "agentic AI, evaluation",
    "weekly_hours": "8",
}

# (filename, path, full_page). Full-page for anything whose argument is its length —
# a roadmap cut off at the fold is not a roadmap.
SHOTS = [
    ("home", "/", False),
    ("career-advisor", "/career", True),
    ("career-roadmap", "/careers/qa-to-ai", True),
    ("marketplace", "/courses?category=ai-ml", False),
    ("explore", "/explore", False),
    ("course-detail", "/products/ai-agents-with-langgraph", True),
    ("dashboard", "/me", True),
    ("admin", "/admin", False),
]
ADMIN_ONLY = {"admin"}


def sign_in(page, email: str, password: str) -> None:
    page.goto(f"{BASE}/login", wait_until="networkidle")
    page.fill('input[type=email]', email)
    page.fill('input[type=password]', password)
    page.click('button[type=submit]')
    page.wait_for_load_state("networkidle")


def fill_career_profile(page) -> None:
    page.goto(f"{BASE}/career", wait_until="networkidle")
    for field, value in DEMO_PROFILE.items():
        selector = f'[name="{field}"]'
        if page.locator(selector).count() == 0:
            continue
        if field == "target_role":
            page.select_option(selector, value)
        else:
            page.fill(selector, value)
    page.click('#advisor button[type=submit]')
    page.wait_for_load_state("networkidle")


def capture(page, name: str, path: str, full_page: bool) -> None:
    page.goto(f"{BASE}{path}", wait_until="networkidle")
    # Let the shelf arrows settle and any card transition finish, or the capture catches
    # a hover state mid-animation.
    page.wait_for_timeout(600)
    target = OUT / f"{name}.png"
    page.screenshot(path=str(target), full_page=full_page)
    print(f"  {target.name:22} {target.stat().st_size // 1024:>5} KB  {path}")


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()

        # A context per role rather than signing out in between: `/login` redirects away
        # when a session cookie is already set, so reusing one context leaves the second
        # sign-in staring at a page with no form on it.
        def session(email: str, password: str):
            context = browser.new_context(
                # 2x so the images stay sharp on a retina screen and in a GitHub README.
                viewport={"width": 1440, "height": 900},
                device_scale_factor=2,
            )
            page = context.new_page()
            sign_in(page, email, password)
            return context, page

        probe = browser.new_page()
        try:
            probe.goto(BASE, timeout=8000)
        except Exception:
            print(f"nothing serving on {BASE} — start it with `make run` first", file=sys.stderr)
            browser.close()
            return 1
        probe.close()

        print("signing in as the learner and building the demo career plan…")
        context, page = session(*USER)
        fill_career_profile(page)

        print("capturing:")
        for name, path, full_page in SHOTS:
            if name not in ADMIN_ONLY:
                capture(page, name, path, full_page)
        context.close()

        context, page = session(*ADMIN)
        for name, path, full_page in SHOTS:
            if name in ADMIN_ONLY:
                capture(page, name, path, full_page)
        context.close()

        browser.close()
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
