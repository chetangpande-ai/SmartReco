"""The career layer: taxonomy, skill-gap analysis, roadmaps and enrollment."""

import pytest
from sqlalchemy import select

from app import taxonomy as T
from app.db import session_scope
from app.models import CareerProfile, Product

# The `catalog` fixture is the seeded course set; the service module is imported
# under a different name so the fixture does not shadow it inside a test.
from app.services import advisor, learning
from app.services import catalog as catalog_service


def _profile(user_id: int, **kwargs) -> CareerProfile:
    """A stated career profile, committed. Free-text skills go through the same
    resolver the form uses, so a test types what a person would type."""
    skills = kwargs.pop("skills", "")
    known, unknown = T.resolve_skills(skills)
    with session_scope() as db:
        row = db.get(CareerProfile, user_id) or CareerProfile(user_id=user_id)
        row.skills = known
        row.extra_skills = unknown
        row.current_role = kwargs.pop("current_role", "")
        row.years_experience = kwargs.pop("years_experience", 0)
        row.target_role = kwargs.pop("target_role", "")
        row.interests = kwargs.pop("interests", "")
        row.weekly_hours = kwargs.pop("weekly_hours", 0)
        row.level = kwargs.pop("level", "")
        db.add(row)
        db.flush()
        db.expunge(row)
        return row


class TestTaxonomy:
    def test_loads_the_shipped_catalogue(self):
        tax = T.taxonomy()
        assert len(tax.categories) == 21
        assert len(tax.roles) == 22
        assert len(tax.paths) == 10

    def test_subcategory_ids_are_only_unique_within_a_category(self):
        """`tools` is both a Project Management and a UI/UX subcategory. Keyed by the
        bare id, one silently replaces the other and half the tree points at the wrong
        parent."""
        assert T.subcategory("project-management/tools").category == "project-management"
        assert T.subcategory("ui-ux/tools").category == "ui-ux"
        assert T.subcategory("tools") is None

    @pytest.mark.parametrize(
        ("typed", "expected"),
        [
            ("Python", "python"),
            ("ML", "machine-learning"),
            ("AI Evals", "ai-evaluation"),
            ("agentic ai", "ai-agents"),
            ("ci/cd", "ci-cd"),
            ("API Testing", "api-testing"),
        ],
    )
    def test_resolves_what_people_actually_type(self, typed, expected):
        assert T.resolve_skill(typed) == expected

    def test_unrecognised_input_is_returned_not_swallowed(self):
        known, unknown = T.resolve_skills("Python, Blorptastic Framework")
        assert known == ["python"]
        assert unknown == ["Blorptastic Framework"]

    def test_implications_close_transitively(self):
        """RAG implies LLMs implies Generative AI implies LLM Fundamentals. A one-level
        pass stops three skills short and puts them all back in the gap list."""
        expanded = T.expand_skills(["rag"])
        assert {"llms", "generative-ai", "llm-fundamentals", "embeddings"} <= set(expanded)

    def test_the_source_skills_come_first(self):
        assert T.expand_skills(["selenium"])[0] == "selenium"

    def test_a_transition_path_names_both_ends(self):
        path = T.path("qa-to-ai")
        assert path.from_role == "qa-engineer" and path.to_role == "ai-engineer"

    def test_transitions_are_offered_before_the_from_scratch_route(self):
        assert T.paths_for_role("ai-engineer")[0].slug == "qa-to-ai"


class TestSkillGap:
    def test_a_tester_is_not_told_to_learn_testing(self, catalog, user_factory):
        """The headline failure mode. Someone who lists Selenium and API testing has
        been testing for a living, and a literal set difference says otherwise."""
        uid = user_factory()
        profile = _profile(
            uid, current_role="QA Engineer", years_experience=10,
            skills="Java, Selenium, API testing", target_role="ai-engineer",
        )
        with session_scope() as db:
            analysis = advisor.analyse(db, profile)

        assert "testing" not in analysis.gap_slugs
        assert "automation" not in analysis.gap_slugs

    def test_it_picks_the_bridge_path_over_the_generic_one(self, catalog, user_factory):
        uid = user_factory()
        profile = _profile(
            uid, current_role="QA Engineer", years_experience=10,
            skills="Selenium", target_role="ai-engineer",
        )
        with session_scope() as db:
            assert advisor.analyse(db, profile).path.slug == "qa-to-ai"

    def test_held_skills_are_reported_as_held(self, catalog, user_factory):
        uid = user_factory()
        profile = _profile(uid, skills="Python, SQL", target_role="data-scientist")
        with session_scope() as db:
            analysis = advisor.analyse(db, profile)

        assert {"python", "sql"} <= set(analysis.have)
        assert not {"python", "sql"} & set(analysis.gap_slugs)
        assert 0 < analysis.readiness < 1

    def test_unrecognised_skills_are_surfaced_not_dropped(self, catalog, user_factory):
        uid = user_factory()
        profile = _profile(uid, skills="Python, Blorptastic", target_role="data-scientist")
        with session_scope() as db:
            assert "Blorptastic" in advisor.analyse(db, profile).unknown


class TestCourseSequencing:
    def test_a_course_never_precedes_its_own_prerequisites(self, catalog, user_factory):
        """The property that makes the plan a plan. Walking it in order, each course's
        prerequisites are satisfied by the learner or by a course already scheduled."""
        uid = user_factory()
        profile = _profile(uid, skills="", target_role="ai-engineer")
        with session_scope() as db:
            analysis = advisor.analyse(db, profile)
            held = set(analysis.have)
            for course in analysis.courses:
                unmet = [r for r in course.requires if r not in held]
                assert not unmet, f"{course.title} needs {unmet} before it is reachable"
                held.update(T.expand_skills(course.teaches))

    def test_one_course_covering_four_gaps_is_one_step(self, catalog, user_factory):
        uid = user_factory()
        profile = _profile(uid, skills="", target_role="ai-engineer")
        with session_scope() as db:
            analysis = advisor.analyse(db, profile)
            stage = next(s for s in analysis.stages if s["key"] == "courses")

        ids = [c.id for c in analysis.courses]
        assert len(ids) == len(set(ids))
        assert len(stage["entries"]) == len(ids)

    def test_interview_prep_is_never_how_you_learn_a_skill(self, catalog, user_factory):
        """An interview course lists Python because its problems are written in it.
        Sending a Java developer there to acquire Python is the wrong answer."""
        uid = user_factory()
        profile = _profile(
            uid, current_role="QA Engineer", years_experience=10,
            skills="Java, Selenium", target_role="ai-engineer",
        )
        with session_scope() as db:
            titles = {c.format for c in advisor.analyse(db, profile).courses}
        assert not titles & advisor.NON_TEACHING_FORMATS

    def test_seniority_does_not_skip_the_entry_point_for_a_new_subject(
        self, catalog, user_factory
    ):
        """Ten years of testing says nothing about how to meet machine learning for the
        first time. The plan should still start at the beginning there."""
        uid = user_factory()
        profile = _profile(
            uid, current_role="QA Engineer", years_experience=15,
            skills="Selenium, API testing", target_role="data-scientist",
        )
        with session_scope() as db:
            analysis = advisor.analyse(db, profile)
            ml = next((g for g in analysis.gaps if g.skill == "machine-learning"), None)

        assert ml is not None and ml.course is not None
        assert ml.course.tier == "beginner"

    def test_completed_courses_close_their_gaps(self, catalog, user_factory):
        uid = user_factory()
        profile = _profile(uid, skills="", target_role="data-scientist")
        with session_scope() as db:
            before = set(advisor.analyse(db, profile).gap_slugs)
            product = db.get(Product, catalog["SQL for Data Analysis"])
            learning.set_progress(db, uid, product, 100)
        with session_scope() as db:
            after = set(advisor.analyse(db, db.get(CareerProfile, uid)).gap_slugs)

        assert "sql" in before and "sql" not in after


class TestPlans:
    def test_a_plan_is_generated_and_becomes_current(self, catalog, user_factory):
        uid = user_factory()
        _profile(uid, current_role="QA Engineer", skills="Selenium", target_role="ai-engineer")
        with session_scope() as db:
            plan = advisor.generate(db, uid)
            assert plan.headline and plan.narrative
            assert plan.strategy == "deterministic", "LLM is off in tests"
            assert advisor.current_plan(db, uid).id == plan.id

    def test_regenerating_supersedes_the_previous_plan(self, catalog, user_factory):
        uid = user_factory()
        _profile(uid, skills="Python", target_role="data-scientist")
        with session_scope() as db:
            first = advisor.generate(db, uid)
            second = advisor.generate(db, uid)
            assert first.id != second.id
            assert advisor.current_plan(db, uid).id == second.id

    def test_no_target_role_means_no_plan_rather_than_a_crash(self, catalog, user_factory):
        uid = user_factory()
        _profile(uid, skills="Python", target_role="")
        with session_scope() as db:
            assert advisor.generate(db, uid) is None

    def test_every_shipped_path_previews_without_a_learner(self, catalog):
        """The signed-out career pages. A path that renders an empty roadmap is a page
        that argues against signing up."""
        with session_scope() as db:
            for slug in T.paths():
                analysis = advisor.preview(db, T.path(slug))
                assert analysis.title
                assert analysis.gaps, f"{slug} has no steps"

    def test_zero_overlap_still_reads_like_a_person_wrote_it(self, catalog, user_factory):
        """A QA engineer has none of AI Engineer's listed skills. "0% ready" is true and
        reads as a dismissal, so the copy takes a different shape."""
        uid = user_factory()
        _profile(
            uid, current_role="QA Engineer", years_experience=10,
            skills="Java, Selenium, API testing", target_role="ai-engineer",
        )
        with session_scope() as db:
            plan = advisor.generate(db, uid)
        assert "0%" not in plan.narrative
        assert "not a restart" in plan.narrative


class TestCourseFit:
    def test_it_says_no_when_a_prerequisite_is_missing(self, catalog, user_factory):
        uid = user_factory()
        _profile(uid, skills="", target_role="ai-engineer")
        with session_scope() as db:
            product = db.get(Product, catalog["Deep Learning Specialization"])
            answer = advisor.course_fit(db, product, uid)

        assert answer["verdict"] == "not yet"
        assert "python" in answer["missing"]

    def test_it_says_yes_when_they_are_ready(self, catalog, user_factory):
        uid = user_factory()
        _profile(uid, skills="Python", target_role="ai-engineer")
        with session_scope() as db:
            product = db.get(Product, catalog["Deep Learning Specialization"])
            answer = advisor.course_fit(db, product, uid)

        assert answer["verdict"] == "good fit" and not answer["missing"]

    def test_it_answers_for_a_signed_out_visitor(self, catalog):
        with session_scope() as db:
            product = db.get(Product, catalog["SQL for Data Analysis"])
            assert advisor.course_fit(db, product, None)["answer"]


class TestEnrollment:
    def test_starting_twice_is_one_enrollment(self, catalog, user_factory):
        uid = user_factory()
        with session_scope() as db:
            product = db.get(Product, catalog["SQL for Data Analysis"])
            first = learning.start(db, uid, product)
            second = learning.start(db, uid, product)
            assert first.id == second.id

    def test_completion_issues_exactly_one_certificate(self, catalog, user_factory):
        uid = user_factory()
        with session_scope() as db:
            product = db.get(Product, catalog["SQL for Data Analysis"])
            row = learning.set_progress(db, uid, product, 100)
            code = row.certificate_code
            assert row.status == "completed" and code

            again = learning.set_progress(db, uid, product, 100)
            assert again.certificate_code == code

    def test_reopening_a_finished_course_keeps_the_certificate(self, catalog, user_factory):
        """It was earned. Taking it back because someone rewatched a lesson is absurd."""
        uid = user_factory()
        with session_scope() as db:
            product = db.get(Product, catalog["SQL for Data Analysis"])
            code = learning.set_progress(db, uid, product, 100).certificate_code
            reopened = learning.set_progress(db, uid, product, 60)

        assert reopened.status == "active" and reopened.certificate_code == code

    def test_saving_then_starting_is_one_row(self, catalog, user_factory):
        uid = user_factory()
        with session_scope() as db:
            product = db.get(Product, catalog["SQL for Data Analysis"])
            saved = learning.save(db, uid, product)
            assert saved.status == "saved"
            started = learning.start(db, uid, product)
            assert started.id == saved.id and started.status == "active"

    def test_the_dashboard_buckets_by_status(self, catalog, user_factory):
        uid = user_factory()
        with session_scope() as db:
            learning.set_progress(db, uid, db.get(Product, catalog["SQL for Data Analysis"]), 100)
            learning.start(db, uid, db.get(Product, catalog["Total TypeScript"]))
            learning.save(db, uid, db.get(Product, catalog["Complete Intro to React"]))
            board = learning.dashboard(db, uid)

        assert len(board["completed"]) == 1
        assert len(board["in_progress"]) == 1
        assert len(board["saved"]) == 1
        assert "sql" in board["skills_earned"]


class TestMarketplaceFilters:
    def test_filters_compose(self, catalog):
        with session_scope() as db:
            f = catalog_filters(category="ai-ml", level="advanced")
            rows, total = catalog_service.search(db, f)
        assert total == len(rows)
        assert all(r.category == "ai-ml" and r.tier == "advanced" for r in rows)

    def test_a_role_filter_matches_any_of_its_skills(self, catalog):
        with session_scope() as db:
            rows, _ = catalog_service.search(db, catalog_filters(role="data-scientist"))
        assert rows
        wanted = set(T.role("data-scientist").skills)
        assert all(set(r.teaches) & wanted for r in rows)

    def test_free_and_paid_are_complements(self, catalog):
        with session_scope() as db:
            free, n_free = catalog_service.search(db, catalog_filters(price="free"), limit=100)
            paid, n_paid = catalog_service.search(db, catalog_filters(price="paid"), limit=100)
            _, n_all = catalog_service.search(db, catalog_filters(), limit=100)
        assert n_free + n_paid == n_all
        assert all(p.price_cents == 0 for p in free)

    def test_removing_a_facet_leaves_the_others_alone(self):
        """Each chip is a toggle for exactly itself; the link it points at is the whole
        current filter set minus one."""
        f = catalog_filters(category="ai-ml", level="advanced", price="free")
        cleared = f.query(level="")
        assert "category=ai-ml" in cleared and "price=free" in cleared
        assert "level=" not in cleared

    def test_active_filters_read_as_names_not_slugs(self):
        f = catalog_filters(category="ai-ml", role="ai-engineer")
        labels = dict(f.active)
        assert labels["category"] == "AI & Machine Learning"
        assert labels["role"] == "AI Engineer"


def catalog_filters(**kwargs) -> catalog_service.CourseFilters:
    return catalog_service.CourseFilters(**kwargs)


class TestCourseSkillEdges:
    def test_skills_are_stored_canonically_whatever_was_typed(self, catalog):
        from app.schemas import ProductIn

        data = ProductIn(title="Test", category="ai-ml", teaches=["AI Evals", "ML"])
        assert data.teaches == ["ai-evaluation", "machine-learning"]

    def test_unknown_skills_are_dropped_rather_than_stored(self, catalog):
        from app.schemas import ProductIn

        data = ProductIn(title="Test", category="ai-ml", teaches=["python", "Blorptastic"])
        assert data.teaches == ["python"]

    def test_editing_a_course_replaces_its_edges(self, catalog, fresh_vector_store):
        from app.models import CourseSkill
        from app.schemas import ProductIn
        from app.services.catalog import update_product

        with session_scope() as db:
            product = db.get(Product, catalog["SQL for Data Analysis"])
            original = ProductIn(
                title=product.title, description=product.description,
                category=product.category, tier=product.tier, tags=product.tags,
                price_cents=product.price_cents, brand=product.brand, spec=product.spec,
                rating=product.rating, teaches=product.teaches, requires=product.requires,
            )
            update_product(db, product, original.model_copy(update={"teaches": ["python"]}))

        try:
            with session_scope() as db:
                edges = db.scalars(
                    select(CourseSkill.skill).where(
                        CourseSkill.product_id == catalog["SQL for Data Analysis"],
                        CourseSkill.kind == "teaches",
                    )
                ).all()
            assert list(edges) == ["python"]
        finally:
            with session_scope() as db:
                update_product(db, db.get(Product, catalog["SQL for Data Analysis"]), original)


class TestCareerPages:
    """The pages themselves. Cheap, and they catch the whole class of bug where a
    template references a context key the route stopped passing."""

    @pytest.fixture
    def client(self, catalog):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as test_client:
            yield test_client

    @pytest.mark.parametrize(
        "url",
        [
            "/",
            "/explore",
            "/courses",
            "/courses?category=ai-ml&level=advanced&price=free&certificate=1",
            "/courses?role=data-scientist&sort=rating",
            "/explore/categories/ai-ml",
            "/explore/skills/rag",
            "/careers",
            "/careers/qa-to-ai",
            "/careers/genai-engineer",
            "/career",
            "/career?demo=1",
        ],
    )
    def test_renders(self, client, url):
        assert client.get(url).status_code == 200

    def test_unknown_path_and_skill_are_404(self, client):
        assert client.get("/careers/not-a-path").status_code == 404
        assert client.get("/explore/skills/not-a-skill").status_code == 404

    def test_a_stale_filter_degrades_instead_of_erroring(self, client):
        """A bookmarked URL naming a facet that no longer exists should render the
        unfiltered page, not a 400."""
        response = client.get("/courses?category=removed-last-week&level=wizard")
        assert response.status_code == 200

    def test_the_demo_scenario_is_prefilled(self, client):
        assert 'value="QA Engineer"' in client.get("/career?demo=1").text

    def test_signed_out_roadmaps_still_show_real_courses(self, client):
        assert "road-card" in client.get("/careers/data-scientist").text

    def test_building_a_plan_requires_signing_in(self, client):
        from app.security import CSRF_COOKIE, CSRF_FIELD

        client.get("/")
        response = client.post(
            "/career/advisor",
            data={"target_role": "ai-engineer", CSRF_FIELD: client.cookies.get(CSRF_COOKIE)},
            follow_redirects=False,
        )
        assert response.status_code in (303, 401)
        assert response.headers.get("location", "").startswith("/login")


class TestCatalogueIntegrity:
    def test_the_shipped_catalogue_agrees_with_the_taxonomy(self):
        """The same check CI runs. A typo in a skill slug does not raise — it silently
        produces a roadmap step with no course behind it."""
        from scripts.check_catalogue import main

        assert main() == 0
