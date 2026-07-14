"""
Unit tests for agent functions.

Tests cover:
1. Resume Analyst validation logic
2. Scorer formatting helpers
3. Verifier blind profile creation
4. Decider ranking logic
5. Scheduler prompt formatting
"""

import pytest
from pydantic import ValidationError

from agents.decider import generate_shortlist
from agents.resume_analyst import _validate_profile
from agents.verifier import _create_blind_profile
from models.state import CandidateProfile, JDInput, RecruitmentState, Scorecard, ShortlistEntry, VerifiedScore


class TestResumeAnalystValidation:
    """Tests for the Resume Analyst's validation helper."""

    def test_valid_profile_passes(self):
        """A profile with candidate_id and name should pass validation."""
        profile = CandidateProfile(candidate_id="id123", name="John Doe")
        assert _validate_profile(profile) is True

    def test_empty_candidate_id_fails(self):
        """An empty candidate_id should fail validation."""
        profile = CandidateProfile(candidate_id="", name="John Doe")
        assert _validate_profile(profile) is False

    def test_whitespace_only_name_fails(self):
        """A name with only whitespace should fail validation."""
        profile = CandidateProfile(candidate_id="id123", name="   ")
        assert _validate_profile(profile) is False

    def test_none_candidate_id_fails(self):
        """A candidate_id of None should fail validation at Pydantic level."""
        with pytest.raises(ValidationError):
            CandidateProfile(candidate_id=None, name="John", skills=[])  # type: ignore


class TestVerifierBlindProfile:
    """Tests for the Verifier's blind profile creation."""

    def test_blind_profile_removes_name(self):
        """The blind profile should not contain the candidate's name."""
        profile = CandidateProfile(
            candidate_id="id123",
            name="John Doe",
            skills=["Python"],
            experience=[{"role": "Engineer", "company": "TechCorp", "years": 3, "description": "Worked on stuff"}],
        )
        blind = _create_blind_profile(profile)
        assert "John Doe" not in blind
        assert "id123" not in blind

    def test_blind_profile_retains_skills(self):
        """Skills should still be present in the blind profile."""
        profile = CandidateProfile(
            candidate_id="id123",
            name="Jane",
            skills=["Python", "FastAPI"],
        )
        blind = _create_blind_profile(profile)
        assert "Python" in blind
        assert "FastAPI" in blind

    def test_blind_profile_removes_company(self):
        """Company names should be removed to reduce identifiability."""
        profile = CandidateProfile(
            candidate_id="id123",
            name="John",
            experience=[{"role": "Engineer", "company": "Acme Corp", "years": 3, "description": "Built APIs"}],
        )
        blind = _create_blind_profile(profile)
        assert "Acme Corp" not in blind


class TestDeciderShortlist:
    """Tests for the Decider's shortlist generation."""

    def test_high_confidence_candidates(self):
        """High-confidence candidates should use original scores directly."""
        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=[],
            parsed_profiles=[
                CandidateProfile(candidate_id="id1", name="Alice"),
                CandidateProfile(candidate_id="id2", name="Bob"),
            ],
            scorecards=[
                Scorecard(candidate_id="id1", total_score=90.0, is_borderline=False),
                Scorecard(candidate_id="id2", total_score=60.0, is_borderline=False),
            ],
            verified_scores=[],
            revision_count=0,
            shortlist=[],
            step_count=0,
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        result = generate_shortlist(state)
        assert "shortlist" in result
        shortlist = result["shortlist"]
        assert len(shortlist) == 2
        # Alice should be ranked #1 (higher score)
        assert shortlist[0].candidate_id == "id1"
        assert shortlist[0].final_score == 90.0

    def test_verified_scores_override(self):
        """Verified scores should override original scores when difference > 10."""
        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=[],
            parsed_profiles=[
                CandidateProfile(candidate_id="id1", name="Alice"),
            ],
            scorecards=[
                Scorecard(candidate_id="id1", total_score=90.0, is_borderline=True),
            ],
            verified_scores=[
                VerifiedScore(
                    candidate_id="id1",
                    original_score=90.0,
                    blind_score=50.0,
                    score_difference=40.0,
                    is_fair=False,
                    fairness_notes="Discrepancy found",
                    injection_affected=False,
                ),
            ],
            revision_count=0,
            shortlist=[],
            step_count=0,
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        result = generate_shortlist(state)
        shortlist = result["shortlist"]
        assert len(shortlist) == 1
        # Should use blind_score (50) since difference was > 10
        assert shortlist[0].final_score == 50.0

    def test_averaged_scores(self):
        """When verification difference is small, use average of original + blind."""
        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=[],
            parsed_profiles=[
                CandidateProfile(candidate_id="id1", name="Alice"),
            ],
            scorecards=[
                Scorecard(candidate_id="id1", total_score=70.0, is_borderline=True),
            ],
            verified_scores=[
                VerifiedScore(
                    candidate_id="id1",
                    original_score=70.0,
                    blind_score=74.0,
                    score_difference=4.0,
                    is_fair=True,
                    fairness_notes="Close scores",
                    injection_affected=False,
                ),
            ],
            revision_count=0,
            shortlist=[],
            step_count=0,
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        result = generate_shortlist(state)
        shortlist = result["shortlist"]
        assert len(shortlist) == 1
        # Average of 70 and 74 is 72
        assert shortlist[0].final_score == 72.0


class TestRouterLogic:
    """Tests for the conditional routing logic."""

    def test_route_to_verifier_if_borderline(self):
        """If there are borderline scorecards, route to Verifier."""
        from graph.workflow import route_after_scorer

        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=[],
            parsed_profiles=[],
            scorecards=[
                Scorecard(candidate_id="id1", total_score=65.0, is_borderline=True),
                Scorecard(candidate_id="id2", total_score=85.0, is_borderline=False),
            ],
            verified_scores=[],
            revision_count=0,
            shortlist=[],
            step_count=1,
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        route = route_after_scorer(state)
        assert route == "verifier"

    def test_route_to_decider_if_no_borderline(self):
        """If no borderline candidates, route directly to Decider."""
        from graph.workflow import route_after_scorer

        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=[],
            parsed_profiles=[],
            scorecards=[
                Scorecard(candidate_id="id1", total_score=90.0, is_borderline=False),
                Scorecard(candidate_id="id2", total_score=85.0, is_borderline=False),
            ],
            verified_scores=[],
            revision_count=0,
            shortlist=[],
            step_count=1,
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        route = route_after_scorer(state)
        assert route == "decider"

    def test_route_to_end_if_no_scorecards(self):
        """If no scorecards, end the workflow."""
        from graph.workflow import route_after_scorer

        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=[],
            parsed_profiles=[],
            scorecards=[],
            verified_scores=[],
            revision_count=0,
            shortlist=[],
            step_count=1,
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        route = route_after_scorer(state)
        assert route == "end"

    def test_route_to_end_if_step_budget_exceeded(self):
        """If step budget exceeded, end the workflow."""
        from graph.workflow import route_after_scorer

        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=[],
            parsed_profiles=[],
            scorecards=[Scorecard(candidate_id="id1", total_score=65.0, is_borderline=True)],
            verified_scores=[],
            revision_count=0,
            shortlist=[],
            step_count=100,  # Exceeds default max_step_budget of 50
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        route = route_after_scorer(state)
        assert route == "end"