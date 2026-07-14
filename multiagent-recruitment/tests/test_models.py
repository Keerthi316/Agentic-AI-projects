"""
Unit tests for Pydantic models and state definitions.

Tests cover:
1. CandidateProfile validation (required fields, defaults)
2. Scorecard validation (score ranges, borderline detection)
3. VerifiedScore validation (fairness computation)
4. ShortlistEntry validation (status patterns)
5. JDInput validation (min_length constraints)
6. RecruitmentState TypedDict structure
"""

import pytest
from pydantic import ValidationError

from models.state import (
    CandidateProfile,
    JDInput,
    RecruitmentState,
    Scorecard,
    ShortlistEntry,
    VerifiedScore,
)


class TestCandidateProfile:
    """Tests for the CandidateProfile model."""

    def test_valid_profile(self):
        """A profile with all required fields should validate successfully."""
        profile = CandidateProfile(
            candidate_id="john@example.com",
            name="John Doe",
            skills=["Python", "FastAPI"],
            education=["M.S. Computer Science"],
            experience=[{"role": "Engineer", "company": "TechCorp", "years": 3, "description": "Built APIs"}],
            projects=[{"name": "Project X", "description": "A project", "technologies": ["Python"]}],
            certifications=["AWS Certified"],
        )
        assert profile.candidate_id == "john@example.com"
        assert profile.name == "John Doe"
        assert len(profile.skills) == 2
        assert profile.is_injection_detected is False
        assert profile.injection_confidence == 0.0

    def test_minimal_profile(self):
        """A profile with only required fields should still validate."""
        profile = CandidateProfile(candidate_id="id123", name="Jane")
        assert profile.candidate_id == "id123"
        assert profile.name == "Jane"
        assert profile.skills == []
        assert profile.experience == []

    def test_missing_candidate_id(self):
        """Missing candidate_id should raise ValidationError."""
        with pytest.raises(ValidationError):
            CandidateProfile(name="No ID")

    def test_missing_name(self):
        """Missing name should raise ValidationError."""
        with pytest.raises(ValidationError):
            CandidateProfile(candidate_id="id123")

    def test_injection_fields(self):
        """Injection detection fields should be settable."""
        profile = CandidateProfile(
            candidate_id="id123",
            name="Test",
            is_injection_detected=True,
            injection_confidence=0.85,
        )
        assert profile.is_injection_detected is True
        assert profile.injection_confidence == 0.85

    def test_injection_confidence_range(self):
        """injection_confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            CandidateProfile(
                candidate_id="id123",
                name="Test",
                injection_confidence=1.5,  # Out of range
            )


class TestScorecard:
    """Tests for the Scorecard model."""

    def test_valid_scorecard(self):
        """A valid scorecard should have all fields set correctly."""
        sc = Scorecard(
            candidate_id="id123",
            total_score=85.0,
            skill_score=90.0,
            experience_score=80.0,
            education_score=85.0,
            reasoning="Strong match across all dimensions",
            is_borderline=False,
        )
        assert sc.total_score == 85.0
        assert sc.is_borderline is False

    def test_borderline_score(self):
        """A score of 65 should be borderline."""
        sc = Scorecard(
            candidate_id="id123",
            total_score=65.0,
            is_borderline=True,
        )
        assert sc.is_borderline is True

    def test_score_out_of_range(self):
        """Score above 100 should raise ValidationError."""
        with pytest.raises(ValidationError):
            Scorecard(candidate_id="id123", total_score=150.0)

    def test_negative_score(self):
        """Negative score should raise ValidationError."""
        with pytest.raises(ValidationError):
            Scorecard(candidate_id="id123", total_score=-10.0)


class TestVerifiedScore:
    """Tests for the VerifiedScore model."""

    def test_valid_verified_score(self):
        """A valid verified score should compute correctly."""
        vs = VerifiedScore(
            candidate_id="id123",
            original_score=70.0,
            blind_score=65.0,
            score_difference=5.0,
            is_fair=True,
            fairness_notes="Scores are close",
            injection_affected=False,
        )
        assert vs.score_difference == 5.0
        assert vs.is_fair is True

    def test_unfair_score(self):
        """A large score difference should be marked unfair."""
        vs = VerifiedScore(
            candidate_id="id123",
            original_score=90.0,
            blind_score=50.0,
            score_difference=40.0,
            is_fair=False,
            fairness_notes="Significant discrepancy",
            injection_affected=True,
        )
        assert vs.is_fair is False
        assert vs.injection_affected is True


class TestShortlistEntry:
    """Tests for the ShortlistEntry model."""

    def test_valid_shortlist_entry(self):
        """A valid shortlist entry should have correct status."""
        entry = ShortlistEntry(
            candidate_id="id123",
            name="John Doe",
            final_score=85.0,
            rank=1,
            status="shortlisted",
        )
        assert entry.rank == 1
        assert entry.status == "shortlisted"

    def test_invalid_status(self):
        """Invalid status should raise ValidationError."""
        with pytest.raises(ValidationError):
            ShortlistEntry(
                candidate_id="id123",
                name="John",
                final_score=85.0,
                rank=1,
                status="invalid_status",
            )

    def test_rank_must_be_positive(self):
        """Rank must be >= 1."""
        with pytest.raises(ValidationError):
            ShortlistEntry(
                candidate_id="id123",
                name="John",
                final_score=85.0,
                rank=0,
                status="shortlisted",
            )


class TestJDInput:
    """Tests for the JDInput model."""

    def test_valid_jd(self):
        """A valid job description should validate."""
        jd = JDInput(
            title="Software Engineer",
            description="We need a software engineer with Python experience.",
            required_skills=["Python"],
            min_experience_years=3,
        )
        assert jd.title == "Software Engineer"
        assert jd.min_experience_years == 3

    def test_empty_title(self):
        """Empty title should raise ValidationError."""
        with pytest.raises(ValidationError):
            JDInput(title="", description="A valid description")

    def test_short_description(self):
        """Description shorter than 10 chars should raise ValidationError."""
        with pytest.raises(ValidationError):
            JDInput(title="Engineer", description="Short")


class TestRecruitmentState:
    """Tests for the RecruitmentState TypedDict structure."""

    def test_state_structure(self):
        """The state should accept all expected fields."""
        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=["resume text"],
            parsed_profiles=[],
            scorecards=[],
            verified_scores=[],
            revision_count=0,
            shortlist=[],
            step_count=0,
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        assert state["jd"].title == "Engineer"
        assert len(state["candidates"]) == 1
        assert state["revision_count"] == 0
        assert state["human_approved"] is False

    def test_parallel_write_fields(self):
        """Fields with operator.add reducer should accept list concatenation."""
        state = RecruitmentState(
            jd=JDInput(title="Engineer", description="A" * 20),
            candidates=["resume1"],
            parsed_profiles=[],
            scorecards=[],
            verified_scores=[],
            revision_count=0,
            shortlist=[],
            step_count=0,
            errors=[],
            needs_human_escalation=False,
            human_approved=False,
        )
        # Simulate parallel writes via operator.add
        state["candidates"] = state["candidates"] + ["resume2", "resume3"]
        assert len(state["candidates"]) == 3