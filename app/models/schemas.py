"""Data Models (Schemas) for MYRROR. Uses Pydantic to force Gemini into returning structured, typed JSON."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PEOPLE EXTRACTION SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PersonNotesSchema(BaseModel):
    description: Optional[str] = Field(default=None, description="brief description")
    context: Optional[str] = Field(default=None, description="how they came up in conversation")
    power_dynamic: Optional[str] = Field(default=None, description="e.g., user seeks their approval, user feels intimidated, codependent, equal, etc.")

class PersonSchema(BaseModel):
    name: str
    relationship: Optional[str] = Field(default=None)
    notes: Optional[PersonNotesSchema] = Field(default=None)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PSYCHOLOGICAL PROFILE SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClinicalProfileSchema(BaseModel):
    big_five: Optional[Dict[str, int]] = Field(default=None, description="Dict with O, C, E, A, N integer scores 1-10")
    enneagram: Optional[str] = Field(default=None)
    mbti: Optional[str] = Field(default=None, description="MBTI personality type (e.g., INTJ, ENFP)")
    archetype: Optional[str] = Field(default=None)

class EventItemSchema(BaseModel):
    event: str
    timeframe: str

class ProfileSchema(BaseModel):
    name: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None, description="The primary language the user writes in (e.g. 'Spanish', 'English', 'French')")
    age: Optional[int] = Field(default=None)
    location: Optional[str] = Field(default=None)
    job: Optional[str] = Field(default=None)
    life_compass: Optional[str] = Field(default=None)
    current_mood_score: Optional[int] = Field(default=None)
    myrror_strategy: Optional[str] = Field(default=None, description="your master plan for guiding this user")
    upcoming_events: Optional[List[EventItemSchema]] = Field(default=None)
    unresolved_threads: Optional[List[str]] = Field(default=None, description="pending topics to follow up on")
    goals: Optional[List[str]] = Field(default=None)
    fears: Optional[List[str]] = Field(default=None)
    strengths: Optional[List[str]] = Field(default=None)
    weaknesses: Optional[List[str]] = Field(default=None)
    personality_traits: Optional[List[str]] = Field(default=None)
    emotional_state: Optional[str] = Field(default=None)
    emotional_patterns: Optional[List[str]] = Field(default=None)
    media_and_tastes: Optional[List[str]] = Field(default=None, description="Books, music, movies, hobbies, or art they consume")
    avoidance_patterns: Optional[List[str]] = Field(default=None, description="Topics they actively dodge or how they deflect hard questions")
    emotional_volatility: Optional[str] = Field(default=None, description="Assessment of their emotional stability (e.g., stable, highly volatile, numbed, erratic)")
    communication_style: Optional[str] = Field(default=None)
    relationship_patterns: Optional[List[str]] = Field(default=None)
    state_of_mind_anomalies: Optional[List[str]] = Field(default=None, description="Instances of altered states: sleep deprivation, intoxication, manic bursts, etc.")
    core_values: Optional[List[str]] = Field(default=None)
    humor_style: Optional[str] = Field(default=None)
    decision_making: Optional[str] = Field(default=None)
    self_perception: Optional[str] = Field(default=None)
    life_situations: Optional[List[str]] = Field(default=None)
    skills: Optional[List[str]] = Field(default=None)
    learning: Optional[List[str]] = Field(default=None)
    tech_level: Optional[str] = Field(default=None)
    cultural_background: Optional[str] = Field(default=None)
    preferred_tone: Optional[str] = Field(default=None)
    failed_advice: Optional[List[str]] = Field(default=None)
    detected_patterns: Optional[List[str]] = Field(default=None)
    contradictions: Optional[List[str]] = Field(default=None)
    personal_contracts: Optional[List[str]] = Field(default=None)
    insights_from_files: Optional[List[str]] = Field(default=None)
    growth_areas: Optional[List[str]] = Field(default=None)
    core_beliefs: Optional[List[str]] = Field(default=None)
    cognitive_biases: Optional[List[str]] = Field(default=None)
    data_source: Optional[str] = Field(default="inferred", description="explicit|inferred")
    unspoken_fears: Optional[List[str]] = Field(default=None)
    unmet_needs: Optional[List[str]] = Field(default=None)
    shadow_traits: Optional[List[str]] = Field(default=None)
    interaction_manual: Optional[List[str]] = Field(default=None)
    attachment_style: Optional[str] = Field(default=None)
    clinical_profile: Optional[ClinicalProfileSchema] = Field(default=None)
    behavioral_patterns: Optional[List[str]] = Field(default=None, description="Habits, frequent moods, and behavioral tendencies")
    quirks_and_micro_details: Optional[List[str]] = Field(default=None, description="Typo patterns, ignored topics, recurring complaints, minor quirks")
    cognition_style: Optional[str] = Field(default=None, description="How they process information: logical, emotional, impulsive, reflective, etc.")
    psyche_and_motivations: Optional[str] = Field(default=None, description="Detailed analysis of their psyche and underlying motivations")
    unrealized_truths: Optional[List[str]] = Field(default=None, description="objective facts they haven't realized")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EPISODES AND MEMORY SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EpisodeSchema(BaseModel):
    event: str
    domain: Optional[str] = Field(default=None, description="tech|work|personal|health|finance|relationships|learning|emotional")
    impact: Optional[str] = Field(default=None, description="high|medium|low")

class QuizSchema(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 options")