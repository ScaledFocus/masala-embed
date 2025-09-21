"""DSPy signatures and schemas for query generation."""

import json
from typing import Dict, List, Union, Any

import dspy
from pydantic import BaseModel, Field


class QueryDimensions(BaseModel):
    """Schema for query dimensions."""

    dietary_restrictions: str = Field(
        default="", description="Dietary restrictions (e.g., Vegetarian, Vegan)"
    )
    cuisine: str = Field(
        default="", description="Cuisine type (e.g., Indian, Italian)"
    )
    healthiness: str = Field(
        default="", description="Health attributes (e.g., Healthy, Low-calorie)"
    )
    meal_type: str = Field(default="", description="Meal type (e.g., Breakfast, Lunch)")
    nutritional_profile: str = Field(
        default="", description="Nutrition info (e.g., High Protein)"
    )
    urgency: str = Field(default="", description="Delivery urgency (e.g., fast delivery)")
    price: str = Field(default="", description="Price range (e.g., cheap, budget)")
    location: str = Field(default="", description="Location context (e.g., near me)")


class GeneratedQuery(BaseModel):
    """Schema for a single generated query."""

    query: str = Field(description="The natural language query")
    dimensions: Dict[str, Any] = Field(
        default_factory=dict, description="Query dimensions"
    )


class CandidateQueries(BaseModel):
    """Schema for queries generated for a single candidate."""

    id: int = Field(description="Candidate ID")
    name: str = Field(description="Candidate name")
    queries: List[GeneratedQuery] = Field(description="List of generated queries")


class QueryGenerationOutput(BaseModel):
    """Schema for the complete output."""

    candidates: List[CandidateQueries] = Field(description="List of candidates with queries")


class QueryGenerationSignature(dspy.Signature):
    """DSPy signature for query generation task."""

    prompt_with_candidates = dspy.InputField(
        desc="Complete prompt with food candidates and instructions"
    )
    esci_label = dspy.InputField(
        desc="ESCI label (E/S/C/I) to generate queries for"
    )

    generated_queries = dspy.OutputField(
        desc="JSON string containing generated queries following the specified format"
    )


class QueryGenerator(dspy.Module):
    """DSPy module for generating food delivery queries."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(QueryGenerationSignature)

    def forward(self, prompt_with_candidates: str, esci_label: str) -> str:
        """Generate queries using DSPy."""
        result = self.generate(
            prompt_with_candidates=prompt_with_candidates,
            esci_label=esci_label
        )
        return result.generated_queries


def parse_generated_output(json_str: str) -> QueryGenerationOutput:
    """Parse the generated JSON string into structured output."""
    try:
        data = json.loads(json_str)
        return QueryGenerationOutput(**data)
    except (json.JSONDecodeError, Exception) as e:
        raise ValueError(f"Failed to parse generated output: {e}") from e


def setup_dspy_model(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> None:
    """Setup DSPy with OpenAI model."""
    try:
        # Try newer DSPy API first
        lm = dspy.LM(
            model=f"openai/{model}",
            api_key=api_key,
            max_tokens=4000,
            temperature=temperature
        )
    except Exception as e:
        # Raise Error
        raise RuntimeError("Check parameters for dspy.LM initialization.") from e
    dspy.settings.configure(lm=lm)