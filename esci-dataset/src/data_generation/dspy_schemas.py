"""DSPy signatures and schemas for query generation with structured Pydantic output."""

import json  # Still needed for DataFrame conversion

import dspy
import pandas as pd
from pydantic import BaseModel, Field


class QueryDimensions(BaseModel):
    """Schema for query dimensions."""

    dietary_restrictions: str = Field(
        default="", description="Dietary restrictions (e.g., Vegetarian, Vegan)"
    )
    cuisine: str = Field(default="", description="Cuisine type (e.g., Indian, Italian)")
    healthiness: str = Field(
        default="", description="Health attributes (e.g., Healthy, Low-calorie)"
    )
    meal_type: str = Field(default="", description="Meal type (e.g., Breakfast, Lunch)")
    nutritional_profile: str = Field(
        default="", description="Nutrition info (e.g., High Protein)"
    )
    urgency: str = Field(
        default="", description="Delivery urgency (e.g., fast delivery)"
    )
    price: str = Field(default="", description="Price range (e.g., cheap, budget)")
    location: str = Field(default="", description="Location context (e.g., near me)")


class GeneratedQuery(BaseModel):
    """Schema for a single generated query with dimensions."""

    query: str = Field(
        description="Natural language search query like 'paneer tika delivry', 'cheap vegan dinner near me', or 'quick breakfast under $10'. Include realistic imperfections like typos, casual language, and varied vocabulary."
    )
    dimensions: dict[str, str] = Field(
        default_factory=dict,
        description="Query attributes as key-value pairs. Examples: {'price': 'cheap', 'location': 'near me'}, {'cuisine': 'Indian', 'urgency': 'fast delivery'}, {'dietary_restrictions': 'Vegetarian', 'meal_type': 'Starters'}"
    )


class CandidateQueries(BaseModel):
    """Schema for queries generated for a single candidate."""

    id: int = Field(description="Candidate ID from the food database")
    name: str = Field(description="Candidate food name (e.g., 'Paneer Tikka', 'Braised Tofu')")
    queries: list[GeneratedQuery] = Field(
        description="List of generated queries for this candidate, varying from simple to complex with different dimensions"
    )


class QueryGenerationOutput(BaseModel):
    """Schema for the complete output."""

    candidates: list[CandidateQueries] = Field(
        description="List of food candidates, each with multiple realistic search queries covering different complexity levels and user scenarios"
    )


class QueryGenerationSignature(dspy.Signature):
    """DSPy signature for query generation task with structured output."""

    prompt_with_candidates = dspy.InputField(
        desc="Complete prompt with food candidates and instructions"
    )
    esci_label = dspy.InputField(desc="ESCI label (E/S/C/I) to generate queries for")

    generated_queries: QueryGenerationOutput = dspy.OutputField(
        desc="Structured output with candidates and generated queries with dimensions"
    )


class QueryGenerator(dspy.Module):
    """DSPy module for generating food delivery queries with structured output."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(QueryGenerationSignature)

    def forward(
        self, prompt_with_candidates: str, esci_label: str
    ) -> QueryGenerationOutput:
        """Generate queries using DSPy with structured Pydantic output."""
        result = self.generate(
            prompt_with_candidates=prompt_with_candidates, esci_label=esci_label
        )
        return result.generated_queries


# Legacy parse_generated_output function removed in modernized version
# Use structured QueryGenerator.forward() instead


def convert_output_to_dataframe(output: QueryGenerationOutput):
    """
    Convert QueryGenerationOutput to pandas DataFrame with one query per row.

    Args:
        output: QueryGenerationOutput object with candidates and queries

    Returns:
        pandas.DataFrame with columns:
        - candidate_id: int
        - candidate_name: str
        - query: str
        - dimensions_json: str (JSON string of dimensions)
        - Individual dimension columns (dim_cuisine, dim_price, etc.)
    """
    # Import constants for standardized dimension columns
    from src.constants import FOOD_QUERY_DIMENSIONS

    rows = []

    # Use standardized dimension keys from constants instead of output
    dimension_columns = sorted(list(FOOD_QUERY_DIMENSIONS.keys()))

    # Process each candidate and query
    for candidate in output.candidates:
        for query_obj in candidate.queries:
            row = {
                "consumable_id": candidate.id,
                "consumable_name": candidate.name,
                "query": query_obj.query,
                "dimensions_json": json.dumps(query_obj.dimensions)
                if query_obj.dimensions
                else "{}",
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns for better readability
    base_columns = ["consumable_id", "consumable_name", "query", "dimensions_json"]
    df = df[base_columns]

    return df


def setup_dspy_model(
    api_key: str, model: str = "gpt-5-mini", temperature: float = 0.7
) -> None:
    """Setup DSPy with OpenAI model."""
    try:
        # Check if it's a GPT-5 reasoning model with special requirements
        if model.startswith("gpt-5"):
            lm = dspy.LM(
                model=f"openai/{model}",
                api_key=api_key,
                max_tokens=128000,  # GPT-5 requires max_tokens >= 16000
                temperature=1.0,  # GPT-5 requires temperature=1.0
                model_type="responses",  # GPT-5 requires model_type='responses'
            )
        else:
            # GPT-4 and earlier models use standard parameters
            lm = dspy.LM(
                model=f"openai/{model}",
                api_key=api_key,
                max_tokens=4000,
                temperature=temperature,
            )
    except Exception as e:
        # Provide detailed error information
        raise RuntimeError(
            f"DSPy LM initialization failed for model '{model}': {str(e)}"
        ) from e
    dspy.settings.configure(lm=lm)
