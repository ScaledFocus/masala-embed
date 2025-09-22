"""Dynamic prompt template processing for query generation."""

import os
from typing import Optional
import pandas as pd


def load_prompt_template(template_path: str) -> str:
    """Load the prompt template from file."""
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt template not found at: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_markdown_table(df: pd.DataFrame, include_dietary: bool = False) -> str:
    """
    Generate a markdown table from DataFrame.

    Args:
        df: DataFrame with consumable data
        include_dietary: Whether to include dietary columns

    Returns:
        Markdown formatted table string
    """
    # Base columns that are always included
    base_columns = ['id', 'consumable_name', 'consumable_ingredients']

    # Dietary columns to include if requested
    dietary_columns = [
        'is_vegetarian', 'is_vegan', 'is_gluten_free', 'is_dairy_free',
        'is_nut_free', 'is_lacto_vegetarian', 'is_pescetarian', 'is_lacto_ovo_vegetarian'
    ]

    # Select columns based on dietary flag
    if include_dietary:
        # Check which dietary columns exist in the DataFrame
        available_dietary = [col for col in dietary_columns if col in df.columns]
        columns = base_columns + available_dietary
    else:
        columns = base_columns

    # Filter DataFrame to only include available columns
    available_columns = [col for col in columns if col in df.columns]
    df_filtered = df[available_columns].copy()

    # Convert to markdown using pandas built-in method
    return df_filtered[available_columns].to_markdown(index=False)


def load_query_examples(examples_path: str) -> str:
    """Load query examples from file."""
    if not os.path.exists(examples_path):
        raise FileNotFoundError(f"Examples file not found at: {examples_path}")

    with open(examples_path, 'r', encoding='utf-8') as f:
        return f.read()


def replace_template_placeholders(
    template: str,
    esci_label: str,
    markdown_table: str,
    queries_per_item: int = 5,
    examples_content: str = ""
) -> str:
    """
    Replace placeholders in the template with actual values.

    Args:
        template: The prompt template string
        esci_label: ESCI label (E, S, C, or I)
        markdown_table: Markdown formatted table of food candidates
        queries_per_item: Number of queries to generate per food item
        examples_content: Content to insert for examples placeholder (empty string if no examples)

    Returns:
        Template with placeholders replaced
    """
    # Replace ESCI label placeholder
    processed_template = template.replace("[E/S/C/I]", esci_label)

    # Replace food candidates placeholder
    processed_template = processed_template.replace("[FOOD CANDIDATES]", markdown_table)

    # Replace queries per item placeholder
    processed_template = processed_template.replace("[QUERIES_PER_ITEM]", str(queries_per_item))

    # Replace examples placeholder (empty string if no examples)
    processed_template = processed_template.replace("[QUERY_EXAMPLES]", examples_content)

    return processed_template


def prepare_prompt(
    template_path: str,
    df: pd.DataFrame,
    esci_label: str,
    batch_size: int,
    include_dietary: bool = False,
    queries_per_item: int = 5,
    query_examples_path: Optional[str] = None
) -> str:
    """
    Prepare the complete prompt by loading template and replacing placeholders.

    Args:
        template_path: Path to the prompt template file
        df: DataFrame with consumable data
        esci_label: ESCI label (E, S, C, or I)
        batch_size: Number of records to include in the table
        include_dietary: Whether to include dietary columns
        queries_per_item: Number of queries to generate per food item
        query_examples_path: Path to query examples file (None = no examples)

    Returns:
        Complete prompt ready for LLM
    """
    # Load the template
    template = load_prompt_template(template_path)

    # Limit DataFrame to batch size
    df_batch = df.head(batch_size)

    # Generate markdown table
    markdown_table = generate_markdown_table(df_batch, include_dietary)

    # Load examples if path provided
    examples_content = ""
    if query_examples_path:
        try:
            examples_content = load_query_examples(query_examples_path)
        except FileNotFoundError:
            # If examples file not found, proceed without examples
            examples_content = ""
            print(f"Warning: Examples file not found at {query_examples_path}, "
                  f"proceeding without examples")

    # Replace placeholders
    final_prompt = replace_template_placeholders(
        template, esci_label, markdown_table, queries_per_item, examples_content
    )

    return final_prompt


def get_esci_label_description(esci_label: str) -> str:
    """Get description for ESCI label."""
    descriptions = {
        "E": ("Exact match - items that directly and precisely match all "
              "constraints of the search query"),
        "S": "Substitute - items that can be a replacement for the queried item",
        "C": ("Complement - items that are related and typically paired "
              "together with the queried item"),
        "I": ("Irrelevant - products that do not satisfy the query in any "
              "meaningful way")
    }
    return descriptions.get(esci_label, f"Unknown ESCI label: {esci_label}")
