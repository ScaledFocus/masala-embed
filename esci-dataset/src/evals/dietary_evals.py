import json

import pandas as pd

from ..constants import (
    EGG,
    GLUTEN,
    HONEY,
    MILK,
    NON_SEAFOOD_NON_VEG,
    NON_VEG,
    NUTS,
    SEAFOOD,
)


# Single record evaluation functions
def is_non_veg(ingredients_str):
    if pd.isna(ingredients_str):
        return False
    try:
        ingredients = json.loads(ingredients_str)
    except Exception:
        ingredients = [ingredients_str]
    return any(any(nv in str(ing).lower() for nv in NON_VEG) for ing in ingredients)


def has_egg(ingredients_str):
    if pd.isna(ingredients_str):
        return False
    try:
        ingredients = json.loads(ingredients_str)
    except Exception:
        ingredients = [ingredients_str]
    return any(any(eg in str(ing).lower() for eg in EGG) for ing in ingredients)


def has_milk(ingredients_str):
    if pd.isna(ingredients_str):
        return False
    try:
        ingredients = json.loads(ingredients_str)
    except Exception:
        ingredients = [ingredients_str]
    return any(any(mk in str(ing).lower() for mk in MILK) for ing in ingredients)


def has_honey(ingredients_str):
    if pd.isna(ingredients_str):
        return False
    try:
        ingredients = json.loads(ingredients_str)
    except Exception:
        ingredients = [ingredients_str]
    return any(any(hn in str(ing).lower() for hn in HONEY) for ing in ingredients)


def has_gluten(ingredients_str):
    if pd.isna(ingredients_str):
        return False
    try:
        ingredients = json.loads(ingredients_str)
    except Exception:
        ingredients = [ingredients_str]
    return any(any(gt in str(ing).lower() for gt in GLUTEN) for ing in ingredients)


def has_nuts(ingredients_str):
    if pd.isna(ingredients_str):
        return False
    try:
        ingredients = json.loads(ingredients_str)
    except Exception:
        ingredients = [ingredients_str]
    return any(any(nt in str(ing).lower() for nt in NUTS) for ing in ingredients)


def has_seafood_only(ingredients_str):
    """Check if has seafood but no other non-veg (pescetarian-friendly)"""
    if pd.isna(ingredients_str):
        return False
    try:
        ingredients = json.loads(ingredients_str)
    except Exception:
        ingredients = [ingredients_str]

    has_seafood_ing = any(
        any(sf in str(ing).lower() for sf in SEAFOOD) for ing in ingredients
    )
    has_other_nonveg = any(
        any(nv in str(ing).lower() for nv in NON_SEAFOOD_NON_VEG) for ing in ingredients
    )

    return has_seafood_ing and not has_other_nonveg


# DataFrame evaluation functions
def evaluate_dietary_flags(df, ingredients_col="consumable_ingredients"):
    """Apply all dietary flags to a DataFrame."""
    df_eval = df.copy()

    df_eval["is_non_veg"] = df_eval[ingredients_col].apply(is_non_veg)
    df_eval["has_egg"] = df_eval[ingredients_col].apply(has_egg)
    df_eval["has_milk"] = df_eval[ingredients_col].apply(has_milk)
    df_eval["has_honey"] = df_eval[ingredients_col].apply(has_honey)
    df_eval["has_gluten"] = df_eval[ingredients_col].apply(has_gluten)
    df_eval["has_nuts"] = df_eval[ingredients_col].apply(has_nuts)
    df_eval["has_seafood_only"] = df_eval[ingredients_col].apply(has_seafood_only)

    return df_eval

def get_all_dietary_combinations(df):
    """After getting the dietary flags, get the following combinations:
        Vegetarian
        Vegan
        Non-vegetarian
        Halal
        Kosher
        Gluten-free
        Dairy-free
        Nut-free
        Low-carb
        Keto
        Paleo
        Diabetic-friendly
        Lacto Vegetarian
        Pescetarian
        Lacto-Ovo Vegetarian
        Gluten Free
        Nut Free
    """

    df_eval = df.copy()

    # Vegetarian: No non-veg
    df_eval["is_vegetarian"] = ~df_eval["is_non_veg"]

    # Vegan: No non-veg, no egg, no milk, no honey
    df_eval["is_vegan"] = (
        (~df_eval["is_non_veg"])
        & (~df_eval["has_egg"])
        & (~df_eval["has_milk"])
        & (~df_eval["has_honey"])
    )

    # Gluten-free
    df_eval["is_gluten_free"] = ~df_eval["has_gluten"]

    # Dairy-free
    df_eval["is_dairy_free"] = ~df_eval["has_milk"]

    # Nut-free
    df_eval["is_nut_free"] = ~df_eval["has_nuts"]

    # Lacto Vegetarian: Has milk, no egg, no non-veg
    df_eval["is_lacto_vegetarian"] = (
        (df_eval["has_milk"]) & (~df_eval["has_egg"]) & (~df_eval["is_non_veg"])
    )

    # Pescetarian: Has seafood only
    df_eval["is_pescetarian"] = df_eval["has_seafood_only"]

    # Lacto-Ovo Vegetarian: Has milk or egg, no non-veg
    df_eval["is_lacto_ovo_vegetarian"] = (
        (df_eval["has_milk"] | df_eval["has_egg"]) & (~df_eval["is_non_veg"])
    )
    dietary_columns = [
        "is_vegetarian",
        "is_vegan",
        "is_gluten_free",
        "is_dairy_free",
        "is_nut_free",
        "is_lacto_vegetarian",
        "is_pescetarian",
        "is_lacto_ovo_vegetarian",
    ]
    return df_eval, dietary_columns

def print_dietary_stats(df):
    """Print dietary statistics for a DataFrame."""
    total = len(df)

    print(f"Dietary Analysis for {total:,} items:")
    print(f"Non-veg: {df['is_non_veg'].sum():,} ({df['is_non_veg'].mean() * 100:.1f}%)")
    print(f"Contains egg: {df['has_egg'].sum():,} ({df['has_egg'].mean() * 100:.1f}%)")
    print(
        f"Contains milk:\
        {df['has_milk'].sum():,} ({df['has_milk'].mean() * 100:.1f}%)"
    )
    print(
        f"Contains honey:\
        {df['has_honey'].sum():,} ({df['has_honey'].mean() * 100:.1f}%)"
    )
    print(
        f"Contains gluten:\
        {df['has_gluten'].sum():,} ({df['has_gluten'].mean() * 100:.1f}%)"
    )
    print(
        f"Contains nuts:\
        {df['has_nuts'].sum():,} ({df['has_nuts'].mean() * 100:.1f}%)"
    )
    # vegetarian
    vegetarian = (~df["is_non_veg"]).sum()
    print(f"Vegetarian (no non-veg): {vegetarian:,} ({vegetarian / total * 100:.1f}%)")
    # no animal products, vegan
    vegan = (
        (~df["is_non_veg"]) & (~df["has_egg"]) & (~df["has_milk"]) & (~df["has_honey"])
    ).sum()
    print(f"Vegan (no animal products): {vegan:,} ({vegan / total * 100:.1f}%)")

    # Special combinations
    # milk not egg not non-veg
    milk_not_egg_nonveg = (df["has_milk"] & ~df["has_egg"] & ~df["is_non_veg"]).sum()
    print(
        f"Lacto Veg: {milk_not_egg_nonveg:,} ({milk_not_egg_nonveg / total * 100:.1f}%)"
    )
    # gluten free
    gluten_free = (~df["has_gluten"]).sum()
    print(f"Gluten-free: {gluten_free:,} ({gluten_free / total * 100:.1f}%)")
    # nut free
    nut_free = (~df["has_nuts"]).sum()
    print(f"Nut-free: {nut_free:,} ({nut_free / total * 100:.1f}%)")
    # pescetarian (seafood but no other meat)
    pescetarian = df["has_seafood_only"].sum()
    print(
        f"Pescetarian-friendly (seafood only):\
        {pescetarian:,} ({pescetarian / total * 100:.1f}%)"
    )
