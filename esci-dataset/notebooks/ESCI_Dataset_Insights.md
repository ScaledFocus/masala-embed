# AI GENERATED
# Amazon ESCI Dataset: Key Insights for Query2Dish Food Search Model

*Analysis based on exploration of Amazon ESCI (E-commerce Search Click-through) dataset*

## Dataset Overview

### Scale & Structure
- **130,652 unique queries** linking to **1.8M unique products**
- **2.6M total query-product pairs** (examples) with ESCI labels
- **Multilingual support**: English, Spanish, Japanese queries
- **Cross-locale product matching** capabilities

### ESCI Label Distribution
| Label | Percentage | Description | Count (approx) |
|-------|------------|-------------|----------------|
| **E (Exact)** | 65.2% | Products exactly matching query intent | 1.7M |
| **S (Substitute)** | 21.9% | Alternative products serving similar purpose | 574K |
| **I (Irrelevant)** | 10.0% | Products not matching query intent | 262K |
| **C (Complement)** | 2.9% | Products that accessorize/complement query | 76K |

## Query Behavior Patterns

### Frequency Distribution
- **Highly skewed long-tail distribution**
- Top query: **198 associated products** 
- Median: **20 products per query**
- Most queries (75th percentile): **≤31 products**
- **Popular generic queries**: "laptop", "shoes", "printer", "tv"

### Data Sources
| Source | Percentage | Description |
|--------|------------|-------------|
| other | 87.0% | General query collection |
| negations | 5.3% | Queries with negative terms |
| parse_pattern | 4.7% | Pattern-based query generation |
| behavioral | 2.9% | User behavior-derived queries |
| nlqec | 0.1% | Natural language query expansion |

## Food Database Design Implications

### 1. ESCI Framework Adaptation for Food

#### Exact (E) - Same Dish/Ingredient
- **Direct matches**: "pasta" → pasta products
- **Brand equivalents**: "coca cola" → Coca-Cola products
- **Measurement variations**: "1 cup flour" → flour products

#### Substitute (S) - Nutritionally/Functionally Similar
- **Dietary alternatives**: "milk" → almond milk, oat milk
- **Cooking method alternatives**: "fried chicken" → baked chicken, air-fried chicken
- **Nutritional equivalents**: "white rice" → brown rice, quinoa
- **Seasonal substitutes**: "fresh tomatoes" → canned tomatoes

#### Complement (C) - Pairing & Accompaniments
- **Side dishes**: "steak" → mashed potatoes, vegetables
- **Beverages**: "pizza" → soda, beer
- **Condiments**: "fries" → ketchup, mayo
- **Ingredients**: "pasta" → parmesan cheese, olive oil

#### Irrelevant (I) - Non-Food or Incompatible
- **Non-food items**: "apple" → iPhone (wrong context)
- **Incompatible dietary**: "vegan meal" → meat products
- **Wrong cuisine context**: "sushi" → pizza (unless fusion)

### 2. Database Schema Considerations

#### Query Table Extensions
```sql
-- Enhanced query table for food context
ALTER TABLE query ADD COLUMN cuisine_type text;
ALTER TABLE query ADD COLUMN dietary_restrictions text[];
ALTER TABLE query ADD COLUMN meal_type text; -- breakfast, lunch, dinner, snack
ALTER TABLE query ADD COLUMN cooking_method text;
```

#### Consumable Table Enhancements
```sql
-- Food-specific fields based on ESCI insights
ALTER TABLE consumable ADD COLUMN nutrition_tags text[];
ALTER TABLE consumable ADD COLUMN dietary_labels text[]; -- vegan, gluten-free, etc.
ALTER TABLE consumable ADD COLUMN cuisine_origin text;
ALTER TABLE consumable ADD COLUMN meal_categories text[];
ALTER TABLE consumable ADD COLUMN substitute_groups text[]; -- for S relationships
ALTER TABLE consumable ADD COLUMN complement_categories text[]; -- for C relationships
```

### 3. Search Strategy Implications

#### Long-tail Handling
- **Popular queries** (top 20%): Pre-compute results, extensive candidate sets
- **Niche queries** (bottom 80%): Real-time semantic matching, smaller candidate sets
- **Caching strategy**: Cache results for high-frequency food queries

#### Multilingual Food Search
- **Cultural dish names**: "ramen" vs "ラーメン"
- **Regional variations**: "eggplant" vs "aubergine"
- **Ingredient translations**: Support for ethnic cuisine ingredients

#### Substitute Discovery Pipeline
- **Nutritional similarity matching** (22% of relationships)
- **Cooking method compatibility**
- **Dietary restriction compliance**
- **Seasonal availability alternatives**

### 4. Labeling Strategy for Food Domain

#### Human Labelers
- **Nutrition experts**: For substitute relationships
- **Culinary professionals**: For complement pairings
- **Cultural consultants**: For ethnic cuisine accuracy

#### AI-Assisted Labeling
- **Nutritional database integration**: Automatic substitute suggestions
- **Recipe analysis**: Complement relationship extraction
- **Dietary restriction checking**: Automated relevance filtering

## Implementation Priorities

### Phase 1: Core ESCI Framework
1. Implement basic E/S/C/I labeling for food items
2. Build substitute relationship detection (highest ROI - 22% of data)
3. Create complement pairing system for meal planning

### Phase 2: Food-Specific Enhancements  
1. Integrate nutritional similarity matching
2. Add dietary restriction compliance checking
3. Implement cooking method-based substitution

### Phase 3: Advanced Features
1. Seasonal availability substitution
2. Cultural cuisine pairing intelligence
3. Personalized dietary preference adaptation

## Success Metrics

### Relevance Metrics
- **Exact match precision**: Target >90% for direct food queries
- **Substitute satisfaction**: Target >75% user acceptance for alternatives
- **Complement relevance**: Target >80% for meal pairing suggestions

### Coverage Metrics
- **Long-tail coverage**: Handle queries with <5 examples effectively
- **Multilingual support**: Support for top 5 cuisine languages
- **Dietary compliance**: 100% accuracy for allergen/dietary restrictions

---

*This analysis provides the foundation for designing a food-specific search system that leverages e-commerce search patterns while addressing the unique requirements of culinary applications.*