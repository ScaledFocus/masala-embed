-- SQL script to create 'consumable' table
-- This table stores food items with their metadata and nutritional information

create table if not exists consumable (
    -- Primary identifier
    id serial primary key,
    
    -- Required fields
    image_url text not null,
    consumable_name text not null,
    
    -- Optional fields
    consumable_type text,
    consumable_ingredients text,
    consumable_portion_size text,
    consumable_nutritional_profile jsonb,
    consumable_cooking_method text,
    
    -- Timestamps
    created_at timestamp with time zone default now()
);

-- Index for efficient name searches
create index if not exists idx_consumable_name on consumable (consumable_name);

-- Index for filtering by type
create index if not exists idx_consumable_type on consumable (consumable_type);

-- Index for filtering by cooking method
create index if not exists idx_consumable_cooking_method on consumable (consumable_cooking_method);

-- Optional comments for clarity
comment on table consumable is 'Food or beverage items with metadata and nutritional information';
comment on column consumable.consumable_nutritional_profile is 'JSON structure for nutritional data (calories, macros, etc.)';