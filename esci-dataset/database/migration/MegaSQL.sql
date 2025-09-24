-- MegaSQL: Combined database schema for ESCI Dataset
-- Combines all database creation scripts in order (001-006)

-- ========================================
-- 001: Create Labeler Table
-- ========================================
-- SQL script to create 'labeler' table in Supabase
create table labeler (
    id serial primary key,
    name text not null,
    role text not null check (role in ('labeler', 'reviewer')),
    type text not null,
    created_at timestamp with time zone default now()
);

-- ========================================
-- 002: Insert Labeler Info
-- ========================================
INSERT INTO labeler (name, role, type) VALUES
    ('Luv', 'labeler', 'human'),
    ('Nirant', 'labeler', 'human'),
    ('4784ad5', 'labeler', 'synthetic generator');

-- ========================================
-- 003: Create Query Table
-- ========================================
-- SQL script to create 'query' table
-- This table stores unique search queries with optional filters and metadata

create table if not exists query (
    -- Primary identifier
    id serial primary key,
    
    -- Query content
    query_content text not null,
    
    -- Optional structured filters (JSON)
    query_filters jsonb,
    
    -- Data generation metadata for tracking
    data_gen_hash text,
    
    -- Timestamps
    created_at timestamp with time zone default now()
);

-- Index for efficient query text searches
create index if not exists idx_query_content on query (query_content);

-- Index for filtering by data generation hash
create index if not exists idx_query_data_gen_hash on query (data_gen_hash);

-- Optional comments for clarity
comment on table query is 'Unique search queries with optional filters and metadata';
comment on column query.query_filters is 'JSON structure for search filters (cuisine, dietary restrictions, etc.)';
comment on column query.data_gen_hash is 'Hash for tracking data generation batches';
-- query_content can be text or image URL
comment on column query.query_content is 'The search query text or image URL';

-- ========================================
-- 004: Create Consumable Table
-- ========================================
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

-- ========================================
-- 005: Create Example Table
-- ========================================
-- SQL script to create 'example' table
-- This table represents the junction between queries and consumables

create table if not exists example (
    -- Primary identifier
    id serial primary key,
    
    -- Foreign keys
    query_id integer not null references query(id),
    consumable_id integer not null references consumable(id),
    
    -- Timestamps
    created_at timestamp with time zone default now()
);

-- Unique constraint to prevent duplicate query-consumable pairs
create unique index if not exists idx_example_query_consumable on example (query_id, consumable_id);

-- Index for efficient lookups by query
create index if not exists idx_example_query_id on example (query_id);

-- Index for efficient lookups by consumable
create index if not exists idx_example_consumable_id on example (consumable_id);

-- Optional comments for clarity
comment on table example is 'Junction table linking queries to consumables for ESCI labeling';
comment on column example.query_id is 'References query.id';
comment on column example.consumable_id is 'References consumable.id';

-- ========================================
-- 006: Create Label Table
-- ========================================
-- SQL script to create 'label' table
-- This table stores ESCI labels from human and AI labelers

create table if not exists label (
    -- Primary identifier
    id serial primary key,
    
    -- Foreign keys
    labeler_id integer not null references labeler(id),
    example_id integer not null references example(id),
    
    -- Label data
    esci_label text not null check (esci_label in ('E', 'S', 'C', 'I')),
    label_reason text,
    auto_label_score numeric(5,4), -- null for human labels
    
    -- Timestamps
    created_at timestamp with time zone default now()
);

-- Index for efficient lookups by labeler
create index if not exists idx_label_labeler_id on label (labeler_id);

-- Index for efficient lookups by example
create index if not exists idx_label_example_id on label (example_id);

-- Index for filtering by ESCI label
create index if not exists idx_label_esci_label on label (esci_label);

-- Index for filtering by auto-generated labels
create index if not exists idx_label_auto_score on label (auto_label_score) where auto_label_score is not null;

-- Optional comments for clarity
comment on table label is 'ESCI labels assigned by human or AI labelers';
comment on column label.esci_label is 'ESCI classification: Exact, Substitute, Complement, or Irrelevant';
comment on column label.auto_label_score is 'Confidence score for AI-generated labels (null for human labels)';

-- ========================================
-- 007: Add example_gen_hash to Example Table
-- ========================================
-- SQL script to add example_gen_hash column to 'example' table

ALTER TABLE example
ADD COLUMN example_gen_hash TEXT;

--=======================================
-- 008: Add mlflow_run_id to Query Table
--=======================================

-- SQL script to add mlflow_run_id column to 'query' table
-- This enables direct linking from database records to MLflow experiments

ALTER TABLE query
ADD COLUMN mlflow_run_id TEXT;

-- Add index for efficient lookups by MLflow run ID
CREATE INDEX IF NOT EXISTS idx_query_mlflow_run_id ON query (mlflow_run_id);

-- Optional comment for clarity
COMMENT ON COLUMN query.mlflow_run_id IS 'MLflow run ID for direct experiment access and detailed reproduction info';