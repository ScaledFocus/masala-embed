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
