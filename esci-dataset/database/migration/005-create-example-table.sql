-- SQL script to create 'example' table
-- This table represents the junction between queries and consumables

create table if not exists example (
    -- Primary identifier
    id serial primary key,
    
    -- Foreign keys
    query_id integer not null references query(id),
    consumable_id text not null references consumable(id),
    
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