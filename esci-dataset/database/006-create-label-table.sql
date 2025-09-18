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