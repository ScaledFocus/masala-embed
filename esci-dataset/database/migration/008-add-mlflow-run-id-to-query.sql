-- SQL script to add mlflow_run_id column to 'query' table
-- This enables direct linking from database records to MLflow experiments

ALTER TABLE query
ADD COLUMN mlflow_run_id TEXT;

-- Add index for efficient lookups by MLflow run ID
CREATE INDEX IF NOT EXISTS idx_query_mlflow_run_id ON query (mlflow_run_id);

-- Optional comment for clarity
COMMENT ON COLUMN query.mlflow_run_id IS 'MLflow run ID for direct experiment access and detailed reproduction info';