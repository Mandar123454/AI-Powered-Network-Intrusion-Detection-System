-- AI-NIDS PostgreSQL Initialization Script
-- Creates necessary extensions and initial setup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create indexes for better query performance (will be applied after tables are created)
-- These are hints for manual optimization after Flask-Migrate creates the schema

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ai_nids TO nids;

-- Create schema for better organization (optional)
-- CREATE SCHEMA IF NOT EXISTS nids;
-- ALTER ROLE nids SET search_path TO nids, public;
