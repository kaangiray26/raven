import duckdb

research_table = """
CREATE TABLE IF NOT EXISTS research (
    id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""

topics_table = """
CREATE TABLE IF NOT EXISTS topics (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""

jobs_table = """
CREATE TABLE IF NOT EXISTS jobs(
    id UUID PRIMARY KEY,
    type TEXT DEFAULT NULL,
    argument TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp,
    started_at TIMESTAMP DEFAULT NULL,
    finished_at TIMESTAMP DEFAULT NULL,
    success BOOLEAN DEFAULT FALSE,
    logfile TEXT DEFAULT NULL,
    related TEXT DEFAULT NULL
)
"""

papers_table = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT,
    source TEXT,
    title TEXT,
    abstract TEXT,
    authors TEXT[],
    refs TEXT[],
    published_at TIMESTAMP,
    reference_count INT,
    citation_count INT,
    pdf_url TEXT,
    source_url TEXT PRIMARY KEY,
    related_url INT,
    cited_by_url TEXT,
    related TEXT DEFAULT NULL,
    added_at TIMESTAMP DEFAULT current_timestamp,
)
"""

keywords_table = """
CREATE TABLE IF NOT EXISTS keywords (
    hash INT128 PRIMARY KEY,
    keyword TEXT
)
"""

embeddings_table = """
CREATE TABLE IF NOT EXISTS embeddings (
    hash INT128 PRIMARY KEY,
    vec FLOAT[{dimension}]
)
"""

paper_noun_chunks_table = """
CREATE TABLE IF NOT EXISTS paper_noun_chunks (
    id UUID DEFAULT uuid(),
    source_url TEXT,
    hash INT128
)
"""

similar_vectors_table = """
CREATE TABLE IF NOT EXISTS similar_vectors (
    source_url TEXT PRIMARY KEY,
    vec FLOAT[768]
)
"""

favorites_table = """
CREATE TABLE IF NOT EXISTS favorites (
    source_url TEXT PRIMARY KEY,
    added_at TIMESTAMP DEFAULT current_timestamp
)
"""

# Tables
tables = [
    research_table,
    topics_table,
    jobs_table,
    papers_table,
    keywords_table,
    embeddings_table,
    similar_vectors_table,
    favorites_table,
    paper_noun_chunks_table
]

# Table keys
rows = {
    "research": ["id", "query", "updated_at"],
    "topics": ["id", "name", "updated_at"],
    "jobs": ["id", "type", "argument", "created_at", "started_at", "finished_at", "success", "logfile", "related"],
    "topics_and_jobs": ["query", "id", "topic_id", "created_at", "started_at", "finished_at", "success", "logfile"],
    "papers": ["id", "source", "title", "abstract", "authors", "refs", "published_at", "reference_count", "citation_count", "pdf_url", "source_url", "related_url", "cited_by_url", "related", "added_at"],
    "paper_vectors": ["source_url", "vec"],
    "similar_vectors": ["source_url", "vec"],
    "keywords": ["hash", "keyword"],
    "favorites": ["source_url", "added_at"]
}

# Helper functions
def row_factory(table, row):
    return dict(zip(rows[table], row))

def row_factory_list(table: str, rows: list):
    return [row_factory(table, row) for row in rows]

def row_factory_custom(keys: list, row):
    return dict(zip(keys, row))

def row_factory_custom_list(keys: list, rows: list):
    return [row_factory_custom(keys, row) for row in rows]

def get_db():
    return duckdb.connect("raven.db", config={
        "autoinstall_known_extensions": True,
        "autoload_known_extensions": True
    })

def get_tables(dimension):
    return [table.format(dimension=dimension) for table in tables]