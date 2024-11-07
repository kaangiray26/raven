import os
import json
import shutil
from collections import deque
from lib.tables import get_tables, get_db

def get_config() -> dict:
    with open("config.json") as f:
        config = json.load(f)
    return config

def reset():
    # Remove the database
    if os.path.exists("raven.db"):
        os.remove("raven.db")

    # Remove directories
    shutil.rmtree("logs", ignore_errors=True)
    shutil.rmtree("temp", ignore_errors=True)
    shutil.rmtree("objects", ignore_errors=True)

def check(all=True, quiet=False):
    if not quiet:
        print("Waiter doing checks...")

    # Set the environment variable
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Prepare the directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("objects", exist_ok=True)

    # Check config
    if not os.path.exists("config.json"):
        # Copy the default config
        shutil.copy(os.path.join("lib", "extra", "default_config.json"), "config.json")
        exit("Please configure the config.json file before running the program.")

    # Load the config
    config = get_config()

    # Prepare the DB
    with get_db() as db:
        # Create tables
        deque(map(db.sql, get_tables(config["embeddings"]["dimensions"])))

        # Load the extensions
        db.install_extension("fts")
        db.load_extension("fts")
        db.execute("PRAGMA create_fts_index('papers', 'source_url', 'title', 'abstract', overwrite=1)")

        # Create the tables
        db.commit()

    # Return if not all
    if not all:
        return

    # Fix the queue
    print("Fixing the queue...")
    with get_db() as db:
        db.sql("CREATE OR REPLACE TEMP TABLE unfinished_jobs AS SELECT * FROM jobs WHERE finished_at IS NULL")
        db.sql("UPDATE jobs SET started_at = NULL WHERE id IN (SELECT id FROM unfinished_jobs)")
        db.commit()

    if not quiet:
        print("Waiter's job is done!")