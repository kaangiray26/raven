import os
import re
import time
import shutil
import asyncio
import requests
import threading
import numpy as np
import pandas as pd
from uuid import uuid4
from base64 import b64encode
from datetime import datetime
from lib.job import Job
from lib.crawlers import Crawler, parse_source_url
from lib.crawlers.semantic import Semantic_Scholar
from lib.waiter import get_config
from lib.costs import pricing_for_usage
from lib.tables import get_db, row_factory, row_factory_list, row_factory_custom_list, rows as row_columns

# Calculate the centroid vector
def get_centroids(hash_list, embeddings_df):
    return [np.mean(np.array(embeddings_df[embeddings_df["hash"].isin(hashes)]["vec"].tolist()), axis=0) for hashes in hash_list]

# Calculate the mean vector of a list of vectors
def get_mean_vector(vectors) -> list:
    return np.mean(vectors, axis=0)

# Calculate text similarity
def fuzzy_score(text, query) -> float:
    # step 1: check for the complete query
    if re.search(rf'\b{re.escape(query)}\b', text, re.IGNORECASE):
        return 1.0

    # step 2: check for partial matches
    score = 0
    words = query.split()
    for word in words:
        score += len(re.findall(rf'\b{re.escape(word)}\b', text, re.IGNORECASE))

    # normalize the score
    return score / len(words)

def btoa(value):
    return b64encode(value.encode()).decode()

def min_max_normalize(value, min_val, max_val):
    range_val = max_val - min_val

    # Check for zero range
    if range_val == 0:
        return 0
    return (value - min_val) / range_val

def time_decay(published_at):
    # time_decay
    diff = (datetime.today() - published_at).days
    if diff < 365:
        decay_rate = 0.000200
    elif diff < 365 * 2:
        decay_rate = 0.000225
    elif diff < 365 * 5:
        decay_rate = 0.000250
    else:
        decay_rate = 0.000275
    return np.exp(-decay_rate * diff)

class Worker:
    def __init__(self):
        self.id = str(uuid4())
        self.kill = False
        self.running = []
        self.config = get_config()

        # citation_count, reference_count, author_count
        self.max_values = [0, 0, 0]

    # Config
    def reload_config(self):
        self.config = get_config()

    def get_max_values(self):
        with get_db() as db:
            return db.sql("""
                SELECT
                    MAX(citation_count) AS citation_count,
                    MAX(reference_count) AS reference_count,
                    MAX(len(authors)) AS author_count
                FROM papers
            """).df().fillna(0).to_numpy()[0]

    # Paper sort
    async def paper_sort(self, papers):
        # Create the function
        def rank_paper(_max_similarity, _similarity, _published_at, _citation_count, _reference_count, _author_count) -> float:
            # Set scores
            scores = {
                "similarity": 0,
                "recency": 0,
                "citation_score": 0,
                "reference_score": 0,
                "author_score": 0,
            }

            # similarity score
            if _max_similarity:
                similarity_score = np.log(1 + _similarity) / np.log(1 + _max_similarity)
                scores["similarity"] = max(0, min(similarity_score, 1))

            # time_decay
            diff = (datetime.today() - _published_at).days
            if diff < 365:
                decay_rate = 0.000200
            elif diff < 365 * 2:
                decay_rate = 0.000225
            elif diff < 365 * 5:
                decay_rate = 0.000250
            else:
                decay_rate = 0.000275
            recency = np.exp(-decay_rate * diff)
            scores["recency"] = max(0, min(recency, 1))

            # citation_count score
            if self.max_values[0]:
                citation_score = np.log(1 + _citation_count) / np.log(1 + self.max_values[0])
                scores["citation_score"] = max(0, min(citation_score, 1))

            # reference_count score
            if self.max_values[1]:
                reference_score = np.log(1 + _reference_count) / np.log(1 + self.max_values[1])
                scores["reference_score"] = max(0, min(reference_score, 1))

            # author_count score
            if self.max_values[2]:
                author_score = np.log(1 + _author_count) / np.log(1 + self.max_values[2])
                scores["author_score"] = max(0, min(author_score, 1))

            # Return the rank score using the weights
            return sum(scores[key] * self.config["ranking"]["weights"][key] for key in scores.keys())

        max_rank_score = np.max([paper["rank_score"] for paper in papers])

        # Sort papers using the rank_paper function
        papers = sorted(
            papers,
            key=lambda x: rank_paper(max_rank_score, x["rank_score"], x["published_at"], x["citation_count"], x["reference_count"], len(x["authors"])),
            reverse=True
        )

        return papers

    # Ranking methods
    async def rank_mixed(self, query):
        # Use bm25 to rank papers using titles and abstracts
        with get_db() as db:
            db.load_extension("fts")
            rows = db.execute("""
                SELECT *, rank_score
                FROM (
                    SELECT *, fts_main_papers.match_bm25(source_url, $1)
                    AS rank_score FROM papers
                ) sq
                WHERE rank_score IS NOT NULL
                ORDER BY rank_score DESC
                LIMIT $2
            """, (query, self.config["ranking"]["limit"])).fetchall()
        papers = row_factory_custom_list(row_columns["papers"] + ["rank_score"], rows)
        return papers

    async def rank_title(self, query):
        # Use bm25 to rank papers using titles
        with get_db() as db:
            rows = db.execute("""
                SELECT *, rank_score
                FROM (
                    SELECT *, fts_main_papers.match_bm25(source_url, $1, fields := 'title')
                    AS rank_score
                    FROM papers
                ) sq
                WHERE rank_score IS NOT NULL
                ORDER BY rank_score DESC
                LIMIT $2
            """, (query, self.config["ranking"]["limit"])).fetchall()

        papers = row_factory_custom_list(row_columns["papers"] + ["rank_score"], rows)
        return papers

    async def rank_abstract(self, query):
        # Use bm25 to rank papers using abstracts
        with get_db() as db:
            rows = db.execute("""
                SELECT *, rank_score
                FROM (
                    SELECT *, fts_main_papers.match_bm25(source_url, $1, fields := 'abstract')
                    AS rank_score
                    FROM papers
                ) sq
                WHERE rank_score IS NOT NULL
                ORDER BY rank_score DESC
                LIMIT $2
            """, (query, self.config["ranking"]["limit"])).fetchall()

        papers = row_factory_custom_list(row_columns["papers"] + ["rank_score"], rows)
        return papers

    async def rank_specter2(self, query):
        self.ensure("chat")
        self.ensure("embeds")

        # Create title and abstract from the query
        res = await self.chat.generate_title_abstract_from_query(query)

        # Create embeddings for the response
        embeddings = self.embeds.get_specter_embeddings([res["title"] + " " + res["abstract"]])[0]

        # Rank the papers
        with get_db() as db:
            rows = db.execute("""
                SELECT
                    p.*,
                    array_cosine_similarity(sv.vec, $1::FLOAT[768]) AS rank_score
                FROM papers p
                JOIN similar_vectors sv ON sv.source_url = p.source_url
                WHERE rank_score > 0.9
                ORDER BY rank_score DESC
                LIMIT $2
            """, (embeddings, self.config["ranking"]["limit"])).fetchall()

        # Return papers
        papers = row_factory_custom_list(row_columns["papers"] + ["rank_score"], rows)
        return papers

    async def rank_skyrank(self, query):
        if query.startswith("keywords:"):
            keywords = [word.strip() for word in query[9:].split(",")]
        elif query.startswith("llm:"):
            # Sort the keywords from general to specific
            self.ensure("chat")
            result = await self.chat.extract_keywords_and_sort(query)
            keywords = result["keywords"]
        elif len(query.split()) == 1:
            keywords = [query]
        else:
            self.ensure("docs")
            keywords = self.docs.extract_noun_chunks(query)

        # Fallback
        if not keywords:
            keywords = [query]
        print("Extracted keywords:", keywords)

        # Get embeddings for each keywords
        self.ensure("embeds")
        embeddings = self.embeds.get_embeddings_from_texts(keywords)

        # Get papers with multi-step semantic keyword filtering
        with get_db() as db:
            db.sql("""
                CREATE OR REPLACE TEMP TABLE
                matching_hashes (hash INT128 PRIMARY KEY, similarity FLOAT)
            """)
            # Start with filtering
            for embedding in embeddings:
                db.execute("""
                    INSERT INTO matching_hashes
                    SELECT DISTINCT ON (pnc.hash) pnc.hash,
                        array_cosine_similarity(e.vec, $1::FLOAT[384]) AS similarity
                    FROM paper_noun_chunks pnc
                    JOIN embeddings e ON e.hash = pnc.hash
                    WHERE array_cosine_similarity(e.vec, $1::FLOAT[384]) > $2
                    ORDER BY array_cosine_similarity(e.vec, $1::FLOAT[384]) DESC
                    LIMIT $3
                    ON CONFLICT DO NOTHING
                """, [embedding, self.config["ranking"]["skyrank_c"], self.config["ranking"]["skyrank_k"]])

            # Find papers
            rows = db.execute("""
                SELECT p.*, rank_score, keywords
                FROM (
                    SELECT
                        p.source_url,
                        array_agg(DISTINCT k.keyword) AS keywords,
                        len(keywords) AS rank_score
                    FROM paper_noun_chunks p
                    RIGHT JOIN matching_hashes mh ON mh.hash = p.hash
                    JOIN keywords k ON k.hash = p.hash
                    GROUP BY p.source_url
                ) AS sq
                JOIN papers p ON p.source_url = sq.source_url
                ORDER BY rank_score DESC
                LIMIT $1
            """, [self.config["ranking"]["limit"]]).fetchall()

        # Return papers
        papers = row_factory_custom_list(row_columns["papers"]+["rank_score", "keywords"], rows)
        return papers

    # Lazy load modules
    def ensure(self, module):
        match module:
            case "chat":
                if not hasattr(self, "chat"):
                    from lib.chat import Chat
                    self.chat = Chat()

            case "docs":
                if not hasattr(self, "docs"):
                    from lib.documents import Documents
                    self.docs = Documents()

            case "embeds":
                if not hasattr(self, "embeds"):
                    from lib.embeddings import Embeddings
                    self.embeds = Embeddings()

    def run(self):
        threading.Thread(target=self.listen).start()

    def stop(self):
        self.kill = True
        time.sleep(1)

    # Check the queue periodically for tasks to run
    def listen(self):
        # Ensure the embeds module is loaded
        self.ensure("embeds")

        print("Worker listening...")
        while not self.kill:
            # Is a sleep really necessary?
            time.sleep(1)
            with get_db() as db:
                rows = db.sql("SELECT * FROM jobs WHERE started_at IS NULL ORDER BY created_at ASC").fetchall()
            # If there are no tasks or there are tasks running, continue
            if not len(rows) or len(self.running):
                continue

            # Prepare the first task in the queue for running
            self.handle_job(rows[0])
        print("Worker stopped.")

    def handle_job(self, row):
        job = row_factory("jobs", row)
        self.running.append(job["id"])

        with get_db() as db:
        # Update the job status to get started
            db.execute("UPDATE jobs SET started_at = current_timestamp WHERE id = ?", (job["id"],))
            db.commit()

        # Run the job
        asyncio.run(self.run_job(job))

    def remove_temp_files(self):
        shutil.rmtree("temp", ignore_errors=True)
        os.makedirs("temp", exist_ok=True)

    async def get_paper_keywords(self, source_url):
        with get_db() as db:
            rows = db.execute("""
                SELECT DISTINCT ON (k.keyword) k.keyword
                FROM keywords k
                JOIN paper_noun_chunks pnc ON pnc.hash = k.hash
                WHERE pnc.source_url = $1
            """, [source_url]).fetchall()
        keywords = [row[0] for row in rows]
        return keywords

    async def add_to_favorites(self, source_url):
        # Insert to the database
        with get_db() as db:
            db.execute("INSERT INTO favorites (source_url) VALUES (?) ON CONFLICT DO NOTHING", [source_url])
            db.commit()

    async def remove_from_favorites(self, source_url):
        # Remove from table
        with get_db() as db:
            db.execute("DELETE FROM favorites WHERE source_url = ?", [source_url])
            db.commit()

    async def get_favorites(self, offset):
        # Get the favorites from the database
        with get_db() as db:
            # Find papers JOIN ON favorites
            rows = db.execute("""
                SELECT
                    p.*,
                    fav.added_at
                FROM papers p
                JOIN favorites fav ON p.source_url = fav.source_url
                ORDER BY fav.added_at DESC
                LIMIT 25
                OFFSET $1
            """, [offset]).fetchall()

        papers = row_factory_custom_list(row_columns["papers"] + ["added_at"], rows)

        # Return the papers
        return papers


    async def get_costs(self):
        # Get usage data from openai
        headers = {
            "Authorization": f"Bearer {self.config["api_keys"]["openai"]}"
        }
        params = {
            "date": datetime.today().strftime("%Y-%m-%d")
        }
        with requests.get("https://api.openai.com/v1/usage", headers=headers, params=params) as r:
            usage_data = r.json()["data"]

        # Calculate the total usage per model
        models = {}
        for data in usage_data:
            if data["snapshot_id"] not in models:
                models[data["snapshot_id"]] = 0
            models[data["snapshot_id"]] += data["n_context_tokens_total"]

        # Calculate the pricing for the usage
        costs = pricing_for_usage(models)
        return costs

    async def get_db_summary(self):
        with get_db() as db:
            # Can we get everything with just a single query?
            summary = db.sql("""
                SELECT
                    (SELECT COUNT(*) FROM papers) AS paper_count,
                    (SELECT COUNT(*) FROM topics) AS topic_count,
                    (SELECT COUNT(*) FROM keywords) AS keyword_count,
                    (SELECT
                        (SELECT COUNT(vec) FROM similar_vectors) + (SELECT COUNT(*) FROM embeddings)
                    ) AS vector_count,
                    (SELECT
                        round(SUM(EXTRACT(EPOCH FROM finished_at - started_at)), 2)
                        AS total_duration FROM jobs
                        WHERE finished_at IS NOT NULL
                    ) AS total_duration
            """).fetchdf()
        return {
            "paper_count": int(summary["paper_count"][0]),
            "topic_count": int(summary["topic_count"][0]),
            "keyword_count": int(summary["keyword_count"][0]),
            "vector_count": int(summary["vector_count"][0]),
            "total_duration": float(summary["total_duration"][0])
        }

    async def get_keywords(self, offset=0):
        # Get keywords from the database
        with get_db() as db:
            rows = db.execute("""
                SELECT *
                FROM keywords
                ORDER BY keyword ASC
                LIMIT 512
            OFFSET ?""", [offset]).fetchall()
            total_count = db.sql("SELECT COUNT(*) FROM keywords").fetchone()
        return {
            "keywords": row_factory_list("keywords", rows),
            "total_count": total_count[0]
        }

    async def crawl_papers(self, query, related) -> str:
        # Crawl the web
        crawler = Semantic_Scholar(related)
        crawler.crawl(query)

        # Create a DataFrame from the crawled data
        if not len(crawler.papers):
            return "No papers found!"

        # Insert papers into the database
        papers_df = pd.DataFrame(crawler.papers)
        with get_db() as db:
            db.sql(f"INSERT INTO papers BY NAME SELECT * FROM papers_df ON CONFLICT DO NOTHING")
            db.execute("PRAGMA create_fts_index('papers', 'source_url', 'title', 'abstract', overwrite=1)")
            db.commit()

        # Return the number of papers
        return f"Number of papers crawled: {len(crawler.papers)}"

    async def crawl_latest_papers(self, query, related) -> str:
        # Crawl the web
        crawler = Semantic_Scholar(related)
        crawler.crawl_latest(query)

        # Create a DataFrame from the crawled data
        if not len(crawler.papers):
            return "No papers found!"

        # Insert papers into the database
        papers_df = pd.DataFrame(crawler.papers)
        with get_db() as db:
            db.sql(f"INSERT INTO papers BY NAME SELECT * FROM papers_df ON CONFLICT DO NOTHING")
            db.execute("PRAGMA create_fts_index('papers', 'source_url', 'title', 'abstract', overwrite=1)")
            db.commit()

        # Return the number of papers
        return f"Number of papers crawled: {len(crawler.papers)}"

    async def crawl_paper(self, url) -> str:
        # Check conditionally which crawler to use
        crawler = Crawler()
        crawler.crawl_paper_by_url(url)

        # Create a DataFrame from the crawled data
        if not len(crawler.papers):
            return "No papers found!"

        # Insert papers into the database
        papers_df = pd.DataFrame(crawler.papers)
        with get_db() as db:
            db.sql(f"INSERT INTO papers BY NAME SELECT * FROM papers_df ON CONFLICT DO NOTHING")
            db.execute("PRAGMA create_fts_index('papers', 'source_url', 'title', 'abstract', overwrite=1)")
            db.commit()

        # Return the number of papers
        return f"Number of papers crawled: {len(crawler.papers)}"

    async def search_papers(self, query: str) -> list:
        with get_db() as db:
            rows = db.execute("SELECT * FROM papers WHERE title LIKE ?", [f"%{query}%"]).fetchall()
            papers = row_factory_list("papers", rows)
            return papers

    # Get the papers from the database
    async def find_papers(self, query, criterion="title", sorted=False):
        # Reload the config
        self.reload_config()

        # Get max values
        self.max_values = self.get_max_values()

        # Check the search criterion
        print("Finding papers:", query)
        match criterion:
            case "mixed":
                papers = await self.rank_mixed(query)
            case "title":
                papers = await self.rank_title(query)
            case "abstract":
                papers = await self.rank_abstract(query)
            case "specter2":
                papers = await self.rank_specter2(query)
            case "skyrank":
                papers = await self.rank_skyrank(query)
            case _:
                return []

        # Use paper scoring
        if sorted:
            return await self.paper_sort(papers)

        # Return papers
        return papers

    async def generate_answers(self, query, source_urls):
        if not len(source_urls):
            return None

        # Create a pandas df from urls with their order using enumerate
        ordered_url_df = pd.DataFrame(
            list(enumerate(source_urls)),
            columns=["order", "source_url"]
        )

        # Get papers based on the dataframe
        with get_db() as db:
            rows = db.sql("""
                SELECT p.title, p.abstract, p.source_url
                FROM papers p
                JOIN (
                    SELECT * FROM ordered_url_df
                ) ou ON p.source_url = ou.source_url
                ORDER BY ou.order
            """).fetchall()
        papers = row_factory_custom_list(["title", "abstract", "source_url"], rows)

        # Ensure modules are loaded
        self.ensure("chat")

        # Use RAG to answer the question from our findings
        return await self.chat.rag_with_citations(query, papers)

    # Find similar papers
    async def find_similar(self, source_url):
        # Use the similar_vectors table to find similar papers via array_cosine_similarity function
        with get_db() as db:
            rows = db.execute("""
                SELECT
                    p.*,
                    array_cosine_similarity(
                        (SELECT vec FROM similar_vectors WHERE source_url = $1),
                        sv.vec
                    ) AS similarity_score
                FROM papers p
                JOIN similar_vectors sv ON p.source_url = sv.source_url
                WHERE sv.source_url != $1 AND similarity_score > 0.9
                ORDER BY similarity_score DESC
            """, [source_url]).fetchall()
        papers = row_factory_custom_list(row_columns["papers"] + ["similarity_score"], rows)

        # Return the papers
        return papers

    async def get_topic_papers(self, topic_id, offset):
        print(f"Getting papers for topic {topic_id} at offset {offset}")
        # Find the papers related to the topic
        with get_db() as db:
            rows = db.execute("""
                SELECT * FROM papers
                WHERE related = ?
                ORDER BY published_at DESC
                LIMIT 25
                OFFSET ?
            """, [f"topics:{topic_id}", offset]).fetchall()
        papers = row_factory_list("papers", rows)

        # Return the papers
        return papers

    async def research(self, query):
        # Check for already processed entries
        with get_db() as db:
            row = db.execute("SELECT * FROM research WHERE query = ?", (query,)).fetchone()
        if row:
            research = row_factory("research", row)
            return {
                "type": "research",
                "data": str(research['id'])
            }

        # Create new entries
        research = {
            "id": str(uuid4()),
            "query": query
        }

        job = {
            "id": str(uuid4()),
            "type": "research",
            "argument": query,
            "related": f"research:{research['id']}"
        }

        # Insert into the database
        with get_db() as db:
            db.execute("""
                INSERT INTO research (id, query, updated_at)
                VALUES (?, ?, current_timestamp)
                ON CONFLICT DO NOTHING
            """, (research["id"], job["argument"]))
            db.execute("""
                INSERT INTO jobs (id, type, argument, related)
                VALUES (?, ?, ?, ?)
            """, (job["id"], job["type"], job["argument"], job["related"]))
            db.commit()

        return {
            "type": "research",
            "data": str(research['id'])
        }

    async def add_paper(self, query):
        # Parse the source_url
        source_url = parse_source_url(query)
        if not source_url:
            return {
                "type": "add_paper",
                "data": {
                    "error": "URL not supported"
                }
            }

        # Check paper source_url
        with get_db() as db:
            row = db.execute("SELECT * FROM papers WHERE source_url = ?", (source_url,)).fetchone()
        if row:
            return {
                "type": "route",
                "data": f"/papers/{b64encode(source_url.encode()).decode()}"
            }

        # Create new entries
        job = {
            "id": str(uuid4()),
            "type": "add_paper",
            "argument": source_url
        }

        # Insert into the database
        with get_db() as db:
            db.execute("INSERT INTO jobs (id, type, argument) VALUES (?, ?, ?)", (job["id"], job["type"], job["argument"]))
            db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "embeddings"))
            db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "similarity"))
            db.commit()

        return {
            "type": "route",
            "data": "/jobs"
        }

    async def recursive_search(self, source_url):
        # Create new entries
        job = {
            "id": str(uuid4()),
            "type": "recursive_search",
            "argument": source_url
        }

        # Insert into the database
        with get_db() as db:
            db.execute("INSERT INTO jobs (id, type, argument) VALUES (?, ?, ?)", (job["id"], job["type"], job["argument"]))
            db.commit()

        return {
            "type": "route",
            "data": "/jobs"
        }

    async def add_topic(self, query):
        # Transform query
        name = " ".join(query.lower().split()).strip()

        # Check for already processed entries
        with get_db() as db:
            row = db.execute("SELECT * FROM topics WHERE name = ?", (name,)).fetchone()
        if row:
            topic = row_factory("topics", row)
            return {
                "type": "topic",
                "data": str(topic['id'])
            }

        # Create new entries
        topic = {
            "id": str(uuid4()),
            "name": name
        }

        job = {
            "id": str(uuid4()),
            "type": "crawl",
            "argument": name,
            "related": f"topics:{topic["id"]}"
        }

        # Insert into the database
        with get_db() as db:
            db.execute("INSERT INTO topics (id, name) VALUES (?, ?) ON CONFLICT DO NOTHING", (topic["id"], topic["name"]))
            db.execute("""
                INSERT INTO jobs (id, type, argument, related)
                VALUES (?, ?, ?, ?)
            """, (job["id"], job["type"], job["argument"], job["related"]))
            db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "embeddings"))
            db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "similarity"))
            db.commit()

        return {
            "type": "route",
            "data": "/jobs"
        }

    async def download_pdf(self, id, url):
        # Check if we have an url
        print("Downloading:", url)
        if not url:
            return None

        # Check if we have the object already
        if os.path.exists(os.path.join("objects", id)):
            return os.path.join("objects", id)

        # Download the PDF
        with requests.get(url, stream=True, allow_redirects=True, headers={
            'user-agent': "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0"
        }) as r:
            with open(os.path.join("objects", id), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return os.path.join("objects", id)

    async def update_topic(self, id):
        # Check for topic
        with get_db() as db:
            row = db.execute("SELECT * FROM topics WHERE id = ?", (id,)).fetchone()
        if not row:
            return {
                "type": "route",
                "data": "/topics"
            }

        # Convert row to topic
        topic = row_factory("topics", row)

        # Create new job
        job = {
            "id": str(uuid4()),
            "type": "update",
            "argument": topic["name"],
            "related": f"topics:{topic["id"]}"
        }

        # Insert into the database
        with get_db() as db:
            db.execute("""
                INSERT INTO jobs (id, type, argument, related)
                VALUES (?, ?, ?, ?)
            """, (job["id"], job["type"], job["argument"], job["related"]))
            db.execute("""
                INSERT INTO jobs (id, type, argument)
                VALUES (?, ?, ?)
            """, (str(uuid4()), "update_topic_timestamp", topic["id"]))
            db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "embeddings"))
            db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "similarity"))
            db.commit()

        return {
            "type": "route",
            "data": "/jobs"
        }

    # Run the job asynchronously
    async def run_job(self, job):
        print("Running job:", job["id"], job["type"], job["argument"])
        job_instance = Job(job["id"], job["type"], job["argument"])

        # Check the type of the job
        match job["type"]:
            case "research":
                # Ensure modules are loaded
                self.ensure("chat")

                # Generate a list of topics and then add topic jobs to the queue
                res = await self.chat.get_topics_from_query(job["argument"])
                topics = res["topics"]

                # Transform query
                topic_names = [" ".join(topic.lower().split()).strip() for topic in topics]

                # Find topic names that are not already in the database
                with get_db() as db:
                    rows = db.execute("SELECT name FROM topics WHERE name IN ?", (topic_names,)).fetchall()
                existing_topic_names = [row[0] for row in rows]
                new_topic_names = [name for name in topic_names if name not in existing_topic_names]

                # Add logs
                job_instance.log(f"New topics generated: {new_topic_names}")

                # Create entries
                topic_ids = [str(uuid4()) for _ in range(len(new_topic_names))]
                topics_df = pd.DataFrame({
                    "id": topic_ids,
                    "name": new_topic_names
                })
                jobs_df = pd.DataFrame({
                    "id": [str(uuid4()) for _ in range(len(new_topic_names))],
                    "type": ["crawl" for _ in range(len(new_topic_names))],
                    "argument": new_topic_names,
                    "related": [f"topics:{topic_id}" for topic_id in topic_ids]
                })

                # Update the job and create new jobs
                with get_db() as db:
                    db.sql("INSERT INTO topics (id, name) SELECT * FROM topics_df")
                    db.sql("INSERT INTO jobs (id, type, argument, related) SELECT * FROM jobs_df")
                    db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "embeddings"))
                    db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "similarity"))
                    db.commit()

                # Set success
                job_instance.success = True

            case "crawl":
                # Search for the topic and collect the papers
                output = await self.crawl_papers(job["argument"], job["related"])

                # Add logs
                job_instance.log(output)

                # Set success
                job_instance.success = True

            case "embeddings":
                job_instance.log("Loading modules...")
                # Ensure modules are loaded
                self.ensure("docs")
                self.ensure("embeds")

                # Find all papers that do not have any paper_noun_chunks yet
                with get_db() as db:
                    rows = db.sql("""
                        SELECT title, abstract, source_url
                        FROM papers p
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM paper_noun_chunks pnc
                            WHERE pnc.source_url = p.source_url
                        ) AND p.abstract IS NOT NULL
                    """).fetchall()
                papers = row_factory_custom_list(["title", "abstract", "source_url"], rows)

                if len(papers):
                    # Extract noun_chunks
                    job_instance.log(f"Found {len(papers)} papers without noun_chunks.")
                    job_instance.log(f"Extracting keywords...")
                    self.docs.extract_noun_chunks([paper["title"] + " " + paper["abstract"] for paper in papers])
                    job_instance.log(f"Extracted {len(self.docs.keywords)} keywords.")

                    # Construct dataframes
                    keywords_df = pd.DataFrame({
                        "hash": list(self.docs.keywords.keys()),
                        "keyword": list(self.docs.keywords.values())
                    })
                    paper_noun_chunks_df = pd.DataFrame({
                        "source_url": [paper["source_url"] for paper in papers],
                        "hash": self.docs.paper_keywords
                    })

                    # Explode the hash
                    paper_noun_chunks_df = paper_noun_chunks_df.explode("hash", ignore_index=True)

                    # Insert data into the database
                    job_instance.log("Inserting data into the database...")
                    with get_db() as db:
                        db.sql("INSERT INTO keywords SELECT * FROM keywords_df ON CONFLICT DO NOTHING")
                        db.sql("INSERT INTO paper_noun_chunks BY NAME SELECT * FROM paper_noun_chunks_df")
                        db.commit()

                        # Get keywords without embeddings
                        keywords = db.sql("""
                            SELECT hash, keyword
                            FROM keywords k
                            WHERE NOT EXISTS (
                                SELECT 1 FROM embeddings e WHERE e.hash = k.hash
                            )
                        """).fetchall()

                    # Get embeddings for the new keywords
                    if len(keywords):
                        # Create embeddings for the keywords
                        job_instance.log(f"Creating embeddings for {len(keywords)} keywords...")
                        embeddings_df = pd.DataFrame({
                            "hash": [keyword[0] for keyword in keywords],
                            "vec": self.embeds.get_embeddings_from_texts([keyword[1] for keyword in keywords])
                        })

                        # Insert embeddings into the database
                        job_instance.log("Inserting embeddings into the database...")
                        with get_db() as db:
                            db.sql("INSERT INTO embeddings SELECT * FROM embeddings_df ON CONFLICT DO NOTHING")
                            db.commit()

                # Set success
                job_instance.success = True

            case "similarity":
                job_instance.log("Loading modules...")
                # Ensure modules are loaded
                self.ensure("embeds")

                # Find all papers that do not have any similar_vector yet
                job_instance.log("Getting papers...")
                with get_db() as db:
                    rows = db.sql("""
                        SELECT *
                        FROM papers p
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM similar_vectors sv
                            WHERE sv.source_url = p.source_url
                        ) AND p.abstract IS NOT NULL
                    """).fetchall()
                papers = row_factory_list("papers", rows)
                job_instance.log(f"Found {len(papers)} papers with an abstract.")

                if len(papers):
                    # Get embeddings for all title+abstract pairs
                    job_instance.log(f"Getting embeddings using the specter2 model...")
                    similar_vectors_df = pd.DataFrame({
                        "source_url": [paper["source_url"] for paper in papers],
                        "vec": [embedding for embedding in self.embeds.get_specter_embeddings([paper["title"] + "[SEP]" + paper["abstract"] for paper in papers])]
                    })

                    # Insert embeddings into the database
                    job_instance.log("Inserting embeddings into the database...")
                    with get_db() as db:
                        db.sql("INSERT INTO similar_vectors SELECT * FROM similar_vectors_df ON CONFLICT DO NOTHING")
                        db.commit()

                # Set success
                job_instance.success = True

            case "update_topic_timestamp":
                # Update the topic's updated_at field
                with get_db() as db:
                    db.execute("UPDATE topics SET updated_at = current_timestamp WHERE id = ?", (job["argument"],))
                    db.commit()

                # Set success
                job_instance.success = True

            case "add_paper":
                # Crawl a paper and add it to the database
                try:
                    output = await self.crawl_paper(job["argument"])
                    job_instance.log(output)
                    job_instance.success = True
                except Exception as e:
                    job_instance.log(str(e))

            case "recursive_search":
                try:
                    # Ensure modules are loaded
                    self.ensure("chat")

                    # Get the paper entry
                    with get_db() as db:
                        row = db.execute("SELECT * FROM papers WHERE source_url = ?", [job["argument"]]).fetchone()
                    paper = row_factory("papers", row)

                    # Download the PDF
                    filepath = await self.download_pdf(btoa(paper["source_url"]), paper["pdf_url"])

                    # Use the chat model to digest the paper and extract search queries for further papers
                    res = await self.chat.get_queries_from_paper(paper["title"], paper["abstract"], filepath)
                    queries = res["queries"]

                    # Transform queries
                    topic_names = [" ".join(topic.lower().split()).strip() for topic in queries]

                    # Find topic names that are not already in the database
                    with get_db() as db:
                        rows = db.execute("SELECT name FROM topics WHERE name IN ?", [topic_names]).fetchall()
                    existing_topic_names = [row[0] for row in rows]
                    new_topic_names = [name for name in topic_names if name not in existing_topic_names]

                    # Log the queries
                    job_instance.log(f"New queries: {topic_names}")

                    # Create entries
                    topic_ids = [str(uuid4()) for _ in range(len(new_topic_names))]
                    topics_df = pd.DataFrame({
                        "id": topic_ids,
                        "name": new_topic_names
                    })
                    jobs_df = pd.DataFrame({
                        "id": [str(uuid4()) for _ in range(len(new_topic_names))],
                        "type": ["crawl" for _ in range(len(new_topic_names))],
                        "argument": new_topic_names,
                        "related": [f"topics:{topic_id}" for topic_id in topic_ids]
                    })

                    # Insert entries into the database
                    with get_db() as db:
                        db.sql("INSERT INTO topics (id, name) SELECT * FROM topics_df")
                        db.sql("INSERT INTO jobs (id, type, argument, related) SELECT * FROM jobs_df")
                        db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "embeddings"))
                        db.execute("INSERT INTO jobs (id, type) VALUES (?, ?)", (str(uuid4()), "similarity"))
                        db.commit()

                    # Set success
                    job_instance.success = True
                except Exception as e:
                    job_instance.log(str(e))

            case "update":
                # Update the topic with the latest papers
                # Search for the topic and collect the papers
                output = await self.crawl_latest_papers(job["argument"], job["related"])

                # Add logs
                job_instance.log(output)

                # Set success
                job_instance.success = True

            case _:
                print("Unknown job type:", job["type"])
                job_instance.log("Unknown job type.")

        # Add log
        job_instance.log("Job finished.")

        # Update the job
        with get_db() as db:
            db.execute("""
                UPDATE jobs
                SET
                    finished_at = current_timestamp,
                    success = ?,
                    logfile = ?
                WHERE id = ?
            """, (job_instance.success, job["id"], job["id"]))
            db.commit()

        # Remove the task from the running list
        self.running.remove(job["id"])
        print("Job:", job["id"], job["type"], job["argument"])