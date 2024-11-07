import requests
from datetime import datetime
from lib.paper import Paper

class Semantic_Scholar:
    def __init__(self, related=None, limit=10):
        self.papers = []
        self.limit = limit
        self.related = related
        self.headers = {
            "Content-Type": "application/json",
        }

    def crawl_paper_by_url(self, url) -> dict:
        print(f"Crawling Semantic Scholar for {url}...")

        # Get the paper ID
        paper_id = url.split("/")[-1]
        return self.crawl_paper_by_id(paper_id)

    def crawl_paper_by_id(self, id) -> dict:
        # Make the request
        with requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={
                "fields":"paperId,url,title,authors,externalIds,abstract,referenceCount,citationCount,openAccessPdf,year,publicationDate",
            },
            json={"ids": [id]}
        ) as r:
            if r.status_code != 200:
                print(r.text)
                raise Exception("Failed to fetch the page!")

            # Get the JSON
            docs = r.json()

            # Parse the JSON
            return self.get_paper_from_entry(docs[0])

    def relevance_search(self, topic, offset):
        print(f"Crawling Semantic Scholar for {topic}...")

        # Make the request
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": topic,
            "offset": offset,
            "fields":"paperId,url,title,authors,externalIds,abstract,referenceCount,citationCount,openAccessPdf,year,publicationDate",
            "openAccessPdf":"",
            "minCitationCount":"1",
            "year":"1990-",
            "fieldsOfStudy":"Computer Science"
        }

        with requests.get(url, params=params, headers=self.headers) as r:
            if r.status_code != 200:
                print(r.text)
                raise Exception("Failed to fetch the page!")

            # Get the JSON
            doc = r.json()

            # Parse the JSON
            self.papers.extend([self.get_paper_from_entry_sorted(entry, index+offset) for (index, entry) in enumerate(doc["data"])])

    def crawl(self, topic):
        print(f"Crawling Semantic Scholar for {topic}...")

        # Make the request
        url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        params = {
            "query": topic,
            "fields":"paperId,url,title,authors,externalIds,abstract,referenceCount,citationCount,openAccessPdf,year,publicationDate",
            "sort":"citationCount:desc",
            "openAccessPdf":"",
            "minCitationCount":"1",
            "year":"1990-",
            "fieldsOfStudy":"Computer Science"
        }

        with requests.get(url, params=params, headers=self.headers) as r:
            if r.status_code != 200:
                print(r.text)
                raise Exception("Failed to fetch the page!")

            # Get the JSON
            doc = r.json()

            # Parse the JSON
            self.parse_document(doc, "data")

    def crawl_latest(self, topic):
        print(f"Crawling Semantic Scholar for {topic}...")

        # Make the request
        url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        params = {
            "query": topic,
            "fields":"paperId,url,title,authors,externalIds,abstract,referenceCount,citationCount,openAccessPdf,year,publicationDate",
            "sort":"publicationDate:desc",
            "openAccessPdf":"",
            "minCitationCount":"1",
            "year":"1990-",
            "fieldsOfStudy":"Computer Science"
        }

        with requests.get(url, params=params, headers=self.headers) as r:
            if r.status_code != 200:
                print(r.text)
                raise Exception("Failed to fetch the page!")

            # Get the JSON
            doc = r.json()

            # Parse the JSON
            self.parse_document(doc, "data")

    def parse_document(self, document, key):
        print("Parsing the JSON...")
        self.papers = [self.get_paper_from_entry(entry) for entry in document[key]]

    def get_paper_from_entry(self, entry) -> dict:
        # Paper object
        paper = Paper()

        # Platform specific
        paper.set("id", entry["paperId"])
        paper.set("source", "Semantic Scholar")

        # Title
        title = entry["title"]
        paper.set("title", title)

        # Abstract
        abstract = entry["abstract"]
        paper.set("abstract", abstract)

        # Authors
        authors = []
        for item in entry["authors"]:
            uri = f"ss:{item['authorId']}"
            authors.append(uri)
        paper.set("authors", authors)

        # Published at
        if entry["publicationDate"]:
            published_at = datetime.strptime(str(entry["publicationDate"]), "%Y-%m-%d")
        elif entry["year"]:
            published_at = datetime.strptime(str(entry["year"]), "%Y")
        else:
            published_at = datetime(1970, 1, 1)
        paper.set("published_at", published_at)

        # Reference count
        reference_count = entry["referenceCount"]
        paper.set("reference_count", reference_count)

        # Citation count
        citation_count = entry["citationCount"]
        paper.set("citation_count", citation_count)

        # PDF URL
        if entry["openAccessPdf"]:
            pdf_url = entry["openAccessPdf"]["url"]
            paper.set("pdf_url", pdf_url)

        # Source URL
        source_url = entry["url"]
        paper.set("source_url", source_url)

        # Related database object
        paper.set("related", self.related)

        # Cited by URL
        cited_by_url = f"https://api.semanticscholar.org/graph/v1/paper/{entry['paperId']}/citations"
        paper.set("cited_by_url", cited_by_url)

        # References
        # PASS

        # Related URL
        # PASS

        # Return the paper
        return paper.dump()

    def get_paper_from_entry_sorted(self, entry, index=0) -> dict:
        # Paper object
        paper = Paper()

        # Platform specific
        paper.set("id", entry["paperId"])
        paper.set("source", "Semantic Scholar")

        # Title
        title = entry["title"]
        paper.set("title", title)

        # Abstract
        abstract = entry["abstract"]
        paper.set("abstract", abstract)

        # Authors
        authors = []
        for item in entry["authors"]:
            uri = f"ss:{item['authorId']}"
            authors.append(uri)
        paper.set("authors", authors)

        # References
        # PASS

        # Published at
        if entry["publicationDate"]:
            published_at = datetime.strptime(str(entry["publicationDate"]), "%Y-%m-%d")
        elif entry["year"]:
            published_at = datetime.strptime(str(entry["year"]), "%Y")
        else:
            published_at = datetime(1970, 1, 1)
        paper.set("published_at", published_at)

        # Reference count
        reference_count = entry["referenceCount"]
        paper.set("reference_count", reference_count)

        # Citation count
        citation_count = entry["citationCount"]
        paper.set("citation_count", citation_count)

        # PDF URL
        if entry["openAccessPdf"]:
            pdf_url = entry["openAccessPdf"]["url"]
            paper.set("pdf_url", pdf_url)

        # Source URL
        source_url = entry["url"]
        paper.set("source_url", source_url)

        # Related URL
        paper.set("related_url", index)

        # Related database object
        paper.set("related", self.related)

        # Cited by URL
        cited_by_url = f"https://api.semanticscholar.org/graph/v1/paper/{entry['paperId']}/citations"
        paper.set("cited_by_url", cited_by_url)

        # Return the paper
        return paper.dump()