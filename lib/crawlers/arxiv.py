import requests
from lxml import etree
from datetime import datetime
from lib.paper import Paper

class arXiv:
    def __init__(self, related=None, limit=10):
        self.papers = []
        self.limit = limit
        self.related = related
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:129.0) Gecko/20100101 Firefox/129.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br, zstd",
        }

        # Additional attributes
        self.xpaths = {
            "title": ".//atom:title/text()",
            "abstract": ".//atom:summary/text()",
            "authors": ".//atom:author",
            "published_at": ".//atom:published/text()",
            "source_url": ".//atom:link[@type='text/html']/@href",
            "citation_count": None,
            "pdf_url": ".//atom:link[@type='application/pdf']/@href",
            "related_url": None,
            "cited_by_url": None
        }
        self.namespaces = {
            "atom": "http://www.w3.org/2005/Atom"
        }

    def crawl_paper_by_url(self, url) -> dict:
        # Get the paper ID
        paper_id = url.split("/")[-1]
        return self.crawl_paper_by_id(paper_id)

    def crawl_paper_by_id(self, id) -> dict:
        print(f"Crawling arXiv for {id}...")

        # Make the request
        with requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "id_list": id
            }
        ) as r:
            if r.status_code != 200:
                raise Exception("Failed to fetch the page!")

            # Parse the ATOM
            root = etree.fromstring(r.text.encode())
            entries = root.xpath("//atom:entry", namespaces=self.namespaces)
            if not len(entries):
                raise Exception("Entry does not exist!")

            return self.get_paper_from_entry(entries[0])

    def get_paper_from_entry(self, entry) -> dict:
        # Paper object
        paper = Paper()

        # Platform specific
        data = entry.xpath(".//atom:id/text()", namespaces=self.namespaces)
        paper.set("id", data[0])
        paper.set("source", "arXiv")

        # Title
        data = entry.xpath(self.xpaths["title"], namespaces=self.namespaces)
        title = " ".join(data[0].split())
        paper.set("title", title)

        # Abstract
        data = entry.xpath(self.xpaths["abstract"], namespaces=self.namespaces)
        abstract = " ".join(data[0].split())
        paper.set("abstract", abstract)

        # Authors
        authors = []
        for author in entry.xpath(self.xpaths["authors"], namespaces=self.namespaces):
            uri = f"ax:{author.xpath(".//atom:name/text()", namespaces=self.namespaces)[0]}"
            authors.append(uri)
        paper.set("authors", authors)

        # Published at
        data = entry.xpath(self.xpaths["published_at"], namespaces=self.namespaces)
        published_at = datetime.strptime(data[0], "%Y-%m-%dT%H:%M:%SZ")
        paper.set("published_at", published_at)

        # Reference count
        # PASS

        # Citation count
        # PASS

        # PDF URL
        data = entry.xpath(self.xpaths["pdf_url"], namespaces=self.namespaces)
        pdf_url = data[0]
        paper.set("pdf_url", pdf_url)

        # Source URL
        data = entry.xpath(self.xpaths["source_url"], namespaces=self.namespaces)
        source_url = data[0]
        paper.set("source_url", source_url)

        # Related database object
        paper.set("related", self.related)

        # Cited by URL
        cited_by_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper.get("id")}/citations"
        paper.set("cited_by_url", cited_by_url)

        # References
        # PASS

        # Related URL
        # PASS

        # Return the paper
        return paper.dump()