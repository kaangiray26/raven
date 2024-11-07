import requests
from lxml import etree
from datetime import datetime
from lib.paper import Paper

class DBLP:
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
            "title": ".//title/text()",
            "abstract": ".//summary/text()",
            "authors": ".//author",
            "published_at": ".//year/text()",
            "source_url": ".//link[@type='text/html']/@href",
            "citation_count": None,
            "pdf_url": ".//ee/text()",
            "related_url": None,
            "cited_by_url": None
        }

    def crawl_paper_by_url(self, url) -> dict:
        # Get the paper ID
        paper_id = url
        return self.crawl_paper_by_id(paper_id)

    def crawl_paper_by_id(self, id) -> dict:
        print(f"Crawling DBLP for {id}...")

        # Make the request
        with requests.get(
            f"{id}.xml",
        ) as r:
            if r.status_code != 200:
                raise Exception("Failed to fetch the page!")

            # Parse the XML
            root = etree.fromstring(r.text.encode())
            entries = root.xpath("/dblp/article")
            if not len(entries):
                raise Exception("Entry does not exist!")

            return self.get_paper_from_entry(entries[0])

        return {}

    def get_paper_from_entry(self, entry) -> dict:
        # Paper object
        paper = Paper()

        # Platform specific
        # Get the value of the key "key" from the entry element
        data = entry.xpath("@key")
        paper.set("id", data[0])
        paper.set("source", "DBLP")

        # Title
        data = entry.xpath(".//title/text()")
        title = " ".join(data[0].split())
        paper.set("title", title)

        # Abstract
        # PASS

        # Authors
        authors = []
        for author in entry.xpath(".//author"):
            uri = f"dp:{author.xpath(".//text()")[0]}"
            authors.append(uri)
        paper.set("authors", authors)

        # Published at
        data = entry.xpath(".//year/text()")
        published_at = datetime.strptime(data[0], "%Y")
        paper.set("published_at", published_at)

        # Reference count
        # PASS

        # Citation count
        # PASS

        # PDF URL
        data = entry.xpath(".//ee/text()")
        pdf_url = data[0]
        paper.set("pdf_url", pdf_url)

        # Source URL
        paper.set("source_url", f"https://dblp.org/rec/{paper.get('id')}")

        # Related database object
        paper.set("related", self.related)

        # Cited by URL
        # PASS

        # References
        # PASS

        # Related URL
        # PASS

        # Return the paper
        return paper.dump()