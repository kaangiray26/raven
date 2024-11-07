import requests
from lxml import html
from datetime import datetime
from lib.paper import Paper

class ACM_DL:
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

    def crawl_paper_by_url(self, url) -> dict:
        # Get the paper ID
        paper_id = url
        return self.crawl_paper_by_id(paper_id)

    def crawl_paper_by_id(self, id) -> dict:
        print(f"Crawling ACM DL for {id}...")

        # Make the request
        with requests.get(id) as r:
            if r.status_code != 200:
                raise Exception("Failed to fetch the page!")

            # Parse the HTML
            root = html.fromstring(r.text.encode())
            entries = root.xpath("//article")
            if not len(entries):
                raise Exception("Entry does not exist!")

            return self.get_paper_from_entry(entries[0])

    def get_paper_from_entry(self, entry) -> dict:
        # Paper object
        paper = Paper()

        # Platform specific
        data = entry.xpath(".//div[@class='doi']/a/@href")
        paper.set("id", data[0])
        paper.set("source", "ACM DL")

        # Title
        data = entry.xpath(".//div[@property='name']/text()")
        title = " ".join(data[0].split())
        paper.set("title", title)

        # Abstract
        data = entry.xpath(".//section[@id='abstract']/div/text()")
        abstract = " ".join(data[0].split())
        paper.set("abstract", abstract)

        # Authors
        authors = []
        for author in entry.xpath(".//span[@property='author']"):
            uri = f"acmdl:{author.xpath(".//text()")[0]}"
            authors.append(uri)
        paper.set("authors", authors)

        # Published at
        data = entry.xpath(".//span[@class='core-date-published']/text()")
        published_at = datetime.strptime(data[0], "%d %B %Y")
        paper.set("published_at", published_at)

        # Reference count
        # PASS

        # Citation count
        # PASS

        # PDF URL
        data = entry.xpath(".//a[@class='btn btn--pdf red']/@href")
        if len(data):
            pdf_url = data[0]
            paper.set("pdf_url", pdf_url)

        # Source URL
        data = paper.get("id").split("https://doi.org/")[-1]
        paper.set("source_url", f"https://dl.acm.org/doi/{data}")

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