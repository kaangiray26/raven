from urllib.parse import urlparse
from lib.crawlers.dblp import DBLP
from lib.crawlers.acm import ACM_DL
from lib.crawlers.arxiv import arXiv
from lib.crawlers.semantic import Semantic_Scholar

domains= [
    "www.semanticscholar.org",
    "arxiv.org",
    "dblp.org",
    "dl.acm.org"
]

def parse_source_url(url):
    u = urlparse(url)
    if u.netloc not in domains:
        return None

    # Sanitize the URL
    match u.netloc:
        case "www.semanticscholar.org":
            id = u.path.split("/")[-1]
            return f"https://www.semanticscholar.org/paper/{id}"

        case "arxiv.org":
            id = u.path.split("/")[-1]
            return f"https://arxiv.org/abs/{id}"

        case "dblp.org":
            id = u.path.rstrip(".html")
            return f"https://dblp.org{id}"

        case "dl.acm.org":
            id = u.path
            return f"https://dl.acm.org{id}"

        case _:
            return None

def get_crawler_from_url(url):
    # Check the domain of the url
    u = urlparse(url)
    if u.netloc not in domains:
        return None

    # Get the crawler
    match u.netloc:
        case "www.semanticscholar.org":
            return Semantic_Scholar()

        case "arxiv.org":
            return arXiv()

        case "dblp.org":
            return DBLP()

        case "dl.acm.org":
            return ACM_DL()

        case _:
            return None

def get_crawler_from_domain(domain):
    if domain not in domains:
        return None

    # Get the crawler
    match domain:
        case "www.semanticscholar.org":
            return Semantic_Scholar()

        case "arxiv.org":
            return arXiv()

        case "dblp.org":
            return DBLP()

        case "dl.acm.org":
            return ACM_DL()

        case _:
            return None

class Crawler:
    def __init__(self, related=None, limit=10):
        self.papers =[]
        self.limit = limit
        self.related = related
        self.headers = {
            "Content-Type": "application/json",
        }

    def crawl(self, topic):
        ...

    def crawl_paper_by_url(self, url):
        # Get specific crawler from URL
        crawler = get_crawler_from_url(url)
        print("Got crawler:", crawler)
        if not crawler:
            return

        # Get single paper
        self.papers = [crawler.crawl_paper_by_url(url)]

    def crawl_paper_by_id(self, domain, id):
        # Get specific crawler from domain
        crawler = get_crawler_from_domain(domain)
        if not crawler:
            return

        # Get single paper
        self.papers = [crawler.crawl_paper_by_id(id)]