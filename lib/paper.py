class Paper:
    def __init__(self):
        self.data = {
            # Platform specific
            "id":"",
            "source":"",
            # Basic
            "title": None,
            "abstract": None,
            "authors": [],
            "refs": [],
            "published_at": None,
            # Counts
            "reference_count": 1,
            "citation_count": 1,
            # URLs
            "pdf_url": None,
            "source_url": None,
            "related_url":None,
            "cited_by_url": None,
            "related": None
        }

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def dump(self):
        return self.data

    def to_tuple(self):
        return (self.data[key] for key in self.data.keys())

class Author:
    def __init__(self, uri=None, name=None):
        self.data = {
            "uri": uri,
            "name": name
        }
    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def dump(self):
        return self.data

    def to_tuple(self):
        keys = ["uri", "name"]
        return [self.data[key] for key in keys]

class Reference:
    def __init__(self, uri=None, source=None):
        self.data = {
            "uri": uri,
            "source": source
        }
    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def dump(self):
        return self.data

    def to_tuple(self):
        keys = ["uri", "source"]
        return [self.data[key] for key in keys]