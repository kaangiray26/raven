import os
from numpy import ndarray
from typing import Any
from lib.waiter import get_config
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self):
        # Set the configuration
        self.config = get_config()
        self.load_specter2()
        self.load_embeddings_model()

    def load_specter2(self):
        print("Loading the specter2 model...")
        if not os.path.exists("specter2_base"):
            self.specter2 = SentenceTransformer("allenai/specter2_base",
                device=self.config["embeddings"]["device"],
                backend="onnx"
            )
            self.specter2.save_pretrained("specter2_base")
            return

        # Load from the path
        self.specter2 = SentenceTransformer("specter2_base",
            device=self.config["embeddings"]["device"],
            backend="onnx"
        )

    def load_embeddings_model(self):
        if self.config["embeddings"]["prefer"] == "huggingface":
            print("Loading the huggingface model...")
            self.embeddings_model = SentenceTransformer(self.config["embeddings"]["huggingface_model"],
                device=self.config["embeddings"]["device"],
                backend="onnx",
                model_kwargs={"file_name": "onnx/model_O3.onnx"},
            )
            return

        if self.config["embeddings"]["prefer"] == "openai":
            print("Loading the openai model...")
            self.embeddings_model = OpenAIEmbeddings(
                api_key=self.config["api_keys"]["openai"],
                model=self.config["embeddings"]["openai_model"],
                dimensions=self.config["embeddings"]["dimensions"]
            )
            return

        raise ValueError("Invalid [embeddings][pref], please check your config.json file.")

    def get_embeddings_from_texts(self, texts: list) -> list:
        # Check the embeddings model
        if self.config["embeddings"]["prefer"] == "huggingface":
            embeddings = self.embeddings_model.encode(texts,
                normalize_embeddings=True,
            )
            return [embedding for embedding in embeddings]
        return self.embeddings_model.embed_documents(texts)

    # Create embeddings from a text
    def get_embeddings_from_text(self, text) -> ndarray[Any, Any]:
        # Check the embeddings model
        if self.config["embeddings"]["prefer"] == "huggingface":
            embeddings = self.embeddings_model.encode([text],
                normalize_embeddings=True,
            )
            return embeddings[0]
        return self.embeddings_model.embed_query(text)

    def get_specter_embeddings(self, texts: list):
        return self.specter2.encode(texts,
            normalize_embeddings=True
        )