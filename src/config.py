from dataclasses import dataclass
from pathlib import Path

from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv
import openai

load_dotenv()


@dataclass
class Paths:
    root: Path = Path(__file__).parent
    data: Path = root / "data"
    book: Path = (
        data
        / "Marcus_Aurelius_Antoninus_-_His_Meditations_concerning_himselfe/index.html"
    )

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = "https://cog-vlnygtsnpw4pe.openai.azure.com/"
openai.api_key = "02964635d4a1475c80a6c326736df0b8"

embedding = OpenAIEmbeddings(deployment="embedding", model="text-embedding-ada-002", chunk_size=1)
