from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch
import openai


class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

def main():
    texts = ["Mountains are majestic natural wonders that captivate the imagination with their towering peaks, rugged terrain, and breathtaking beauty. These geological giants, formed over millions of years, have played a significant role in shaping the planet's landscape and have become a source of inspiration for countless adventurers, poets, and nature enthusiasts. As one approaches the base of a mountain, a sense of awe and reverence fills the air. The crisp, cool breeze carries the faint scent of pine, while the sound of gushing streams and cascading waterfalls echoes through the valleys. The verdant slopes are adorned with a vibrant tapestry of flora, ranging from delicate alpine flowers to hardy evergreen trees that cling tenaciously to the rugged slopes. Ascending the mountain requires determination and perseverance. The path is strewn with rocks and boulders, and the air grows thinner with every step. Yet, the rewards that await those who brave the ascent are unparalleled. Each new elevation offers a fresh perspective, revealing a panorama of awe-inspiring vistas that stretch as far as the eye can see. The jagged peaks rise majestically into the heavens, seemingly touching the clouds, while the valleys unfold like a green carpet beneath. Explorers and mountaineers from around the world are drawn to these lofty summits, driven by the desire to conquer the unconquerable. Scaling these formidable heights demands courage, physical strength, and meticulous planning. The climb is often a battle against one's own limitations, where determination and endurance become the guiding forces. Yet, with each milestone reached, a sense of triumph surges through the veins, fueling the spirit to push further, higher. Beyond the physical challenges, mountains offer a sanctuary for solitude and introspection. Standing atop a peak, the world below seems distant, and the worries of everyday life fade into insignificance. Here, amid the grandeur of nature, one can find solace and peace, a rare opportunity to connect with the primordial forces that have shaped the Earth. Mountains are not only remarkable for their physical grandeur but also for the diverse ecosystems they sustain. These natural havens are home to a multitude of creatures, each adapted to the unique demands of their high-altitude habitat. Mountain goats gracefully traverse the rocky slopes, while elusive snow leopards prowl through the silent snowfields. Rare and endemic plant species cling to life in the harshest of environments, their resilience a testament to the wonders of adaptation. As the sun sets behind the mountain range, casting a warm glow across the landscape, a serene stillness envelops the peaks. The tranquil beauty of the mountains is a reminder of the delicate balance of nature and the power it holds. These majestic giants stand as guardians of our planet, reminding us of the importance of preserving and cherishing the natural world. In conclusion, mountains possess an allure that is hard to resist. They beckon us to explore, challenge our limits, and reconnect with nature. From their majestic peaks to their hidden valleys, mountains hold secrets and wonders that continue to inspire and fascinate. They remind us of the immense beauty and power of our planet, inviting us to embark on a journey of discovery, both within ourselves and in the vastness of the natural world.", "Text 2", "Text 3"]
    metadatas = [
        {"author": "Author 1", "date": "2022-01-01"},
        {"author": "Author 2", "date": "2022-01-02"},
        {"author": "Author 3", "date": "2022-01-03"}
    ]

    data = []
    for text, metadata in zip(texts, metadatas):
        document = Document(page_content=text, metadata=metadata)
        data.append(document)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    documents = text_splitter.split_documents(data)

    openai.api_type = "azure"
    openai.api_version = "2022-12-01"
    openai.api_base = "https://cog-vlnygtsnpw4pe.openai.azure.com/"
    openai.api_key = "02964635d4a1475c80a6c326736df0b8"

    embedding = OpenAIEmbeddings(deployment="embedding", model="text-embedding-ada-002", chunk_size=1)

    db = ElasticVectorSearch.from_documents(
        documents,
        embedding,
        elasticsearch_url="https://elastic:=OAxbvmXbY0oRydGrYWG@elasticsearch.itxbpm.com:9200/",
        index_name="chat"
    )
    print(db.client.info())


if __name__ == "__main__":
    main()
