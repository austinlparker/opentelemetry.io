import argparse
import requests
import json
from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

class DocumentProcessor:
    def __init__(self, concepts_path):
        self.concepts_path = concepts_path
        self.loader = DirectoryLoader(self.concepts_path, glob="**/*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader, silent_errors=True)
        self.docs = self.loader.load()
        self.splits = [
            ('#', "Header 1"),
            ('##', "Header 2"),
            ('###', "Header 3")
        ]
        self.store = LocalFileStore("./cache/")
        self.db = None
        self.embedder = CacheBackedEmbeddings.from_bytes_store(
            OpenAIEmbeddings(),
            self.store,
            namespace=OpenAIEmbeddings().model
        )
        

    def process_documents(self):
        for doc in self.docs:
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.splits)
            md_header_split = markdown_splitter.split_text(doc.page_content)
            try:
                self.db = Chroma.from_documents(md_header_split, self.embedder)
            except ValueError:
                print(f"Error processing {doc}")

docs_map = {
    "concepts": "opentelemetry.io/content/en/docs/concepts/"
}

def get_releases(repo, count):
    """Get the last `count` releases from the repository at `url`."""
    api_url = f"https://api.github.com/repos/open-telemetry/{repo}/releases"
    response = requests.get(api_url)
    releases = response.json()
    releases.sort(key=lambda x: x['published_at'])
    releases = releases[-count:]
    release = json.dumps(releases[0])
    return json.loads(release)

def main(args):
    for language in docs_map:
        processor = DocumentProcessor(docs_map[language])
        processor.process_documents()
    releases = get_releases(args.repo, args.num_releases)
    
    retriever = processor.db.as_retriever()
    template = """You are a technical writer for an open source project.
    Your role is to compose a short, two sentence summary from release notes.
    Make sure to highlight breaking changes. You will be given additional
    context to help you understand what terms in the release notes refer to.
    CONTEXT: {context}
    NOTES: {notes}
    SUMMARY:
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4-turbo-preview")
    chain = (
        {"context": retriever, "notes": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    print(chain.invoke(releases['body']))

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="name of the repository")
    parser.add_argument("--num_releases", type=int, required=True, help="Number of releases to summarize")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
