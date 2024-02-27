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
from tqdm import tqdm
import os

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

repos = [
    "opentelemetry-java", "opentelemetry-specification", 
    "opentelemetry-dotnet", "opentelemetry-python",
    "opentelemetry-erlang", "opentelemetry-collector", 
    "opentelemetry-js", "opentelemetry-go", 
    "opentelemetry-proto", "opentelemetry-ruby",
    "opentelemetry-cpp", "opentelemetry-collector-contrib",
    "opentelemetry-java-instrumentation", "opentelemetry-php",
    "opentelemetry-python-contrib", "opentelemetry-erlang-api", 
    "opentelemetry-rust", "opentelemetry-operator",
    "opentelemetry-swift", "opentelemetry-go-contrib", 
    "opentelemetry-js-contrib", "opentelemetry-dotnet-contrib",  
    "opentelemetry-dotnet-instrumentation", "opentelemetry-java-contrib", 
    "opentelemetry-helm-charts", "opentelemetry-cpp-contrib", 
    "opentelemetry-proto-go", "opentelemetry-lambda", 
    "opentelemetry-js-api", "opentelemetry-log-collection",
    "opentelemetry-collector-builder", "opentelemetry-erlang-contrib",
    "opentelemetry-php-contrib", "opentelemetry-network",
    "opentelemetry-collector-releases", "opentelemetry-proto-java",
    "opamp-go", "opamp-spec", "opentelemetry-ruby-contrib", 
    "opentelemetry-demo", 
    "opentelemetry-go-instrumentation", "opentelemetry-php-instrumentation",
    "otel-arrow-collector", "opamp-java", "opentelemetry-profiling",
    "opentelemetry-configuration", "semantic-conventions",
    "opentelemetry-android", "opentelemetry-rust-contrib"
]

def get_releases(repo, count):
    """Get the last `count` releases from the repository at `url`."""
    token = os.getenv('GH_TOKEN')
    if not token:
        print("GH_TOKEN environment variable not set")
        return
    try:
        api_url = f"https://api.github.com/repos/open-telemetry/{repo}/releases"
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(api_url, headers=headers)
        releases = response.json()
        releases.sort(key=lambda x: x['published_at'])
        releases = releases[-count:]
        release = json.dumps(releases[0])
        return json.loads(release)
    except Exception:
        print(f"Error getting releases for {repo}")
        return None

def process_release(rag, release, name):
    """Process a single release and return a summary."""
    retriever = rag.db.as_retriever()
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
    summary = chain.invoke(release['body'])
    new_json_object = {
        "repo_name": name,
        "release_url": release['html_url'],
        "release_version": release['tag_name'],
        "summary": summary
    }
    return(json.dumps(new_json_object))

def main(args):
    output = []
    for language in tqdm(docs_map, desc="Processing documents"):
        processor = DocumentProcessor(docs_map[language])
        processor.process_documents()
    if args.repo and args.num_releases:
        try:
            release = get_releases(args.repo, args.num_releases)
            output.append(process_release(processor, release, args.repo))
        except Exception as e:
            print(f"Error processing release for repo {args.repo}: {e}")
    else:
        for repo in tqdm(repos, desc="Processing repositories"):
            try:
                release = get_releases(repo, 1)
                output.append(process_release(processor, release, repo))
            except Exception as e:
                print(f"Error processing release for repo {repo}: {e}")
    with open('out.json', 'w') as f:
        json.dump(output, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=False, help="name of the repository")
    parser.add_argument("--num_releases", type=int, required=False, help="Number of releases to summarize")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
