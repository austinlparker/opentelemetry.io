import argparse
import requests
import json
from langchain.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
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
            self.db = Chroma.from_documents(md_header_split, self.embedder)


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
    concepts_path = "../../content/en/docs/concepts/"
    processor = DocumentProcessor(concepts_path)
    processor.process_documents()
    releases = get_releases(args.repo, args.num_releases)
    print(releases['body'])
    
    retriever = processor.db.as_retriever()
    template = """You are a helpful assistant. Your role is to summarize release
               notes from repositories in the OpenTelemetry organization. The
               goal of these summaries is to make users aware of breaking
               changes and draw attention to new features. You will be given
               additional context to help you understand OpenTelemetry concepts.

               CONTEXT: {context}

               An example of notes and a summary is shown below.

               NOTES: This release targets the OpenTelemetry SDK 1.34.1.

Note that many artifacts have the -alpha suffix attached to their version number, reflecting that they are still alpha quality and will continue to have breaking changes. Please see the VERSIONING.md for more details.

The 2.0.0 release contains significant breaking changes that will most likely affect all users, please be sure to read the breaking changes below carefully.

Note: 1.32.x will be security patched for at least 6 months in case some of the changes below are too disruptive to adopt right away.

⚠️⚠️ Breaking changes ⚠️⚠️
The default OTLP protocol has been changed from grpc to http/protobuf in order to align with the specification. You can switch to the grpc protocol using OTEL_EXPORTER_OTLP_PROTOCOL=grpc or -Dotel.exporter.otlp.protocol=grpc.
Micrometer metric bridge has been disabled by default. You can enable it using OTEL_INSTRUMENTATION_MICROMETER_ENABLED=true or -Dotel.instrumentation.micrometer.enabled=true.
The OTLP logs exporter is now enabled by default. You can disable it using OTEL_LOGS_EXPORTER=none or -Dotel.logs.exporter=none.
Controller spans are now disabled by default. You can enable them using OTEL_INSTRUMENTATION_COMMON_EXPERIMENTAL_CONTROLLER_TELEMETRY_ENABLED=true or -Dotel.instrumentation.common.experimental.controller-telemetry.enabled=true.
View spans are now disabled by default. You can enable them using OTEL_INSTRUMENTATION_COMMON_EXPERIMENTAL_VIEW_TELEMETRY_ENABLED=true or -Dotel.instrumentation.common.experimental.view-telemetry.enabled=true.
⚠️⚠️ Stable HTTP semantic conventions are now emitted ⚠️⚠️ - TOO MANY CHANGES TO LIST HERE, be sure to review the full list of changes.
Stable JVM semantic conventions are now emitted. - Memory metrics - process.runtime.jvm.memory.usage renamed to jvm.memory.used - process.runtime.jvm.memory.committed renamed to jvm.memory.committed - process.runtime.jvm.memory.limit renamed to jvm.memory.limit - process.runtime.jvm.memory.usage_after_last_gc renamed to jvm.memory.used_after_last_gc - process.runtime.jvm.memory.init renamed to jvm.memory.init (still experimental) - Metric attributes - type renamed to jvm.memory.type - pool renamed to jvm.memory.pool.name - Garbage collection metrics - process.runtime.jvm.gc.duration renamed to jvm.gc.duration - Metric attributes - name renamed to jvm.gc.name - action renamed to jvm.gc.action - Thread metrics - process.runtime.jvm.threads.count renamed to jvm.threads.count - Metric attributes - daemon renamed to jvm.thread.daemon - Classes metrics - process.runtime.jvm.classes.loaded renamed to jvm.classes.loaded - process.runtime.jvm.classes.unloaded renamed to jvm.classes.unloaded - process.runtime.jvm.classes.current_loaded renamed to jvm.classes.count - CPU metrics - process.runtime.jvm.cpu.utilization renamed to jvm.cpu.recent_utilization - process.runtime.jvm.system.cpu.load_1m renamed to jvm.system.cpu.load_1m (still experimental) - process.runtime.jvm.system.cpu.utilization renamed to jvm.system.cpu.utilization (still experimental) - Buffer metrics - process.runtime.jvm.buffer.limit renamed to jvm.buffer.memory.limit (still experimental) - process.runtime.jvm.buffer.count renamed to jvm.buffer.count (still experimental) - process.runtime.jvm.buffer.usage renamed to jvm.buffer.memory.usage (still experimental) - Metric attributes - pool renamed to jvm.buffer.pool.name
More migration notes
Lettuce CONNECT spans are now disabled by default. You can enable them using OTEL_INSTRUMENTATION_LETTUCE_CONNECTION_TELEMETRY_ENABLED=true or -Dotel.instrumentation.lettuce.connection-telemetry.enabled=true.
The configuration property otel.instrumentation.log4j-appender.experimental.capture-context-data-attributes has been renamed to otel.instrumentation.log4j-appender.experimental.capture-mdc-attributes.
MDC attribute prefixes (log4j.mdc. and logback.mdc.*) have been removed.
The artifact instrumentation-api-semconv has been renamed to instrumentation-api-incubator.
HTTP classes have been moved from instrumentation-api-incubator to instrumentation-api and as a result are now stable.
🌟 New javaagent instrumentation
Vert.x redis client (#9838)
📈 Enhancements
Reduce reactor stack trace depth (#9923)
Implement error.type in spring-webflux and reactor-netty instrumentations (#9967)
Bridge metric advice in OpenTelemetry API 1.32 (#10026)
Capture http.route for akka-http (#10039)
Rename telemetry.auto.version to telemetry.distro.version and add telemetry.distro.name (#9065)
Implement forEach support for aws sqs tracing list (#10062)
Add http client response attributes to aws sqs process spans (#10074)
Add support for OTEL_RESOURCE_ATTRIBUTES, OTEL_SERVICE_NAME, OTEL_EXPORTER_OTLP_HEADERS, and OTEL_EXPORTER_OTLP_PROTOCOL for spring boot starter (#9950)
Add elasticsearch-api-client as instrumentation name to elasticsearch-api-client-7.16 (#10102)
Add instrumentation for druid connection pool (#9935)
Remove deprecated rocketmq setting (#10125)
JMX metrics for Tomcat with 'Tomcat' JMX domain (#10115)
Capture the SNS topic ARN under the 'messaging.destination.name' span attribute. (#10096)
Add network attributes to rabbitmq process spans (#10210)
Add UserExcludedClassloadersConfigurer (#10134)
Apply both server attributes & network attributes to Lettuce 5.1 (#10197)
🛠️ Bug fixes
Fix aws propagator presence check in spring boot starter (#9924)
Capture authority from apache httpclient request when HttpHost is null (#9990)
Fix NoSuchBeanDefinitionException with the JDBC driver configuration in spring boot starter (#9978)
Null check for nullable response object in aws sdk 1.1 instrumentation (#10029)
Fix using opentelemetry-spring-boot with Java 8 and Gradle (#10066)
Fix transforming Java record types (#10052)
Fix warnings from the spring boot starter (#10086)
Resolve ParameterNameDiscoverer Bean Conflict in spring-boot-autoconfigure (#10105)
🙇 Thank you
This release was possible thanks to the following contributors who shared their brilliant ideas and awesome pull requests:

@AlchemyDing
@anhermon
@anuraaga
@bcarter97
@breedx-splk
@happyuser23
@heyams
@jack-berg
@jaydeluca
@jeanbisutti
@JonasKunz
@kenfinnigan
@knbk
@laurit
@mateuszrzeszutek
@moznion
@nilsga
@PaurushGarg
@PeterF778
@rBrda
@SHaaD94
@stevesea
@SylvainJuge
@tduncan
@theletterf
@trask
@TylerHelmuth
@vallabhnatu
@xiongchun
@zeitlinger

               SUMMARY: OpenTelemetry Java Instrumentation 2.0 has been released
               with the following breaking changes:
                 - The default OTLP export protocol is now http/protobuf, per specification.
                 - The Micrometer metric bridge is now disabled by default.
                 - OTLP logs are now emitted by default.
                 - Certain types of spans are now disabled by default.
                 - Stable HTTP and JVM semantic conventions are now emitted.
               In addition, there have been several bug fixes and enhancements.

    NOTES: {notes}
    SUMMARY:
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4-1106-preview")
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
