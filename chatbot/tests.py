from django.test import TestCase

# Create your tests here.
You are an expert programmer and problem-solver, tasked with answering any question about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the given question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${number}] notation. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end. If different results refer to different entities within the same name, write separate answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge bank, not part of the conversation with the user. 

<context>
    <doc id='0'>MultiVector Retriever | 🦜️🔗 Langchain

[Skip to main content](#__docusaurus_skipToContent_fallback)# MultiVector Retriever

It can often be beneficial to store multiple vectors per document. There
are multiple use cases where this is beneficial. LangChain has a base
`MultiVectorRetriever` which makes querying this type of setup easy. A
lot of the complexity lies in how to create the multiple vectors per
document. This notebook covers some of the common ways to create those
vectors and use the `MultiVectorRetriever`.

The methods to create multiple vectors per document include:

- Smaller chunks: split a document into smaller chunks, and embed
those (this is ParentDocumentRetriever).

- Summary: create a summary for each document, embed that along with
(or instead of) the document.

- Hypothetical questions: create hypothetical questions that each
document would be appropriate to answer, embed those along with (or
instead of) the document.

Note that this also enables another method of adding embeddings -
manually. This is great because you can explicitly add questions or
queries that should lead to a document being recovered, giving you more
control.

```python
from langchain.retrievers.multi_vector import MultiVectorRetriever
```

```python
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
```

```python
loaders = [
    TextLoader("../../paul_graham_essay.txt"),
    TextLoader("../../state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)
```

## Smaller chunks​

Often times it can be useful to retrieve larger chunks of information,
but embed smaller chunks. This allows for embeddings to capture the
semantic meaning as closely as possible, but for as much context as
possible to be passed downstream. Note that this is what the
`ParentDocumentRetriever` does. Here we show what is going on under the
hood.

```python
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
import uuid

doc_ids = [str(uuid.uuid4()) for _ in docs]
```

```python
# The splitter to use to create smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
```

```python
sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)
```

```python
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
```

```python
# Vectorstore alone retrieves the small chunks
retriever.vectorstore.similarity_search("justice breyer")[0]
```

```text
Document(page_content='Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. \n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.', metadata={'doc_id': '2fd77862-9ed5-4fad-bf76-e487b747b333', 'source': '../../state_of_the_union.txt'})
```

```python
# Retriever returns larger chunks
len(retriever.get_relevant_documents("justice breyer")[0].page_content)
```

```text
9875
```</doc>
<doc id='1'>Vector store-backed retriever | 🦜️🔗 Langchain

[Skip to main content](#__docusaurus_skipToContent_fallback)# Vector store-backed retriever

A vector store retriever is a retriever that uses a vector store to
retrieve documents. It is a lightweight wrapper around the vector store
class to make it conform to the retriever interface. It uses the search
methods implemented by a vector store, like similarity search and MMR,
to query the texts in the vector store.

Once you construct a vector store, it’s very easy to construct a
retriever. Let’s walk through an example.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../state_of_the_union.txt")
```

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
```

```python
retriever = db.as_retriever()
```

```python
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```

## Maximum marginal relevance retrieval​

By default, the vector store retriever uses similarity search. If the
underlying vector store supports maximum marginal relevance search, you
can specify that as the search type.

```python
retriever = db.as_retriever(search_type="mmr")
```

```python
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```

## Similarity score threshold retrieval​

You can also set a retrieval method that sets a similarity score
threshold and only returns documents with a score above that threshold.

```python
retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)
```

```python
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```

## Specifying top k​

You can also specify search kwargs like `k` to use when doing retrieval.

```python
retriever = db.as_retriever(search_kwargs={"k": 1})
```

```python
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
len(docs)
```

```text
1
```

- [Maximum marginal relevance retrieval](#maximum-marginal-relevance-retrieval)

- [Similarity score threshold retrieval](#similarity-score-threshold-retrieval)

- [Specifying top k](#specifying-top-k)</doc>
<doc id='2'>langchain.retrievers.multi_vector.MultiVectorRetriever — 🦜🔗 LangChain 0.1.7

LangChain

Core

Community

Experimental

anthropic

exa

google-genai

google-vertexai

mistralai

nomic

nvidia-ai-endpoints

nvidia-trt

openai

pinecone

robocorp

together

Partner libs

anthropic
exa
google-genai
google-vertexai
mistralai
nomic
nvidia-ai-endpoints
nvidia-trt
openai
pinecone
robocorp
together

Docs

Toggle Menu

Prev
Up
Next

langchain.retrievers.multi_vector.MultiVectorRetriever
MultiVectorRetriever
MultiVectorRetriever.byte_store
MultiVectorRetriever.docstore
MultiVectorRetriever.id_key
MultiVectorRetriever.metadata
MultiVectorRetriever.search_kwargs
MultiVectorRetriever.search_type
MultiVectorRetriever.tags
MultiVectorRetriever.vectorstore
MultiVectorRetriever.abatch()
MultiVectorRetriever.aget_relevant_documents()
MultiVectorRetriever.ainvoke()
MultiVectorRetriever.assign()
MultiVectorRetriever.astream()
MultiVectorRetriever.astream_events()
MultiVectorRetriever.astream_log()
MultiVectorRetriever.atransform()
MultiVectorRetriever.batch()
MultiVectorRetriever.bind()
MultiVectorRetriever.config_schema()
MultiVectorRetriever.configurable_alternatives()
MultiVectorRetriever.configurable_fields()
MultiVectorRetriever.construct()
MultiVectorRetriever.copy()
MultiVectorRetriever.dict()
MultiVectorRetriever.from_orm()
MultiVectorRetriever.get_graph()
MultiVectorRetriever.get_input_schema()
MultiVectorRetriever.get_lc_namespace()
MultiVectorRetriever.get_name()
MultiVectorRetriever.get_output_schema()
MultiVectorRetriever.get_prompts()
MultiVectorRetriever.get_relevant_documents()
MultiVectorRetriever.invoke()
MultiVectorRetriever.is_lc_serializable()
MultiVectorRetriever.json()
MultiVectorRetriever.lc_id()
MultiVectorRetriever.map()
MultiVectorRetriever.parse_file()
MultiVectorRetriever.parse_obj()
MultiVectorRetriever.parse_raw()
MultiVectorRetriever.pick()
MultiVectorRetriever.pipe()
MultiVectorRetriever.schema()
MultiVectorRetriever.schema_json()
MultiVectorRetriever.stream()
MultiVectorRetriever.to_json()
MultiVectorRetriever.to_json_not_implemented()
MultiVectorRetriever.transform()
MultiVectorRetriever.update_forward_refs()
MultiVectorRetriever.validate()
MultiVectorRetriever.with_config()
MultiVectorRetriever.with_fallbacks()
MultiVectorRetriever.with_listeners()
MultiVectorRetriever.with_retry()
MultiVectorRetriever.with_types()
MultiVectorRetriever.InputType
MultiVectorRetriever.OutputType
MultiVectorRetriever.config_specs
MultiVectorRetriever.input_schema
MultiVectorRetriever.lc_attributes
MultiVectorRetriever.lc_secrets
MultiVectorRetriever.name
MultiVectorRetriever.output_schema

langchain.retrievers.multi_vector.MultiVectorRetriever¶

class langchain.retrievers.multi_vector.MultiVectorRetriever[source]¶
Bases: BaseRetriever
Retrieve from a set of multiple embeddings for the same document.
Create a new model by parsing and validating input data from keyword arguments.
Raises ValidationError if the input data cannot be parsed to form a valid model.

param byte_store: Optional[BaseStore[str, bytes]] = None¶
The lower-level backing storage layer for the parent documents

param docstore: BaseStore[str, Document] [Required]¶
The storage interface for the parent documents

param id_key: str = 'doc_id'¶

param metadata: Optional[Dict[str, Any]] = None¶
Optional metadata associated with the retriever. Defaults to None
This metadata will be associated with each call to this retriever,
and passed as arguments to the handlers defined in callbacks.
You can use these to eg identify a specific instance of a retriever with its
use case.

param search_kwargs: dict [Optional]¶
Keyword arguments to pass to the search function.

param search_type: SearchType = SearchType.similarity¶
Type of search to perform (similarity / mmr)</doc>
<doc id='3'>```python
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain.storage import InMemoryStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="big2small", embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    search_type=SearchType.mmr,  # use max marginal relevance search
    search_kwargs={"k": 2},
)

# Add child chunks to vector store
retriever.vectorstore.add_documents(list(children_by_id.values()))

# Add parent chunks to docstore
retriever.docstore.mset(parents_by_id.items())
```

```python
# Query vector store directly, should return chunks
found_chunks = vectorstore.similarity_search(
    "what signs does Birch Street allow on their property?", k=2
)

for chunk in found_chunks:
    print(chunk.page_content)
    print(chunk.metadata[loader.parent_id_key])
```</doc>
<doc id='4'>PGVector | 🦜️🔗 Langchain

[Skip to main content](#__docusaurus_skipToContent_fallback)# PGVector

[PGVector](https://github.com/pgvector/pgvector) is a vector
similarity search for Postgres.

In the notebook, we’ll demo the `SelfQueryRetriever` wrapped around a
`PGVector` vector store.

## Creating a PGVector vector store​

First we’ll want to create a PGVector vector store and seed it with some
data. We’ve created a small demo set of documents that contain summaries
of movies.

**Note:** The self-query retriever requires you to have `lark` installed
(`pip install lark`). We also need the  ` `  package.

```python
%pip install --upgrade --quiet  lark pgvector psycopg2-binary
```

We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
```

```python
from langchain.schema import Document
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

collection = "Name of your collection"
embeddings = OpenAIEmbeddings()
```

```python
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]
vectorstore = PGVector.from_documents(
    docs,
    embeddings,
    collection_name=collection,
)
```

## Creating our self-querying retriever​

Now we can instantiate our retriever. To do this we’ll need to provide
some information upfront about the metadata fields that our documents
support and a short description of the document contents.

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)
```

## Testing it out​

And now we can try actually using our retriever!

```python
# This example only specifies a relevant query
retriever.get_relevant_documents("What are some movies about dinosaurs")
```

```python
# This example only specifies a filter
retriever.get_relevant_documents("I want to watch a movie rated higher than 8.5")
```</doc>
<doc id='5'>- [Querying vectors by time and similarity](#querying-vectors-by-time-and-similarity)

- [3. Using ANN Search Indexes to Speed Up Queries](#using-ann-search-indexes-to-speed-up-queries)

- [4. Self Querying Retriever with Timescale Vector](#self-querying-retriever-with-timescale-vector)

- [5. Working with an existing TimescaleVector vectorstore](#working-with-an-existing-timescalevector-vectorstore)- [Deleting Data](#deleting-data)

- [Overriding a vectorstore](#overriding-a-vectorstore)</doc> 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.