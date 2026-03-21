import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { Document } from "@langchain/core/documents";

const DOCS = [
  "TruLens is an open-source library for evaluating and tracking LLM apps.",
  "RAG combines retrieval with LLM generation for grounded answers.",
  "OpenTelemetry is a vendor-neutral observability framework.",
  "LangChain Expression Language (LCEL) lets you compose chains declaratively.",
];

function formatDocs(docs: Document[]): string {
  return docs.map((d, i) => `[${i + 1}] ${d.pageContent}`).join("\n\n");
}

export async function buildChain() {
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromTexts(
    DOCS,
    DOCS.map((_, i) => ({ id: i })),
    embeddings,
  );
  const retriever = vectorStore.asRetriever({ k: 2 });

  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the question using only the following context:

{context}

Question: {question}`,
  );

  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

  const chain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocs),
      question: new RunnablePassthrough(),
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  return chain;
}
