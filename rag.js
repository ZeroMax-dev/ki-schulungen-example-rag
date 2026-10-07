// 2-step RAG: retrieval ALWAYS runs before the model answers.
// Docs: https://docs.langchain.com/oss/javascript/deepagents/retrieval (section "2-step RAG")

// Load environment variables from .env file
import "dotenv/config";

import * as fs from "node:fs/promises";

import { createAgent, dynamicSystemPromptMiddleware } from "langchain";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
// In LangChain v1 the lightweight in-memory vector store lives in @langchain/classic
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

// Check if OpenAI API key is available
if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is not set in .env file");
  process.exit(1);
}

// --- Optional: corporate proxy ---------------------------------------------
// The OpenAI SDK uses fetch, so a proxy is configured with undici's ProxyAgent
// (the old `httpAgent` option is ignored by the current SDK).
// import { fetch, ProxyAgent } from "undici";
// const proxyAgent = new ProxyAgent(
//   `http://${process.env.PROXY_USERNAME}:${process.env.PROXY_PASSWORD}@webproxy.prod.d003.loc:8080`
// );
// For self-signed certificates in the SSL chain:
// process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

// Extra options for the underlying OpenAI client (same for chat + embeddings)
const openAIConfiguration = {
  // apiKey: process.env.OPENAI_API_KEY,
  // baseURL: "https://api.openai.com/v1",
  // fetch,
  // fetchOptions: { dispatcher: proxyAgent },
};

async function runRAG() {
  console.log("Starting RAG example...");

  // 1. Load the document
  // A plain text file needs no special loader: read it and let the splitter
  // wrap it in LangChain Documents.
  console.log("Loading and processing text...");
  const text = await fs.readFile("state_of_the_union.txt", "utf8");

  // 2. Split it into chunks
  // https://docs.langchain.com/oss/javascript/integrations/splitters
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await textSplitter.createDocuments(
    [text],
    [{ source: "state_of_the_union.txt" }]
  );
  console.log(`Split text into ${docs.length} documents`);

  // 3. Embed the chunks and store them in a vector store
  // https://docs.langchain.com/oss/javascript/integrations/vectorstores
  console.log("Creating vector store from documents...");
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    configuration: openAIConfiguration,
  });
  const vectorStore = new MemoryVectorStore(embeddings);
  await vectorStore.addDocuments(docs);

  // 4. Retrieve + augment: before every model call, search the vector store
  // for the user's question and put the matching chunks into the system prompt.
  // https://docs.langchain.com/oss/javascript/langchain/middleware/overview
  const promptWithContext = dynamicSystemPromptMiddleware(async (state) => {
    const question = state.messages.at(-1).text;
    const relevantDocs = await vectorStore.similaritySearch(question, 4);
    const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");
    return `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
${context}`;
  });

  // 5. Generate: an agent without tools is simply "prompt -> model -> answer".
  // (We pass a ChatOpenAI instance instead of the "openai:gpt-5.4-mini" string
  // so the optional proxy configuration above can be applied.)
  const model = new ChatOpenAI({
    model: "gpt-5.4-mini",
    configuration: openAIConfiguration,
  });
  const agent = createAgent({
    model,
    tools: [],
    middleware: [promptWithContext],
  });

  // Execute an example query
  const question = "What did the president say about Justice Breyer?";
  console.log(`\nQuery: ${question}\n`);

  // streamMode "messages" streams the answer token by token
  // https://docs.langchain.com/oss/javascript/langchain/streaming
  const stream = await agent.stream(
    { messages: [{ role: "user", content: question }] },
    { streamMode: "messages" }
  );
  for await (const [token] of stream) {
    process.stdout.write(token.text);
  }
  process.stdout.write("\n");
}

// Run the RAG example
runRAG().catch((error) => {
  console.error("Error running RAG example:", error);
});
