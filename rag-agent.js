// Agentic RAG: the agent decides WHEN and HOW to search.
// Retrieval is just a tool; the model writes its own search queries and may
// search several times (e.g. once per sub-question) before it answers.
// Docs: https://docs.langchain.com/oss/javascript/deepagents/retrieval (section "Agentic RAG")

// Load environment variables from .env file
import "dotenv/config";

import * as fs from "node:fs/promises";

import * as z from "zod";
import { createAgent, tool } from "langchain";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

// Check if OpenAI API key is available
if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is not set in .env file");
  process.exit(1);
}

// Build the vector store (same steps as in rag.js: load, split, embed, store)
console.log("Indexing state_of_the_union.txt...");
const text = await fs.readFile("state_of_the_union.txt", "utf8");
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const docs = await textSplitter.createDocuments(
  [text],
  [{ source: "state_of_the_union.txt" }]
);
const vectorStore = new MemoryVectorStore(
  new OpenAIEmbeddings({ model: "text-embedding-3-small" })
);
await vectorStore.addDocuments(docs);

// The retrieval tool. The description tells the model what the tool is for.
// responseFormat "content_and_artifact" sends the text to the model and keeps
// the raw Documents as an artifact on the ToolMessage (handy for citations).
// https://docs.langchain.com/oss/javascript/langchain/tools
const retrieveContext = tool(
  async ({ query }) => {
    const relevantDocs = await vectorStore.similaritySearch(query, 4);
    const serialized = relevantDocs
      .map((doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`)
      .join("\n\n");
    return [serialized, relevantDocs];
  },
  {
    name: "retrieve_context",
    description:
      "Search the 2022 State of the Union address for passages relevant to the query.",
    schema: z.object({
      query: z.string().describe("The search query"),
    }),
    responseFormat: "content_and_artifact",
  }
);

const agent = createAgent({
  model: "openai:gpt-5.4-mini",
  tools: [retrieveContext],
  systemPrompt:
    "You answer questions about the 2022 State of the Union address. " +
    "Always use the retrieve_context tool to look up relevant passages before answering. " +
    "If the passages don't contain the answer, say that you don't know.",
});

const question =
  "What did the president say about Justice Breyer, and what did he say about inflation?";
console.log(`\nQuery: ${question}\n`);

// streamMode "updates" yields what each step (model / tools) produced, so we
// can watch the agent call the tool (and with which queries) before it answers.
// https://docs.langchain.com/oss/javascript/langchain/streaming
const stream = await agent.stream(
  { messages: [{ role: "user", content: question }] },
  { streamMode: "updates" }
);
for await (const update of stream) {
  for (const [step, { messages }] of Object.entries(update)) {
    const message = messages.at(-1);
    if (message.tool_calls?.length) {
      for (const call of message.tool_calls) {
        console.log(`[${step}] calls ${call.name}(${JSON.stringify(call.args)})`);
      }
    } else if (step === "tools") {
      console.log(`[${step}] returned ${message.artifact?.length ?? 0} passages`);
    } else {
      console.log(`[${step}] answer:\n${message.text}`);
    }
  }
}
