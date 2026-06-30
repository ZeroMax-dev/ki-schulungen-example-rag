// Load environment variables from .env file
import dotenv from 'dotenv';
dotenv.config({ quiet: true });

import * as fs from "node:fs";

import { HttpsProxyAgent } from "https-proxy-agent";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
// In LangChain v1 the lightweight in-memory vector store moved to @langchain/classic
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Check if OpenAI API key is available
if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is not set in .env file");
  process.exit(1);
}

// For self-signed certifiactes in the SSL chain
// process.env.NODE_TLS_REJECT_UNAUTHORIZED = 0;

// Define proxy URL
// const proxyUrl = `http://${process.env.PROXY_USERNAME}:${process.env.PROXY_PASSWORD}@webproxy.prod.d003.loc:8080`;
// Create proxy agents for HTTP and HTTPS
// const proxyAgent = new HttpsProxyAgent(proxyUrl);

// Helper function to format documents as a string
const formatDocumentsAsString = (documents) => {
  return documents.map((document) => document.pageContent).join("\n\n");
};

// Main function to run the RAG example
async function runRAG() {
  console.log("Starting RAG example...");
  
  try {
    // Initialize the LLM to use to answer the question
    const model = new ChatOpenAI({
      model: "gpt-4o",
      configuration: {
        // apiKey: process.env.OPENAI_API_KEY,
        // baseURL: "https://api.openai.com/v1",
        // httpAgent: proxyAgent
      }
    });
    
    console.log("Loading and processing text...");
    const text = fs.readFileSync("state_of_the_union.txt", "utf8");
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    const docs = await textSplitter.createDocuments([text]);
    
    console.log(`Split text into ${docs.length} documents`);
    
    // Create a vector store from the documents
    console.log("Creating vector store from documents...");
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({
        configuration: {
          // apiKey: process.env.OPENAI_API_KEY,
          // baseURL: "https://api.openai.com/v1",
          // httpAgent: proxyAgent
        }
      })
    );
    
    // Initialize a retriever wrapper around the vector store
    const vectorStoreRetriever = vectorStore.asRetriever();
    
    // Create a system & human prompt for the chat model
    const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}`;
    
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", SYSTEM_TEMPLATE],
      ["human", "{question}"],
    ]);
    
    console.log("Setting up RAG chain...");
    const chain = RunnableSequence.from([
      {
        context: vectorStoreRetriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough(),
      },
      prompt,
      model,
      new StringOutputParser(),
    ]);
    
    // Execute an example query
    const question = "What did the president say about Justice Breyer?";
    console.log(`\nQuery: ${question}`);
    
    console.log("Retrieving answer...");
    const answer = await chain.invoke(question);
    
    console.log({ answer });
  } catch (error) {
    console.error("Error running RAG example:", error);
  }
}

// Run the RAG example
runRAG();
