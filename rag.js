// Load environment variables from .env file
import dotenv from 'dotenv';
dotenv.config();

import * as fs from "node:fs";

import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";

// Check if OpenAI API key is available
if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is not set in .env file");
  process.exit(1);
}

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
      new OpenAIEmbeddings()
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
