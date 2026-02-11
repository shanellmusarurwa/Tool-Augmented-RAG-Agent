#!/usr/bin/env node

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import readline from "readline";
import { ChromaClient } from "chromadb";
import { OpenAI } from "openrouter";
import dotenv from "dotenv";
import pdfParse from "pdf-parse/lib/pdf-parse.js";
import { pipeline } from "@xenova/transformers";

// LangChain imports
import { DynamicTool } from "@langchain/core/tools";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { AgentExecutor, createReactAgent } from "@langchain/core/agents";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ConversationSummaryBufferMemory } from "langchain/memory";

// Load environment variables
dotenv.config();

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
class Config {
  constructor() {
    // Required API keys
    this.openrouterApiKey = process.env.OPENROUTER_API_KEY;
    this.hfApiKey = process.env.HF_API_KEY;

    if (!this.openrouterApiKey) {
      throw new Error("OPENROUTER_API_KEY is required in .env file");
    }

    // Optional configuration with defaults
    this.chromaPersistDirectory =
      process.env.CHROMA_PERSIST_DIRECTORY || "./chroma_db";
    this.embeddingModel =
      process.env.EMBEDDING_MODEL || "sentence-transformers/all-MiniLM-L6-v2";
    this.llmModel =
      process.env.LLM_MODEL || "nvidia/nemotron-3-nano-30b-a3b:free";
    this.temperature = parseFloat(process.env.TEMPERATURE || "0.7");
    this.maxTokens = parseInt(process.env.MAX_TOKENS || "1000");

    // Data directory
    this.dataDirectory = path.join(__dirname, "data");

    // Create directories if they don't exist
    if (!fs.existsSync(this.dataDirectory)) {
      fs.mkdirSync(this.dataDirectory, { recursive: true });
    }
    if (!fs.existsSync(this.chromaPersistDirectory)) {
      fs.mkdirSync(this.chromaPersistDirectory, { recursive: true });
    }
  }
}

// RAG Vector Store Manager
class RAGVectorStore {
  constructor(config) {
    this.config = config;
    this.embeddings = null;
    this.docVectorStore = null;
    this.convVectorStore = null;
    this.retriever = null;
    this.documents = [];

    // ChromaDB client for persistent storage
    this.chromaClient = new ChromaClient({
      path: config.chromaPersistDirectory,
    });
  }

  async initialize() {
    console.log("🔄 Initializing embeddings and vector stores...");

    // Initialize embeddings
    this.embeddings = new HuggingFaceTransformersEmbeddings({
      modelName: this.config.embeddingModel,
      stripNewLines: true,
    });

    // Initialize or load document vector store
    try {
      // Try to load existing vector store
      this.docVectorStore = await Chroma.fromExistingCollection(
        this.embeddings,
        {
          collectionName: "documents",
          url: this.config.chromaPersistDirectory,
        },
      );
      const count = await this.docVectorStore.count();
      console.log(`✅ Loaded existing document store with ${count} documents`);
    } catch (error) {
      // Create new vector store
      console.log("📝 Creating new document vector store");
      this.docVectorStore = new Chroma(this.embeddings, {
        collectionName: "documents",
        url: this.config.chromaPersistDirectory,
      });
    }

    // Initialize or load conversation vector store
    try {
      this.convVectorStore = await Chroma.fromExistingCollection(
        this.embeddings,
        {
          collectionName: "conversations",
          url: this.config.chromaPersistDirectory,
        },
      );
      const count = await this.convVectorStore.count();
      console.log(
        `✅ Loaded existing conversation store with ${count} conversations`,
      );
    } catch (error) {
      console.log("📝 Creating new conversation vector store");
      this.convVectorStore = new Chroma(this.embeddings, {
        collectionName: "conversations",
        url: this.config.chromaPersistDirectory,
      });
    }

    return this;
  }

  async loadAndIndexDocuments() {
    console.log("📚 Loading documents from data directory...");

    if (!fs.existsSync(this.config.dataDirectory)) {
      console.log("⚠️  Data directory does not exist, creating...");
      fs.mkdirSync(this.config.dataDirectory, { recursive: true });
      return;
    }

    const files = fs.readdirSync(this.config.dataDirectory);
    const textFiles = files.filter(
      (file) =>
        file.endsWith(".txt") || file.endsWith(".pdf") || file.endsWith(".md"),
    );

    if (textFiles.length === 0) {
      console.log("⚠️  No documents found in data directory");
      return;
    }

    console.log(`📄 Found ${textFiles.length} documents to process`);

    let allDocuments = [];

    for (const file of textFiles) {
      const filePath = path.join(this.config.dataDirectory, file);
      try {
        let content = "";
        let metadata = {
          source: file,
          path: filePath,
          timestamp: new Date().toISOString(),
        };

        if (file.endsWith(".pdf")) {
          // Parse PDF
          const dataBuffer = fs.readFileSync(filePath);
          const pdfData = await pdfParse(dataBuffer);
          content = pdfData.text;
          metadata.pageCount = pdfData.numpages;
        } else {
          // Read text/markdown files
          content = fs.readFileSync(filePath, "utf8");
          metadata.type = file.endsWith(".md") ? "markdown" : "text";
        }

        allDocuments.push({ content, metadata });
        console.log(`  ✅ Loaded: ${file}`);
      } catch (error) {
        console.error(`  ❌ Error loading ${file}:`, error.message);
      }
    }

    // Split documents into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ["\n\n", "\n", " ", ""],
    });

    const chunks = [];
    for (const doc of allDocuments) {
      const docChunks = await textSplitter.createDocuments(
        [doc.content],
        [doc.metadata],
      );
      chunks.push(...docChunks);
    }

    console.log(`✂️  Split into ${chunks.length} chunks`);

    // Add to vector store
    if (chunks.length > 0) {
      await this.docVectorStore.addDocuments(chunks);
      console.log(`✅ Indexed ${chunks.length} document chunks`);

      // Store documents for BM25 retrieval
      this.documents = chunks;
    }

    // Create hybrid retriever
    await this.createHybridRetriever();
  }

  async createHybridRetriever() {
    // Simple vector retriever for now
    // In a full implementation, you'd want to combine with BM25
    this.retriever = this.docVectorStore.asRetriever(5);
    console.log("🔄 Created hybrid retriever");
  }

  async saveConversationTurn(query, response) {
    const conversationText = `Human: ${query}\nAI: ${response}`;

    const doc = {
      pageContent: conversationText,
      metadata: {
        timestamp: new Date().toISOString(),
        query: query,
        response: response,
      },
    };

    await this.convVectorStore.addDocuments([doc]);
    console.log("💾 Saved conversation turn to memory");
  }

  async retrieveConversationHistory(query, k = 3) {
    try {
      const results = await this.convVectorStore.similaritySearch(query, k);
      return results.map((doc) => doc.pageContent);
    } catch (error) {
      console.error("Error retrieving conversation history:", error.message);
      return [];
    }
  }

  async queryRag(query, k = 5) {
    if (!this.retriever) {
      return [];
    }
    return await this.retriever.invoke(query);
  }
}

// Tool Definitions
class ToolDefinitions {
  static getWeather(location) {
    // Mock implementation
    const weatherData = {
      jos: "Sunny, 25°C",
      lagos: "Cloudy, 28°C",
      abuja: "Partly cloudy, 27°C",
      london: "Rainy, 12°C",
      "new york": "Snowy, -2°C",
      tokyo: "Clear, 15°C",
      paris: "Overcast, 10°C",
    };

    const normalizedLocation = location.toLowerCase().trim();

    if (weatherData[normalizedLocation]) {
      return `The weather in ${location} is ${weatherData[normalizedLocation]}`;
    } else {
      return `Weather information for "${location}" is not available. Try: ${Object.keys(weatherData).join(", ")}`;
    }
  }

  static calculateMath(expression) {
    try {
      // Safe evaluation - only allow basic math operations
      const sanitized = expression.replace(/[^0-9+\-*/().\s]/g, "");

      if (sanitized !== expression) {
        return "Error: Expression contains invalid characters. Only numbers and + - * / ( ) . are allowed.";
      }

      // Using Function constructor is safer than eval
      const result = new Function(`return ${sanitized}`)();
      return `The result of ${expression} is ${result}`;
    } catch (error) {
      return `Error calculating expression: ${error.message}. Please use valid math syntax.`;
    }
  }

  static getCurrentTime(timezone = "UTC") {
    try {
      const now = new Date();

      // Simplified timezone handling
      const timezoneMap = {
        UTC: 0,
        GMT: 0,
        EST: -5,
        EDT: -4,
        CST: -6,
        CDT: -5,
        MST: -7,
        MDT: -6,
        PST: -8,
        PDT: -7,
        CET: 1,
        EET: 2,
        IST: 5.5,
        JST: 9,
        AEST: 10,
        AEDT: 11,
      };

      const upperTimezone = timezone.toUpperCase().trim();

      if (upperTimezone === "UTC" || upperTimezone === "GMT") {
        return `Current time in ${timezone}: ${now.toUTCString()}`;
      } else if (timezoneMap[upperTimezone] !== undefined) {
        const offset = timezoneMap[upperTimezone];
        const utcHours = now.getUTCHours();
        const utcMinutes = now.getUTCMinutes();

        let localHours = utcHours + offset;
        if (localHours < 0) localHours += 24;
        if (localHours >= 24) localHours -= 24;

        const timeStr = `${localHours.toString().padStart(2, "0")}:${utcMinutes.toString().padStart(2, "0")}:${now.getUTCSeconds().toString().padStart(2, "0")}`;
        return `Current time in ${timezone}: ${timeStr} (${now.toDateString()})`;
      } else {
        return `Timezone "${timezone}" not recognized. Using UTC: ${now.toUTCString()}`;
      }
    } catch (error) {
      return `Error getting time: ${error.message}`;
    }
  }

  static async queryInternalKnowledge(query, ragStore) {
    try {
      const docs = await ragStore.queryRag(query, 3);

      if (!docs || docs.length === 0) {
        return "No relevant information found in the knowledge base.";
      }

      let response = "📚 Found relevant information:\n\n";

      docs.forEach((doc, i) => {
        const source = doc.metadata?.source || "Unknown source";
        const content = doc.pageContent.substring(0, 300); // Limit content
        response += `[${i + 1}] From ${source}:\n${content}...\n\n`;
      });

      return response;
    } catch (error) {
      return `Error querying knowledge base: ${error.message}`;
    }
  }
}

// Main RAG Agent
class RAGAgent {
  constructor(config) {
    this.config = config;
    this.ragStore = null;
    this.llm = null;
    this.agent = null;
    this.memory = null;
    this.conversationHistory = [];
  }

  async initialize() {
    console.log("🚀 Initializing RAG Agent...");

    // Initialize RAG store
    this.ragStore = new RAGVectorStore(this.config);
    await this.ragStore.initialize();

    // Initialize LLM with OpenRouter
    this.llm = new ChatOpenAI({
      modelName: this.config.llmModel,
      temperature: this.config.temperature,
      maxTokens: this.config.maxTokens,
      openAIApiKey: this.config.openrouterApiKey,
      configuration: {
        baseURL: "https://openrouter.ai/api/v1",
        defaultHeaders: {
          "HTTP-Referer": "http://localhost",
          "X-Title": "Tool Augmented RAG Agent",
        },
      },
    });

    // Initialize memory
    this.memory = new ConversationSummaryBufferMemory({
      memoryKey: "chat_history",
      returnMessages: true,
      llm: this.llm,
      maxTokenLimit: 2000,
    });

    // Load documents
    await this.ragStore.loadAndIndexDocuments();

    // Create tools
    const tools = await this.createTools();

    // Create agent
    await this.createAgent(tools);

    console.log("✅ Agent initialized successfully!");
    return this;
  }

  async createTools() {
    const weatherTool = new DynamicTool({
      name: "GetWeather",
      description:
        'Get current weather for a location. Input should be a city name. Example: "Jos" or "London"',
      func: async (input) => {
        console.log(`  🔧 Tool: GetWeather called with: "${input}"`);
        const result = ToolDefinitions.getWeather(input);
        console.log(`  ✅ Tool result: ${result}`);
        return result;
      },
    });

    const mathTool = new DynamicTool({
      name: "CalculateMath",
      description:
        'Calculate a mathematical expression. Input should be a valid math expression like "2+2" or "3*5".',
      func: async (input) => {
        console.log(`  🔧 Tool: CalculateMath called with: "${input}"`);
        const result = ToolDefinitions.calculateMath(input);
        console.log(`  ✅ Tool result: ${result}`);
        return result;
      },
    });

    const timeTool = new DynamicTool({
      name: "GetCurrentTime",
      description:
        'Get current time for a timezone. Input should be a timezone name like "UTC", "EST", "PST", "CET". Defaults to UTC if not specified.',
      func: async (input) => {
        console.log(`  🔧 Tool: GetCurrentTime called with: "${input}"`);
        const timezone = input || "UTC";
        const result = ToolDefinitions.getCurrentTime(timezone);
        console.log(`  ✅ Tool result: ${result}`);
        return result;
      },
    });

    const ragTool = new DynamicTool({
      name: "QueryKnowledgeBase",
      description:
        "Search internal knowledge base for information. Use this for any questions about company documents, policies, manuals, or stored information.",
      func: async (input) => {
        console.log(`  🔧 Tool: QueryKnowledgeBase called with: "${input}"`);
        const result = await ToolDefinitions.queryInternalKnowledge(
          input,
          this.ragStore,
        );
        console.log(`  ✅ Tool result: ${result.substring(0, 100)}...`);
        return result;
      },
    });

    return [weatherTool, mathTool, timeTool, ragTool];
  }

  async createAgent(tools) {
    const agentPrompt = PromptTemplate.fromTemplate(`
            You are a helpful AI assistant with access to tools and a knowledge base.

            You have access to the following tools:
            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Previous conversation history:
            {chat_history}

            Question: {input}
            Thought: {agent_scratchpad}
        `);

    // Create the agent
    const agent = await createReactAgent({
      llm: this.llm,
      tools: tools,
      prompt: agentPrompt,
    });

    this.agent = new AgentExecutor({
      agent,
      tools,
      memory: this.memory,
      verbose: true,
      maxIterations: 5,
      returnIntermediateSteps: true,
      earlyStoppingMethod: "generate",
      handleParsingErrors: true,
    });
  }

  async processQuery(query) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`🔍 Processing query: "${query}"`);
    console.log(`${"=".repeat(60)}\n`);

    try {
      // Add to conversation history
      this.conversationHistory.push({
        role: "human",
        content: query,
        timestamp: new Date().toISOString(),
      });

      // Get response from agent
      const response = await this.agent.invoke({
        input: query,
      });

      const responseText = response.output || JSON.stringify(response);

      // Add AI response to conversation history
      this.conversationHistory.push({
        role: "ai",
        content: responseText,
        timestamp: new Date().toISOString(),
      });

      // Save to vector store
      await this.ragStore.saveConversationTurn(query, responseText);

      // Save to memory
      await this.memory.saveContext({ input: query }, { output: responseText });

      return responseText;
    } catch (error) {
      console.error("❌ Error processing query:", error);
      const errorMessage = `I encountered an error while processing your request: ${error.message}`;

      // Save error to conversation history
      this.conversationHistory.push({
        role: "ai",
        content: errorMessage,
        timestamp: new Date().toISOString(),
        error: true,
      });

      return errorMessage;
    }
  }

  printConversationHistory() {
    console.log(`\n${"=".repeat(60)}`);
    console.log("📋 CONVERSATION HISTORY");
    console.log(`${"=".repeat(60)}`);

    if (this.conversationHistory.length === 0) {
      console.log("No conversation history yet.");
      return;
    }

    this.conversationHistory.forEach((entry, index) => {
      const role = entry.role === "human" ? "👤 Human" : "🤖 AI";
      console.log(`\n[${index + 1}] ${role} (${entry.timestamp}):`);
      console.log(`${entry.content}`);
      if (entry.error) {
        console.log("⚠️  [ERROR]");
      }
      console.log("-".repeat(40));
    });

    console.log(`\n${"=".repeat(60)}`);
  }

  async printSystemStatus() {
    console.log(`\n${"=".repeat(60)}`);
    console.log("📊 SYSTEM STATUS");
    console.log(`${"=".repeat(60)}`);

    try {
      // Count documents
      const docCount = (await this.ragStore.docVectorStore?.count()) || 0;
      const convCount = (await this.ragStore.convVectorStore?.count()) || 0;

      console.log(`\n📚 Document Store: ${docCount} chunks indexed`);
      console.log(`💬 Conversation Store: ${convCount} conversations saved`);
      console.log(
        `📁 Data directory: ${fs.existsSync(this.config.dataDirectory) ? "✓" : "✗"}`,
      );
      console.log(
        `🗄️  ChromaDB: ${fs.existsSync(this.config.chromaPersistDirectory) ? "✓" : "✗"}`,
      );

      // List available tools
      console.log(`\n🔧 Available Tools:`);
      console.log(`  • GetWeather - Get current weather for a location`);
      console.log(`  • CalculateMath - Perform mathematical calculations`);
      console.log(
        `  • GetCurrentTime - Get current time in different timezones`,
      );
      console.log(`  • QueryKnowledgeBase - Search internal documents`);
    } catch (error) {
      console.log("Error getting system status:", error.message);
    }

    console.log(`\n${"=".repeat(60)}`);
  }
}

// Main execution
async function main() {
  console.log("\n" + "🌟".repeat(30));
  console.log("🌟  TOOL AUGMENTED RAG AGENT");
  console.log("🌟".repeat(30) + "\n");

  try {
    // Load configuration
    const config = new Config();
    console.log("✅ Configuration loaded");

    // Initialize agent
    const agent = new RAGAgent(config);
    await agent.initialize();

    // Get query from command line arguments
    const query = process.argv.slice(2).join(" ");

    if (!query) {
      console.log("\n⚠️  No query provided. Starting interactive mode...\n");

      // Interactive mode
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });

      console.log(
        'Interactive mode. Type "exit" to quit, "status" for system status, "history" for conversation history.\n',
      );

      const askQuestion = () => {
        rl.question("👤 Your query: ", async (input) => {
          if (input.toLowerCase() === "exit") {
            console.log("\n👋 Goodbye!");
            rl.close();
            return;
          }

          if (input.toLowerCase() === "status") {
            await agent.printSystemStatus();
            askQuestion();
            return;
          }

          if (input.toLowerCase() === "history") {
            agent.printConversationHistory();
            askQuestion();
            return;
          }

          const response = await agent.processQuery(input);
          console.log(`\n🤖 Assistant: ${response}\n`);
          askQuestion();
        });
      };

      askQuestion();
    } else {
      // Single query mode
      console.log(`\n📝 Processing query: "${query}"\n`);

      const response = await agent.processQuery(query);

      // Print conversation history
      agent.printConversationHistory();

      // Print final response
      console.log(`\n${"=".repeat(60)}`);
      console.log("🤖 FINAL RESPONSE");
      console.log(`${"=".repeat(60)}`);
      console.log(`\n${response}\n`);

      process.exit(0);
    }
  } catch (error) {
    console.error("\n❌ Fatal error:", error.message);
    if (error.stack) {
      console.error("\nStack trace:", error.stack);
    }
    process.exit(1);
  }
}

// Handle uncaught errors
process.on("uncaughtException", (error) => {
  console.error("\n💥 Uncaught Exception:", error);
  process.exit(1);
});

process.on("unhandledRejection", (reason, promise) => {
  console.error("\n💥 Unhandled Rejection at:", promise, "reason:", reason);
  process.exit(1);
});

// Run the application
main();
