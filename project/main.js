import dotenv from "dotenv";
import { ChromaClient } from "chromadb";
import { ChatOpenAI } from "@langchain/openai";
import { DynamicTool } from "@langchain/core/tools";
import { HumanMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { pipeline } from "@xenova/transformers";
import pdf from "pdf-parse";

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const DATA_DIR = path.join(__dirname, "data");

// Custom embedding function for ChromaDB
class EmbeddingFunction {
  constructor() {
    this.pipeline = null;
  }

  async initialize() {
    if (!this.pipeline) {
      console.log("🔄 Initializing embedding model...");
      this.pipeline = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2",
      );
      console.log("✅ Embedding model initialized");
    }
  }

  async generate(texts) {
    await this.initialize();
    const embeddings = [];
    for (const text of texts) {
      const result = await this.pipeline(text, {
        pooling: "mean",
        normalize: true,
      });
      embeddings.push(Array.from(result.data));
    }
    return embeddings;
  }
}

// Simple in-memory vector store (since ChromaDB is having issues)
class SimpleVectorStore {
  constructor() {
    this.documents = [];
    this.embeddings = [];
    this.metadatas = [];
    this.embeddingFunction = new EmbeddingFunction();
  }

  async initialize() {
    await this.embeddingFunction.initialize();
  }

  async addDocuments(docs, metadatas) {
    const embeddings = await this.embeddingFunction.generate(docs);
    for (let i = 0; i < docs.length; i++) {
      this.documents.push(docs[i]);
      this.embeddings.push(embeddings[i]);
      this.metadatas.push(metadatas[i] || {});
    }
    console.log(`✅ Added ${docs.length} items to vector store`);
  }

  async similaritySearch(query, k = 3) {
    const queryEmbedding = await this.embeddingFunction.generate([query]);
    const queryVec = queryEmbedding[0];

    // Calculate cosine similarity
    const similarities = this.embeddings.map((emb) => {
      const dotProduct = emb.reduce(
        (sum, val, i) => sum + val * queryVec[i],
        0,
      );
      const normA = Math.sqrt(emb.reduce((sum, val) => sum + val * val, 0));
      const normB = Math.sqrt(
        queryVec.reduce((sum, val) => sum + val * val, 0),
      );
      return dotProduct / (normA * normB);
    });

    // Get top k indices
    const indices = similarities
      .map((sim, idx) => ({ sim, idx }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, k)
      .map((item) => item.idx);

    return indices.map((idx) => ({
      document: this.documents[idx],
      metadata: this.metadatas[idx],
      similarity: similarities[idx],
    }));
  }

  count() {
    return this.documents.length;
  }
}

// Initialize vector stores
let documentStore;
let conversationStore;

// Tool call tracker
let toolCalls = [];

/**
 * Extract text from PDF
 */
async function extractTextFromPDF(filePath) {
  try {
    const dataBuffer = await fs.readFile(filePath);
    const pdfData = await pdf(dataBuffer);
    return pdfData.text;
  } catch (error) {
    console.error("Error extracting PDF text:", error);
    return "";
  }
}

/**
 * Load all documents from data directory
 */
async function loadDocumentsFromDataDir() {
  try {
    // Create data directory if it doesn't exist
    await fs.mkdir(DATA_DIR, { recursive: true });

    const files = await fs.readdir(DATA_DIR);

    if (files.length === 0) {
      console.log(
        "📁 Data directory is empty. Add files to data/ folder for RAG functionality.",
      );
      return;
    }

    for (const file of files) {
      const filePath = path.join(DATA_DIR, file);
      const ext = path.extname(file).toLowerCase();
      let content = "";

      console.log(`📄 Loading file: ${file}`);

      try {
        if (ext === ".txt" || ext === ".md" || ext === ".markdown") {
          content = await fs.readFile(filePath, "utf-8");
          console.log(`  📝 Text file loaded (${content.length} characters)`);
        } else if (ext === ".pdf") {
          content = await extractTextFromPDF(filePath);
          console.log(`  📑 PDF loaded (${content.length} characters)`);
        } else if (ext === ".json") {
          const fileContent = await fs.readFile(filePath, "utf-8");
          const jsonData = JSON.parse(fileContent);
          content = JSON.stringify(jsonData, null, 2);
          console.log(`  🔧 JSON file loaded`);
        } else {
          console.log(`  ⚠️ Unsupported file type: ${ext}`);
          continue;
        }
      } catch (fileError) {
        console.error(`  ❌ Error reading file ${file}:`, fileError.message);
        continue;
      }

      if (content && content.trim().length > 0) {
        // Split content into chunks
        const chunkSize = 1000;
        const overlap = 200;
        const chunks = [];
        const metadatas = [];

        for (let i = 0; i < content.length; i += chunkSize - overlap) {
          const chunk = content.substring(i, i + chunkSize);
          if (chunk.trim().length > 0) {
            chunks.push(chunk);
            metadatas.push({
              source: file,
              chunk: i,
              type: ext.substring(1),
              timestamp: new Date().toISOString(),
            });
          }
        }

        console.log(`  📦 Splitting into ${chunks.length} chunks`);

        // Add to vector store
        await documentStore.addDocuments(chunks, metadatas);

        console.log(`  ✅ Added ${chunks.length} chunks from ${file}`);
      }
    }

    console.log(
      `📚 Total document chunks in database: ${documentStore.count()}`,
    );
  } catch (error) {
    console.error("❌ Error loading documents:", error);
  }
}

/**
 * Initialize stores
 */
async function initializeStores() {
  console.log("🔄 Initializing vector stores...");

  // Create simple vector stores
  documentStore = new SimpleVectorStore();
  conversationStore = new SimpleVectorStore();

  // Initialize embedding function
  await documentStore.initialize();
  await conversationStore.initialize();

  // Load documents
  await loadDocumentsFromDataDir();

  console.log("✅ Vector stores initialized successfully");
}

/**
 * Save conversation to vector store
 */
async function saveConversation(userMessage, assistantMessage) {
  try {
    const conversation = JSON.stringify({
      user: userMessage,
      assistant: assistantMessage,
      timestamp: new Date().toISOString(),
    });

    await conversationStore.addDocuments(
      [conversation],
      [
        {
          type: "conversation",
          timestamp: Date.now(),
          user_query: userMessage.substring(0, 100),
        },
      ],
    );

    console.log("✅ Conversation saved to memory");
  } catch (error) {
    console.error("❌ Error saving conversation:", error);
  }
}

/**
 * Retrieve relevant conversations
 */
async function retrieveConversations(query, k = 3) {
  try {
    const results = await conversationStore.similaritySearch(query, k);
    return results.map((r) => {
      try {
        return JSON.parse(r.document);
      } catch {
        return { content: r.document };
      }
    });
  } catch (error) {
    console.error("❌ Error retrieving conversations:", error);
    return [];
  }
}

/**
 * Query documents using RAG
 */
async function queryDocuments(query, k = 3) {
  try {
    const results = await documentStore.similaritySearch(query, k);
    return results.map((r) => ({
      content: r.document,
      metadata: r.metadata,
      similarity: r.similarity,
    }));
  } catch (error) {
    console.error("❌ Error querying documents:", error);
    return [];
  }
}

/**
 * Tool 1: Get current weather
 */
const weatherTool = new DynamicTool({
  name: "get_weather",
  description:
    "Get the current weather for a specific location. Input should be a city name.",
  func: async (location) => {
    const toolCall = {
      name: "get_weather",
      input: location,
      timestamp: Date.now(),
    };
    toolCalls.push(toolCall);

    console.log(`\n🔧 Tool called: get_weather with location: ${location}`);

    const weatherData = {
      jos: "18°C, partly cloudy with light winds. Humidity: 65%",
      lagos: "28°C, sunny and humid. Humidity: 80%",
      abuja: "25°C, clear skies. Humidity: 55%",
      "new york": "10°C, rainy with moderate winds. Humidity: 75%",
      london: "12°C, cloudy with occasional drizzle. Humidity: 70%",
      tokyo: "15°C, clear skies. Humidity: 60%",
      paris: "14°C, partly cloudy. Humidity: 68%",
    };

    const normalizedLocation = location.toLowerCase().trim();
    const weather = weatherData[normalizedLocation];

    let result;
    if (weather) {
      result = `Current weather in ${location}: ${weather}`;
    } else {
      result = `Weather data not available for "${location}". Available locations: Jos, Lagos, Abuja, New York, London, Tokyo, Paris`;
    }

    toolCall.result = result;
    return result;
  },
});

/**
 * Tool 2: Currency conversion
 */
const currencyTool = new DynamicTool({
  name: "convert_currency",
  description:
    "Convert an amount from one currency to another. Input format: 'amount from_currency to_currency' e.g., '100 USD EUR'",
  func: async (input) => {
    const toolCall = {
      name: "convert_currency",
      input: input,
      timestamp: Date.now(),
    };
    toolCalls.push(toolCall);

    console.log(`\n🔧 Tool called: convert_currency with input: ${input}`);

    const parts = input.trim().split(" ");
    if (parts.length !== 3) {
      const result =
        "❌ Invalid format. Please use: 'amount from_currency to_currency'";
      toolCall.result = result;
      return result;
    }

    const [amount, from, to] = parts;

    if (isNaN(amount) || parseFloat(amount) <= 0) {
      const result = "❌ Please provide a valid positive amount";
      toolCall.result = result;
      return result;
    }

    const rates = {
      USD: 1.0,
      EUR: 0.92,
      GBP: 0.79,
      JPY: 148.5,
      NGN: 1550,
      CAD: 1.35,
      AUD: 1.52,
    };

    const fromUpper = from.toUpperCase();
    const toUpper = to.toUpperCase();

    if (!rates[fromUpper] || !rates[toUpper]) {
      const result = `❌ Currency not supported. Supported: ${Object.keys(rates).join(", ")}`;
      toolCall.result = result;
      return result;
    }

    const amountNum = parseFloat(amount);
    const converted = (amountNum / rates[fromUpper]) * rates[toUpper];

    const result = `✅ ${amountNum.toFixed(2)} ${fromUpper} = ${converted.toFixed(2)} ${toUpper}`;
    toolCall.result = result;
    return result;
  },
});

/**
 * Tool 3: Time zone information
 */
const timeZoneTool = new DynamicTool({
  name: "get_time_zone",
  description:
    "Get time zone information for a city. Input should be a city name.",
  func: async (city) => {
    const toolCall = {
      name: "get_time_zone",
      input: city,
      timestamp: Date.now(),
    };
    toolCalls.push(toolCall);

    console.log(`\n🔧 Tool called: get_time_zone with city: ${city}`);

    const timeZones = {
      jos: "UTC+1 (West Africa Time)",
      lagos: "UTC+1 (West Africa Time)",
      accra: "UTC+0 (Greenwich Mean Time)",
      london: "UTC+0 (Greenwich Mean Time)",
      "new york": "UTC-5 (Eastern Time)",
      tokyo: "UTC+9 (Japan Standard Time)",
      sydney: "UTC+11 (Australian Eastern Time)",
      dubai: "UTC+4 (Gulf Standard Time)",
    };

    const normalizedCity = city.toLowerCase().trim();
    const tzInfo = timeZones[normalizedCity];

    let result;
    if (tzInfo) {
      result = `📍 Time zone for ${city}: ${tzInfo}`;
    } else {
      result = `❌ Time zone info not available for "${city}". Available cities: Jos, Lagos, Accra, London, New York, Tokyo, Sydney, Dubai`;
    }

    toolCall.result = result;
    return result;
  },
});

/**
 * Tool 4: RAG query for internal information
 */
const ragTool = new DynamicTool({
  name: "query_internal_knowledge",
  description:
    "Query the internal document database for information. Use this for questions about loaded documents.",
  func: async (query) => {
    const toolCall = {
      name: "query_internal_knowledge",
      input: query,
      timestamp: Date.now(),
    };
    toolCalls.push(toolCall);

    console.log(
      `\n🔧 Tool called: query_internal_knowledge with query: "${query}"`,
    );

    try {
      const results = await queryDocuments(query, 3);

      if (results.length === 0) {
        const result =
          "📚 No relevant information found in internal documents.";
        toolCall.result = result;
        return result;
      }

      let response = "📚 **Information found in documents:**\n\n";

      for (let i = 0; i < results.length; i++) {
        const doc = results[i];
        response += `📄 From ${doc.metadata.source} (Relevance: ${(doc.similarity * 100).toFixed(1)}%):\n`;
        response += `${doc.content.substring(0, 300)}...\n\n`;
      }

      toolCall.result = response;
      return response;
    } catch (error) {
      console.error("Error in RAG tool:", error);
      const result = "❌ Error querying knowledge base.";
      toolCall.result = result;
      return result;
    }
  },
});

/**
 * Function to determine which tool to use
 */
async function selectAndExecuteTool(query) {
  const lowerQuery = query.toLowerCase();

  // Weather detection
  if (
    lowerQuery.includes("weather") ||
    lowerQuery.includes("temperature") ||
    lowerQuery.includes("forecast")
  ) {
    const cities = [
      "jos",
      "lagos",
      "abuja",
      "new york",
      "london",
      "tokyo",
      "paris",
    ];
    for (const city of cities) {
      if (lowerQuery.includes(city)) {
        return await weatherTool.func(city);
      }
    }
    return "Please specify a city for weather information (e.g., Jos, Lagos, London)";
  }

  // Currency conversion detection
  if (
    lowerQuery.includes("convert") ||
    lowerQuery.includes("exchange") ||
    lowerQuery.includes("currency")
  ) {
    const words = query.split(" ");
    for (let i = 0; i < words.length; i++) {
      if (!isNaN(words[i]) && i < words.length - 2) {
        const amount = words[i];
        const fromCurrency = words[i + 1].toUpperCase();
        const toCurrency = words[i + 2].toUpperCase();
        return await currencyTool.func(
          `${amount} ${fromCurrency} ${toCurrency}`,
        );
      }
    }
    return await currencyTool.func("100 USD EUR");
  }

  // Time zone detection
  if (
    lowerQuery.includes("time") ||
    lowerQuery.includes("timezone") ||
    lowerQuery.includes("time zone") ||
    lowerQuery.includes("clock")
  ) {
    const cities = [
      "jos",
      "lagos",
      "accra",
      "london",
      "new york",
      "tokyo",
      "sydney",
      "dubai",
    ];
    for (const city of cities) {
      if (lowerQuery.includes(city)) {
        return await timeZoneTool.func(city);
      }
    }
    return await timeZoneTool.func("Jos");
  }

  // For any other query, try RAG
  return await ragTool.func(query);
}

/**
 * Generate response using LLM
 */
async function generateResponse(query, toolResult, conversationHistory) {
  try {
    const llm = new ChatOpenAI({
      openAIApiKey: OPENROUTER_API_KEY,
      model: "nvidia/nemotron-3-nano-30b-a3b:free",
      configuration: {
        baseURL: "https://openrouter.ai/api/v1",
      },
      temperature: 0.7,
    });

    let context = "";
    if (toolResult && !toolResult.startsWith("Please specify")) {
      context = `\nTool result: ${toolResult}\n`;
    }

    let history = "";
    if (conversationHistory.length > 0) {
      history = "\nPrevious relevant conversations:\n";
      conversationHistory.forEach((conv, i) => {
        history += `${i + 1}. User: ${conv.user}\n   Assistant: ${conv.assistant}\n`;
      });
    }

    const responsePrompt = PromptTemplate.fromTemplate(`
      You are a helpful assistant. Answer the user's query based on the available information.
      
      User query: {query}
      {context}
      {history}
      
      Provide a helpful, natural response:
    `);

    const chain = RunnableSequence.from([
      responsePrompt,
      llm,
      new StringOutputParser(),
    ]);

    return await chain.invoke({
      query,
      context,
      history,
    });
  } catch (error) {
    console.error("Error generating response:", error);
    return toolResult || "I encountered an error processing your request.";
  }
}

/**
 * Main function
 */
async function main() {
  try {
    console.log("\n" + "=".repeat(70));
    console.log("🚀 TOOL-AUGMENTED RAG AGENT");
    console.log("=".repeat(70) + "\n");

    // Check for OpenRouter API key
    if (!OPENROUTER_API_KEY) {
      console.error("❌ OPENROUTER_API_KEY not found in .env file");
      console.log("\nPlease create a .env file with:");
      console.log("OPENROUTER_API_KEY=your_key_here");
      process.exit(1);
    }

    // Get prompt from command line arguments
    const prompt = process.argv.slice(2).join(" ");

    if (!prompt) {
      console.error("❌ Please provide a prompt");
      console.log('📝 Example: node main.js "What is the weather in Jos?"');
      process.exit(1);
    }

    console.log("📝 USER QUERY:", prompt);
    console.log("─".repeat(70));

    // Initialize stores
    await initializeStores();

    // Clear previous tool calls
    toolCalls = [];

    // Retrieve relevant conversations
    const relevantConversations = await retrieveConversations(prompt);

    // Execute tool
    console.log("\n🤔 Processing your request...\n");
    const toolResult = await selectAndExecuteTool(prompt);

    // Generate final response
    const finalResponse = await generateResponse(
      prompt,
      toolResult,
      relevantConversations,
    );

    // Save conversation
    await saveConversation(prompt, finalResponse);

    // Print results
    console.log("\n" + "=".repeat(70));
    console.log("📋 CONVERSATION HISTORY:");
    console.log("=".repeat(70));

    console.log(`\n👤 USER: ${prompt}`);

    if (toolCalls.length > 0) {
      for (const tool of toolCalls) {
        console.log(`\n🔧 TOOL (${tool.name}):`);
        console.log(`   Input: ${tool.input}`);
        console.log(`   Result: ${tool.result}`);
      }
    }

    console.log(`\n🤖 ASSISTANT: ${finalResponse}`);

    console.log("\n" + "=".repeat(70));
    console.log("✅ FINAL ANSWER:");
    console.log("=".repeat(70));
    console.log(finalResponse);
    console.log("=".repeat(70) + "\n");
  } catch (error) {
    console.error("\n❌ ERROR:", error);
    process.exit(1);
  }
}

// Run the main function
main().catch(console.error);
