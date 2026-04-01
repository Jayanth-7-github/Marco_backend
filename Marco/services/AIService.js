// AIService.js
import NewsService from "./news_service";
import { processUserInput } from "../marcoCore";

class AIService {
  constructor() {
    // 🔹 Gemini configuration
    this.model = {
      name: "Gemini Pro",
      baseUrl:
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
      requiresKey: true,
    };

    // 🔹 API key storage
    this.apiKey = null;
    console.log("Initializing Multi-Model AI Service...");
  }

  // 🔹 Set API key
  setApiKey(key) {
    this.apiKey = key;
    return true;
  }

  // 🔹 Get headers
  getHeaders() {
    return {
      "Content-Type": "application/json",
      "x-goog-api-key": this.apiKey,
    };
  }

  // 🔹 Main chat handler
  async chat(message, context = []) {
    if (!this.apiKey) {
      // If no API key, return a response indicating we're using local processing
      return {
        content: await processUserInput(message, context),
        role: "assistant",
        model: "local",
      };
    }

    const systemPrompt = `You are Marco, a highly knowledgeable AI assistant powered by ${this.model.name}. 
You provide detailed, accurate, and helpful responses while maintaining a friendly tone.`;

    try {
      const fullContext = `${systemPrompt}\n\nPrevious conversation:\n${context
        .map((msg) => `${msg.role}: ${msg.content}`)
        .join("\n")}\n\nUser: ${message}`;

      const response = await fetch(this.model.baseUrl, {
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify({
          contents: [{ parts: [{ text: fullContext }] }],
        }),
      });

      const data = await response.json();
      if (!response.ok)
        throw new Error(data.error?.message || "Gemini API error");

      const content =
        data?.candidates?.[0]?.content?.parts?.[0]?.text ||
        "No response from Gemini.";

      return {
        content,
        role: "assistant",
        model: "gemini",
      };
    } catch (error) {
      console.error(`${this.model.name} API Error:`, error);
      throw new Error(
        error.message || `${this.model.name} failed to generate a response.`
      );
    }
  }

  // 🔹 Check if API key exists
  hasApiKey() {
    return Boolean(this.apiKey);
  }

  async isRunning() {
    return this.hasApiKey();
  }

  // 🔹 Post-process responses (for memory, search, or recall logic)
  async processResponse(userMessage, aiResponse) {
    const userMessageLower = userMessage?.toLowerCase() || "";
    const responseText = aiResponse?.toLowerCase() || "";

    // Handle search commands
    if (userMessageLower.startsWith("search ")) {
      try {
        const searchResults = await this.handleSearchCommand(userMessageLower);
        return {
          content: searchResults.content,
          memory: null,
          type: "search",
          metadata: searchResults.metadata,
        };
      } catch (error) {
        console.error("Search error:", error);
        return {
          content: "I encountered an error while searching. Please try again.",
          memory: null,
          type: "error",
        };
      }
    }

    // Handle memory commands
    if (
      responseText.includes("remember:") ||
      responseText.includes("i will remember")
    ) {
      const fact = aiResponse
        .split(/remember:|i will remember/i)
        .pop()
        .trim();
      return {
        content: aiResponse,
        memory: fact,
        type: "memory",
      };
    }

    return {
      content: aiResponse,
      memory: null,
      type: "normal",
    };
  }

  // 🔹 Handle search commands using NewsService
  async handleSearchCommand(query) {
    try {
      // Remove 'search' from the beginning of the query
      const searchQuery = query.replace(/^search\s+/i, "");

      // For product searches
      if (
        searchQuery.toLowerCase().includes("amazon") ||
        searchQuery.toLowerCase().includes("product")
      ) {
        const productQuery = searchQuery
          .replace(/amazon|products?/gi, "")
          .trim();
        const results = await NewsService.searchNews(
          `${productQuery} product review`
        );

        const formattedResults = results.success
          ? NewsService.formatArticles(results.articles)
          : `Sorry, I couldn't find product information about "${productQuery}". ${results.error}`;

        return {
          content: formattedResults,
          metadata: {
            type: "product",
            query: productQuery,
            articles: results.success ? results.articles : [],
          },
        };
      }

      // For general searches
      const results = await NewsService.searchNews(searchQuery);
      const formattedResults = results.success
        ? NewsService.formatArticles(results.articles)
        : `Sorry, I couldn't find information about "${searchQuery}". ${results.error}`;

      return {
        content: formattedResults,
        metadata: {
          type: "general",
          query: searchQuery,
          articles: results.success ? results.articles : [],
        },
      };
    } catch (error) {
      console.error("Search command error:", error);
      throw error;
    }
  }

  // 🔹 Format search results for display
  formatSearchResults(results) {
    if (!results || !results.content) {
      return {
        text: `❌ Search Error: No results found`,
        links: [],
      };
    }

    // Extract links from the content
    const links = [];
    const lines = results.content.split("\n");
    const formattedLines = lines
      .map((line) => {
        if (line.startsWith("🔗")) {
          const url = line.replace("🔗 ", "").trim();
          links.push(url);
          return null; // Remove URL line since we'll handle it separately
        }
        return line;
      })
      .filter(Boolean); // Remove null entries

    return {
      text: `Search Results:\n\n${formattedLines.join("\n")}`,
      links: links,
    };
  }
}

export default new AIService();
