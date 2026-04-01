import { NativeModules } from "react-native";
import axios from "axios";
import AdvancedSearch from "../utils/advancedSearch";
import ResponseFormatter from "../utils/responseFormatter";

class AIResponseService {
  constructor() {
    this.patterns = {
      greeting: /^(hi|hello|hey|good morning|good evening)$/i,
      farewell: /^(bye|goodbye|see you|farewell|exit|quit)$/i,
      question: /^(what|who|where|when|why|how)\s+.+/i,
      search: /^(search|find|lookup|tell me about)\s+.+/i,
      identity: /^who am i$/i,
    };
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;
    try {
      this.initialized = true;
      console.log("✅ AI Response Service initialized");
    } catch (error) {
      console.error("❌ Error initializing AI Response Service:", error);
      throw error;
    }
  }

  async generateResponse(message, context = []) {
    try {
      if (!message.trim()) return null;

      const text = message.toLowerCase().trim();
      console.log("AIService: Processing message:", text);
      let messageType = "unknown";

      // Determine message type using patterns
      for (const [type, pattern] of Object.entries(this.patterns)) {
        if (pattern.test(text)) {
          messageType = type;
          console.log("AIService: Detected message type:", messageType);
          break;
        }
      }

      // Handle different types of queries
      switch (messageType) {
        case "greeting":
          return this.generateGreeting();

        case "farewell":
          return "Goodbye! Take care and see you soon! 👋";

        case "question":
          // If it's a who/what/where question, treat it like a search
          const searchQuery = text
            .replace(
              /^(what|who|where|when|why|how)\s+(is|are|was|were)\s+/i,
              ""
            )
            .trim();
          return await this.handleSearch("search " + searchQuery);

        case "search":
          return await this.handleSearch(text);

        case "identity":
          return "Let me check my memory for your name...";

        default:
          // If the message seems like a topic or question, treat it as a search
          if (text.length > 3 && !text.match(/^(hi|hey|hello|bye|thanks?)$/i)) {
            return await this.handleSearch("search " + text);
          }
          return await this.generateContextualResponse(text, context);
      }
    } catch (error) {
      console.error("Error generating AI response:", error);
      return "I apologize, but I encountered an error processing that. Could you rephrase your message?";
    }
  }

  analyzeSentiment(text) {
    // Simple sentiment analysis based on positive/negative word patterns
    const positive =
      /(good|great|awesome|happy|excellent|wonderful|nice|love|like|thanks)/i;
    const negative =
      /(bad|terrible|awful|sad|hate|dislike|angry|upset|sorry|disappointed)/i;

    if (positive.test(text)) return 1;
    if (negative.test(text)) return -1;
    return 0;
  }

  async generateGreeting(sentiment) {
    const greetings = [
      "Hello! I'm your AI assistant, ready to provide comprehensive and accurate information on a wide range of topics. How may I assist you today?",
      "Greetings! I'm here to help you with detailed information, analysis, and answers to your questions. What would you like to explore?",
      "Welcome! I'm equipped to provide in-depth assistance across various subjects. How can I help you achieve your goals today?",
      "Hello! I'm here to offer professional assistance with any questions or topics you'd like to discuss. What can I help you understand better?",
    ];
    return ResponseFormatter.addSentimentEmoji(
      greetings[Math.floor(Math.random() * greetings.length)]
    );
  }

  generateFarewell(sentiment) {
    const farewells = [
      "Thank you for our interaction! I've enjoyed helping you today. Take care and don't hesitate to return if you need further assistance.",
      "Goodbye for now! It's been a pleasure assisting you. Feel free to return anytime for more detailed information or guidance.",
      "Thank you for engaging with me! I look forward to our next conversation where I can continue providing helpful insights and information.",
      "Until next time! Remember, I'm here whenever you need comprehensive assistance or have questions to explore.",
    ];
    return ResponseFormatter.formatChatResponse(
      farewells[Math.floor(Math.random() * farewells.length)]
    );
  }

  async handleQuestion(message) {
    try {
      // Remove question words and common stop words
      const stopWords = [
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
      ];
      const searchTerms = message
        .toLowerCase()
        .split(/\s+/)
        .filter(
          (word) =>
            word.length > 2 &&
            !stopWords.includes(word) &&
            !/[?"'.,]/.test(word)
        );

      if (searchTerms.length === 0) {
        return "Could you please provide more details about what you'd like to know?";
      }

      console.log("AIService: Question search terms:", searchTerms);
      const searchQuery = searchTerms.join(" ");
      return await this.searchAndSummarize(searchQuery);
    } catch (error) {
      console.error("Error handling question:", error);
      return "I'm having trouble finding information about that. Could you try asking in a different way?";
    }
  }

  async handleSearch(message) {
    try {
      const searchQuery = message
        .replace(/^(?:search|find|lookup|tell me about)\s+/i, "")
        .trim();
      console.log("AIService: Processing search for:", searchQuery);

      if (!searchQuery) {
        return "What would you like me to search for?";
      }

      // Add some conversational variety to the response introduction
      const intros = [
        "Here's what I know about",
        "Let me tell you about",
        "I can share this information about",
        "Here's some information about",
      ];
      const intro = intros[Math.floor(Math.random() * intros.length)];

      const result = await this.searchAndSummarize(searchQuery);
      console.log("AIService: Response generated successfully");

      return ResponseFormatter.formatSearchResponse(result, searchQuery);
    } catch (error) {
      console.error("Error handling search:", error);
      return ResponseFormatter.formatError(
        "I encountered an issue while searching. I can tell you about topics like Elon Musk, SpaceX, Tesla, AI, climate change, cryptocurrency, and COVID-19. What interests you?"
      );
    }
  }

  async searchAndSummarize(query, options = {}) {
    try {
      // Use advanced search with both knowledge base and news
      const searchResponse = await AdvancedSearch.search(query, {
        includeNews: true,
        includeKnowledgeBase: true,
        limit: 5,
        ...options,
      });

      if (!searchResponse.success) {
        throw new Error(searchResponse.error);
      }

      return AdvancedSearch.formatResults(searchResponse);
    } catch (error) {
      console.error("Error in searchAndSummarize:", error);
      return "I apologize, but I'm having trouble searching for that information right now.";
    }
  }

  async generateContextualResponse(message, context) {
    // Generate response based on message content and conversation context
    const sentiment = this.analyzeSentiment(message);

    // Create engaging responses based on sentiment and context
    const responses = {
      positive: [
        "I appreciate your enthusiasm. I'm here to provide comprehensive assistance and information on any topic you'd like to explore. What specific area would you like to delve into?",
        "Thank you for your positive engagement. I'm equipped to help you with detailed information, analysis, or answers to any questions you may have. How can I assist you today?",
        "I'm glad you're interested in learning more. As an AI assistant, I can provide in-depth information and insights across various topics. What would you like to explore?",
      ],
      negative: [
        "I understand your concern. Let me help you navigate this challenge with clear, step-by-step guidance. Could you provide more context about what you're trying to achieve?",
        "I recognize this might be complex or frustrating. Let's break this down systematically and find a solution together. What specific aspects are you finding challenging?",
        "I'm here to help address your concerns with detailed, practical solutions. Could you elaborate on the particular difficulties you're experiencing?",
      ],
      neutral: [
        "I'm here to provide comprehensive assistance and information. Whether you need detailed explanations, analysis, or practical guidance, I'm equipped to help. What topic interests you?",
        "I can offer in-depth information and insights across a wide range of subjects. From technical explanations to general knowledge, I'm here to help you understand and learn. What would you like to explore?",
        "As your AI assistant, I can provide detailed answers, analyze complex topics, or help you discover new information. How can I assist you in achieving your goals today?",
      ],
    };

    const category =
      sentiment > 0 ? "positive" : sentiment < 0 ? "negative" : "neutral";
    const responseArray = responses[category];
    const response =
      responseArray[Math.floor(Math.random() * responseArray.length)];
    return ResponseFormatter.formatChatResponse(response);
  }
}

export default AIResponseService;
