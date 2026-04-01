// ChatService.js
import AsyncStorage from "@react-native-async-storage/async-storage";
import AIService from "./AIService";
import { processUserInput } from "../marcoCore";
import MemoryManager from "../components/MemoryManager";

class ChatService {
  constructor() {
    this.memoryManager = new MemoryManager();
    this.chatContext = [];
    this.userId = "guest";
    this.chats = new Map(); // Store multiple chats in memory
  }

  async initialize() {
    await this.memoryManager.initialize();
    await this.loadUserId();
    await this.loadApiKeys();
  }

  async loadApiKeys() {
    try {
      const savedKeys = await AsyncStorage.getItem("ai_api_keys");
      if (savedKeys) {
        const keys = JSON.parse(savedKeys);
        Object.entries(keys).forEach(([model, key]) => {
          if (key) AIService.setApiKey(model, key);
        });
      }
    } catch (error) {
      console.error("Error loading API keys:", error);
      throw new Error("Failed to load API keys");
    }
  }

  async saveApiKey(model, key) {
    try {
      const savedKeys = (await AsyncStorage.getItem("ai_api_keys")) || "{}";
      const keys = JSON.parse(savedKeys);
      keys[model] = key;
      await AsyncStorage.setItem("ai_api_keys", JSON.stringify(keys));
      AIService.setApiKey(model, key);
    } catch (error) {
      console.error("Error saving API key:", error);
      throw new Error("Failed to save API key");
    }
  }

  async loadUserId(propUserId = null) {
    // Use the userId from props if available, otherwise load from storage
    const id = propUserId || (await AsyncStorage.getItem("userId"));
    if (!id) {
      throw new Error("No user ID found");
    }
    this.userId = id;
    return id;
  }

  async loadMemories() {
    return await this.memoryManager.loadMemories();
  }

  async saveMemory(fact) {
    return await this.memoryManager.saveMemory(fact);
  }

  async clearMemories() {
    return await this.memoryManager.clearMemories();
  }

  async loadAllChats(userId) {
    try {
      const storageKey = `chats_${userId}`;
      const raw = await AsyncStorage.getItem(storageKey);
      const list = raw ? JSON.parse(raw) : [];
      return list;
    } catch (error) {
      console.error("Error loading chats:", error);
      throw new Error("Failed to load chats");
    }
  }

  async loadPreviousChat(userId) {
    try {
      const list = await this.loadAllChats(userId);
      if (list.length > 0) {
        const mostRecent = list[list.length - 1];
        this.chats.set(mostRecent.id, mostRecent.messages || []);
        return {
          messages: mostRecent.messages || [],
          id: mostRecent.id,
        };
      }
      return null;
    } catch (error) {
      console.error("Error loading previous chat:", error);
      throw new Error("Failed to load previous chat");
    }
  }

  async loadChatById(userId, chatId) {
    const storageKey = `chats_${userId}`;
    const raw = await AsyncStorage.getItem(storageKey);
    const list = raw ? JSON.parse(raw) : [];
    const found = list.find((c) => c.id === chatId && c.userId === userId);
    return found ? { messages: found.messages || [], id: found.id } : null;
  }

  async persistChat(userId, chatId, messages) {
    if (!messages || messages.length === 0 || !userId) return;

    const storageKey = `chats_${userId}`;
    const raw = await AsyncStorage.getItem(storageKey);
    const list = raw ? JSON.parse(raw) : [];
    let newChatId = chatId;

    if (chatId) {
      const idx = list.findIndex((c) => c.id === chatId);
      if (idx !== -1) {
        list[idx] = {
          ...list[idx],
          messages,
          title: list[idx].title || messages[0]?.user || "Chat",
          userId,
        };
      } else {
        list.push({
          id: chatId,
          title: messages[0]?.user || "Chat",
          messages,
          userId,
        });
      }
    } else {
      newChatId = `chat-${Date.now()}-${Math.random()
        .toString(36)
        .slice(2, 8)}`;
      list.push({
        id: newChatId,
        title: messages[0]?.user || `Chat ${new Date().toLocaleString()}`,
        messages,
        userId,
      });
    }

    await AsyncStorage.setItem(storageKey, JSON.stringify(list));
    return newChatId;
  }

  async processUserInput(userText, memories = []) {
    try {
      if (!userText || typeof userText !== "string") {
        throw new Error("Invalid input: message must be a non-empty string");
      }

      // Create memory context
      const memoryContext = memories.map((memory) => ({
        role: "system",
        content: `Known fact: ${memory}`,
      }));

      // Add new message to context
      const newMessage = { role: "user", content: userText };
      const updatedContext = [...this.chatContext, newMessage];

      let response;
      try {
        // Get response from AI service (it will handle fallback to local processing)
        response = await AIService.chat(userText, [
          ...memoryContext,
          ...updatedContext,
        ]);

        // Update chat context
        this.chatContext = updatedContext.concat({
          role: "assistant",
          content: response.content,
        });
      } catch (aiError) {
        console.error("AI/Local processing error:", aiError);
        // Fallback to local processing if AI fails
        const localResponse = await processUserInput(userText, updatedContext);
        response = {
          content: localResponse,
          model: "local",
        };
      }

      // Handle memory-related commands and search results
      let memory = null;
      let links = [];
      try {
        const processedResponse = await AIService.processResponse(
          userText,
          response.content
        );
        memory = processedResponse.memory;

        // Check if this is a search response with links
        if (
          processedResponse.type === "search" &&
          processedResponse.metadata?.articles
        ) {
          links = processedResponse.metadata.articles.map(
            (article) => article.url
          );
          response.content = processedResponse.content;
          response.links = links;
        }
      } catch (memoryError) {
        console.error("Memory processing error:", memoryError);
      }

      return {
        response,
        memory,
        updatedContext: this.chatContext,
      };
    } catch (error) {
      console.error("Error processing input:", error);
      throw new Error(`Failed to process message: ${error.message}`);
    }
  }

  async deleteChat(userId, chatId) {
    try {
      const list = await this.loadAllChats(userId);
      const updatedList = list.filter((chat) => chat.id !== chatId);
      const storageKey = `chats_${userId}`;
      await AsyncStorage.setItem(storageKey, JSON.stringify(updatedList));
      this.chats.delete(chatId);
    } catch (error) {
      console.error("Error deleting chat:", error);
      throw new Error("Failed to delete chat");
    }
  }

  async clearAllChats(userId) {
    try {
      const storageKey = `chats_${userId}`;
      await AsyncStorage.setItem(storageKey, JSON.stringify([]));
      this.chats.clear();
    } catch (error) {
      console.error("Error clearing chats:", error);
      throw new Error("Failed to clear chats");
    }
  }
}

export default new ChatService();
