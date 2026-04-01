import AsyncStorage from "@react-native-async-storage/async-storage";

export class EnhancedMemorySystem {
  constructor() {
    this.shortTermMemory = new Map();
    this.longTermMemory = new Map();
    this.MEMORY_KEY = "enhanced_memory";
    this.MAX_SHORT_TERM = 50;
  }

  // Initialize memory from storage
  async initialize() {
    try {
      const stored = await AsyncStorage.getItem(this.MEMORY_KEY);
      if (stored) {
        this.longTermMemory = new Map(JSON.parse(stored));
      }
    } catch (error) {
      console.error("Error initializing memory:", error);
    }
  }

  // Save memory to persistent storage
  async persistMemory() {
    try {
      await AsyncStorage.setItem(
        this.MEMORY_KEY,
        JSON.stringify(Array.from(this.longTermMemory.entries()))
      );
    } catch (error) {
      console.error("Error persisting memory:", error);
    }
  }

  // Add a memory with metadata
  async addMemory(type, content, metadata = {}) {
    const timestamp = Date.now();
    const memory = {
      type,
      content,
      metadata: {
        ...metadata,
        created: timestamp,
        lastAccessed: timestamp,
        accessCount: 0,
      },
    };

    // Add to short-term first
    this.shortTermMemory.set(timestamp, memory);

    // If short-term is full, move oldest to long-term
    if (this.shortTermMemory.size > this.MAX_SHORT_TERM) {
      this.consolidateMemory();
    }

    await this.persistMemory();
    return timestamp;
  }

  // Move memories from short-term to long-term based on importance
  async consolidateMemory() {
    const memories = Array.from(this.shortTermMemory.entries());
    memories.sort((a, b) => {
      const scoreA = this.calculateImportance(a[1]);
      const scoreB = this.calculateImportance(b[1]);
      return scoreB - scoreA;
    });

    // Move important memories to long-term
    for (const [id, memory] of memories) {
      if (this.calculateImportance(memory) > 0.5) {
        this.longTermMemory.set(id, memory);
      }
      this.shortTermMemory.delete(id);
    }

    await this.persistMemory();
  }

  // Calculate importance score for memory consolidation
  calculateImportance(memory) {
    const age = (Date.now() - memory.metadata.created) / (1000 * 60 * 60 * 24); // Age in days
    const recency =
      (Date.now() - memory.metadata.lastAccessed) / (1000 * 60 * 60); // Hours since last access
    const frequency = memory.metadata.accessCount;

    // Importance formula considering age, recency, and access frequency
    return (
      0.3 * (1 / (age + 1)) + // Newer memories are important
      0.3 * (1 / (recency + 1)) + // Recently accessed memories are important
      0.4 * Math.min(frequency / 10, 1)
    ); // Frequently accessed memories are important
  }

  // Semantic search through memories
  async searchMemories(query, type = null) {
    const results = [];
    const searchStr = query.toLowerCase();

    // Search both short-term and long-term memories
    for (const memory of [
      ...this.shortTermMemory.values(),
      ...this.longTermMemory.values(),
    ]) {
      if (type && memory.type !== type) continue;

      const content = memory.content.toLowerCase();
      const score = this.calculateRelevance(searchStr, content);

      if (score > 0.3) {
        // Minimum relevance threshold
        results.push({
          memory,
          relevance: score,
        });
      }
    }

    // Sort by relevance
    results.sort((a, b) => b.relevance - a.relevance);
    return results.map((r) => r.memory);
  }

  // Calculate relevance score for search
  calculateRelevance(query, content) {
    const queryTerms = query.split(" ");
    let matchCount = 0;

    for (const term of queryTerms) {
      if (content.includes(term)) matchCount++;
    }

    return matchCount / queryTerms.length;
  }

  // Get all memories of a specific type
  async getMemoriesByType(type) {
    const memories = [];

    for (const memory of [
      ...this.shortTermMemory.values(),
      ...this.longTermMemory.values(),
    ]) {
      if (memory.type === type) {
        memories.push(memory);
      }
    }

    return memories;
  }

  // Clear a specific memory by ID
  async clearMemory(id) {
    this.shortTermMemory.delete(id);
    this.longTermMemory.delete(id);
    await this.persistMemory();
  }

  // Clear all memories
  async clearMemories() {
    this.shortTermMemory.clear();
    this.longTermMemory.clear();
    await AsyncStorage.removeItem(this.MEMORY_KEY);
  }
}

export default EnhancedMemorySystem;
