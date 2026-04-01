import React from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Crypto from "expo-crypto";
import EnhancedMemorySystem from "../utils/enhancedMemory";

const API_URL = "https://marco-backend-u19w.onrender.com";

// One-way encryption for cloud memory
const encryptText = async (text) => {
  return await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    text
  );
};

class MemoryManager {
  constructor() {
    this.enhancedMemory = new EnhancedMemorySystem();
    this.initialize();
    this.userId = null;
  }

  async initialize() {
    this.userId = await AsyncStorage.getItem("userId");
    await this.enhancedMemory.initialize();
  }

  async changeUser(userId) {
    this.userId = userId;
    await this.clearLocalData();
    await this.initialize();
  }

  async clearLocalData() {
    // Clear all user-specific data
    const keysToRemove = [
      "memories",
      "chats",
      "learned_patterns",
      this.enhancedMemory.MEMORY_KEY,
    ];

    for (const key of keysToRemove) {
      try {
        await AsyncStorage.removeItem(`${this.userId}_${key}`);
      } catch (e) {
        console.warn(`Failed to remove ${key}:`, e);
      }
    }
  }

  async loadMemories() {
    try {
      const token = await AsyncStorage.getItem("token");

      if (!token) {
        console.warn("⚠️ No token found, please log in again.");
        return [];
      }

      console.log("📡 Fetching memories...");

      // Get memories from enhanced system
      const memories = await this.enhancedMemory.getMemoriesByType("all");
      if (memories.length > 0) {
        console.log("✅ Loaded local memories:", memories);
        return memories.map((m) => m.content);
      }

      // Fallback: fetch from backend (encrypted content)
      const response = await fetch(`${API_URL}/memory`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        console.log("🔒 Failed to load memories:", response.status);
        return [];
      }

      const data = await response.json();
      if (data?.memories) {
        const memoryList = data.memories
          .map((m) => m.content || "")
          .filter(Boolean);
        console.log("✅ Loaded (encrypted) memories from backend:", memoryList);
        return memoryList;
      }
      return [];
    } catch (error) {
      console.error("❌ Error loading memories:", error);
      return [];
    }
  }

  async saveMemory(fact) {
    try {
      const encrypted = await encryptText(fact);
      const response = await fetch(`${API_URL}/memory`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${await AsyncStorage.getItem("token")}`,
        },
        body: JSON.stringify({ type: "fact", content: encrypted }),
      });

      if (!response.ok) {
        console.error("❌ Failed to save memory:", response.status);
        return false;
      }

      await response.json();

      // Persist plaintext locally for reasoning with user-specific storage
      try {
        const key = `${this.userId}_memories`;
        const local = await AsyncStorage.getItem(key);
        const arr = local ? JSON.parse(local) : [];
        arr.push(fact);
        await AsyncStorage.setItem(key, JSON.stringify(arr));
      } catch (e) {
        console.warn("Could not persist local memory:", e);
      }

      console.log("✅ Memory saved:", fact);
      return true;
    } catch (error) {
      console.error("❌ Error saving memory:", error);
      return false;
    }
  }

  async clearMemories() {
    try {
      const token = await AsyncStorage.getItem("token");
      await fetch(`${API_URL}/memory`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });

      try {
        await AsyncStorage.removeItem("memories");
      } catch (e) {
        // ignore
      }

      console.log("🧹 Cleared all memories!");
      return true;
    } catch (error) {
      console.error("❌ Error clearing memories:", error);
      return false;
    }
  }
}

export default MemoryManager;
