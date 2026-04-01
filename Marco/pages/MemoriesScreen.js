import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  Alert,
} from "react-native";
import Sidebar from "../components/Sidebar";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";

export default function MemoriesScreen() {
  const [memories, setMemories] = useState([]);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const [newMemory, setNewMemory] = useState("");
  const [userId, setUserId] = useState(null);

  // Load user ID first
  useEffect(() => {
    const loadUserId = async () => {
      try {
        const id = await AsyncStorage.getItem("userId");
        setUserId(id);
      } catch (e) {
        console.warn("Could not load user ID", e);
      }
    };
    loadUserId();
  }, []);

  // Load memories for the specific user
  useEffect(() => {
    const load = async () => {
      if (!userId) return;
      try {
        const stored = await AsyncStorage.getItem(`memories_${userId}`);
        if (stored) setMemories(JSON.parse(stored));
      } catch (e) {
        console.warn("Could not load memories", e);
      }
    };
    load();
  }, [userId]);

  return (
    <View style={styles.container}>
      {!sidebarVisible && (
        <TouchableOpacity
          onPress={() => setSidebarVisible(true)}
          style={styles.menuButton}
        >
          <Icon name="menu" size={24} color={colors.icon} />
        </TouchableOpacity>
      )}

      <Sidebar
        visible={sidebarVisible}
        onClose={() => setSidebarVisible(false)}
      />
      <View style={styles.titleContainer}>
        <Icon name="brain" size={24} color={colors.primary} />
        <Text style={styles.title}>Memories</Text>
      </View>
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={newMemory}
          onChangeText={setNewMemory}
          placeholder="Type a new memory..."
          placeholderTextColor={colors.textTertiary}
        />
        <TouchableOpacity
          style={[styles.button, !newMemory.trim() && styles.buttonDisabled]}
          disabled={!newMemory.trim()}
          onPress={async () => {
            if (!userId) {
              Alert.alert("Error", "Please log in to save memories");
              return;
            }
            try {
              const updatedMemories = [...memories, newMemory.trim()];
              await AsyncStorage.setItem(
                `memories_${userId}`,
                JSON.stringify(updatedMemories)
              );
              setMemories(updatedMemories);
              setNewMemory("");
              Alert.alert("Success", "Memory added successfully!");
            } catch (e) {
              Alert.alert("Error", "Failed to save memory");
            }
          }}
        >
          <Text style={styles.buttonText}>Add</Text>
        </TouchableOpacity>
      </View>

      {memories.length === 0 ? (
        <Text style={styles.empty}>
          No memories yet. Add your first memory above!
        </Text>
      ) : (
        <FlatList
          data={memories}
          keyExtractor={(item, idx) => String(idx)}
          renderItem={({ item, index }) => (
            <View style={styles.memoryItem}>
              <Text style={styles.memoryText}>
                {typeof item === "string"
                  ? item
                  : item.text || item.content || JSON.stringify(item)}
              </Text>
              <TouchableOpacity
                style={styles.deleteButton}
                onPress={async () => {
                  if (!userId) {
                    Alert.alert("Error", "Please log in to delete memories");
                    return;
                  }
                  try {
                    const updatedMemories = memories.filter(
                      (_, i) => i !== index
                    );
                    await AsyncStorage.setItem(
                      `memories_${userId}`,
                      JSON.stringify(updatedMemories)
                    );
                    setMemories(updatedMemories);
                    Alert.alert("Success", "Memory deleted successfully!");
                  } catch (e) {
                    Alert.alert("Error", "Failed to delete memory");
                  }
                }}
              >
                <Icon name="delete-outline" size={20} color={colors.error} />
              </TouchableOpacity>
            </View>
          )}
        />
      )}
    </View>
  );
}

import { colors } from "../theme/colors";

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: colors.background,
    paddingTop: 60,
  },
  menuButton: {
    position: "absolute",
    top: 60,
    left: 20,
    zIndex: 20,
    backgroundColor: "transparent",
    borderRadius: 25,
    paddingVertical: 8,
    paddingHorizontal: 12,
    elevation: 0,
    borderWidth: 0,
  },
  titleContainer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 25,
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: colors.primary,
    marginLeft: 8,
  },
  inputContainer: {
    flexDirection: "row",
    marginBottom: 20,
    gap: 12,
  },
  input: {
    flex: 1,
    backgroundColor: colors.surface,
    padding: 14,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    color: colors.textPrimary,
    fontSize: 15,
  },
  button: {
    backgroundColor: colors.primary,
    paddingHorizontal: 20,
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 8,
  },
  buttonDisabled: {
    backgroundColor: colors.disabled,
  },
  buttonText: {
    color: colors.buttonText,
    fontWeight: "600",
  },
  empty: {
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: 25,
    fontSize: 15,
  },
  memoryItem: {
    backgroundColor: colors.surface,
    padding: 16,
    borderRadius: 10,
    marginBottom: 12,
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.border,
  },
  memoryText: {
    flex: 1,
    color: colors.textPrimary,
    fontSize: 15,
    lineHeight: 20,
  },
  deleteButton: {
    padding: 8,
    marginLeft: 8,
  },
  deleteButtonText: {
    fontSize: 18,
    color: colors.error,
  },
});
