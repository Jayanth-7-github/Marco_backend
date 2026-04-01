import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from "react-native";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";
import Sidebar from "../components/Sidebar";
import AsyncStorage from "@react-native-async-storage/async-storage";

export default function ChatsScreen({ navigation }) {
  const [chats, setChats] = useState([]);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const [userId, setUserId] = useState(null);

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

  useEffect(() => {
    const load = async () => {
      if (!userId) return;
      try {
        const raw = await AsyncStorage.getItem(`chats_${userId}`);
        const chatList = raw ? JSON.parse(raw) : [];
        console.log("Loaded chats:", chatList); // Debug log
        setChats(chatList);
      } catch (e) {
        console.warn("Could not load chats", e);
      }
    };
    const unsubscribe = navigation.addListener("focus", load);
    load();
    return unsubscribe;
  }, [navigation, userId]); // Add userId as dependency

  const openChat = (chat) => {
    // navigate to Home and pass chat object
    navigation.navigate("Home", { openChatId: chat.id });
  };

  const startNew = () => {
    navigation.navigate("Home", { newChat: true });
  };

  const confirmDelete = (id) => {
    if (!userId) {
      Alert.alert("Error", "Please log in to delete chats");
      return;
    }
    console.log("Attempting to delete chat:", id); // Debug log
    Alert.alert("Delete chat", "Are you sure you want to delete this chat?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: () => {
          deleteChat(id);
        },
      },
    ]);
  };

  const deleteChat = async (id) => {
    if (!userId) {
      Alert.alert("Error", "Please log in to delete chats");
      return;
    }
    try {
      const storageKey = `chats_${userId}`;
      const raw = await AsyncStorage.getItem(storageKey);
      const list = raw ? JSON.parse(raw) : [];
      console.log("Before delete:", list); // Debug log
      const filtered = list.filter((c) => c.id !== id);
      console.log("After delete:", filtered); // Debug log

      if (filtered.length === list.length) {
        console.warn("No chat found with id:", id);
        return;
      }

      await AsyncStorage.setItem(storageKey, JSON.stringify(filtered));
      setChats(filtered);
      Alert.alert("Success", "Chat deleted successfully");
    } catch (e) {
      console.error("Could not delete chat", e);
      Alert.alert("Error", "Failed to delete chat");
    }
  };

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
        onLogout={() => {
          // optional: navigate reset handled by Sidebar; also clear any local state here
          setChats([]);
        }}
      />
      <View style={styles.titleContainer}>
        <Icon name="chat-outline" size={24} color={colors.primary} />
        <Text style={styles.title}>Chats</Text>
      </View>

      <TouchableOpacity style={styles.newButton} onPress={startNew}>
        <Icon name="plus" size={20} color={colors.buttonText} />
        <Text style={styles.newButtonText}>New Chat</Text>
      </TouchableOpacity>

      {chats.length === 0 ? (
        <Text style={styles.empty}>No saved chats yet — start a new one!</Text>
      ) : (
        <FlatList
          data={chats}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <View style={styles.chatRow}>
              <TouchableOpacity
                style={styles.chatBody}
                onPress={() => openChat(item)}
              >
                <Text style={styles.chatTitle}>{item.title || "Chat"}</Text>
                <Text numberOfLines={1} style={styles.chatPreview}>
                  {item.messages && item.messages.length > 0
                    ? item.messages[item.messages.length - 1].user || ""
                    : "No messages yet"}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.deleteBtn}
                onPress={() => confirmDelete(item.id)}
              >
                <Text style={{ color: colors.error }}>Delete</Text>
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
    paddingVertical: 6,
    paddingHorizontal: 10,
    elevation: 0,
    borderWidth: 0,
  },
  titleContainer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 15,
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: colors.primary,
    marginLeft: 8,
  },
  newButton: {
    backgroundColor: colors.primary,
    padding: 12,
    borderRadius: 8,
    alignSelf: "center",
    marginBottom: 15,
    flexDirection: "row",
    alignItems: "center",
  },
  newButtonText: {
    color: colors.buttonText,
    fontWeight: "600",
    marginLeft: 8,
  },
  empty: {
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: 20,
  },
  chatRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    marginHorizontal: -5,
    paddingHorizontal: 5,
  },
  chatBody: {
    flex: 1,
    paddingVertical: 4,
  },
  chatTitle: {
    fontWeight: "600",
    color: colors.textPrimary,
    fontSize: 15,
    marginBottom: 4,
  },
  chatPreview: {
    color: colors.textSecondary,
    fontSize: 14,
  },
  deleteBtn: {
    paddingHorizontal: 12,
    opacity: 0.8,
  },
});
