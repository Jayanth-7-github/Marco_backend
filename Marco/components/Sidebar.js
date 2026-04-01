import React, { useRef, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
  Alert,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";

const { width } = Dimensions.get("window");
const DRAWER_WIDTH = width * 0.7; // 70% of screen width
const BACKEND_URL = "https://marco-backend-u19w.onrender.com";

export default function Sidebar({ visible, onClose, onLogout }) {
  const navigation = useNavigation();
  const slideAnim = useRef(new Animated.Value(-DRAWER_WIDTH)).current;

  // 🔹 Animate Sidebar open/close
  useEffect(() => {
    Animated.timing(slideAnim, {
      toValue: visible ? 0 : -DRAWER_WIDTH,
      duration: 300,
      useNativeDriver: true,
    }).start();
  }, [visible]);

  // 🔹 Updated logout logic
  const handleLogout = async () => {
    console.log("🚪 Logout initiated...");
    try {
      const userId = await AsyncStorage.getItem("userId");
      console.log("🧠 Retrieved userId from AsyncStorage:", userId);

      // Inform backend that user logged out (optional)
      if (userId) {
        console.log("📡 Sending logout request to backend...");
        try {
          const response = await fetch(`${BACKEND_URL}/logout`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ userId }),
          });
          console.log("🌐 Backend response status:", response.status);
          if (response.ok) {
            Alert.alert("Logout", "You have been logged out successfully.");
          } else {
            const errorText = await response.text();
            console.log("❌ Logout failed. Server response:", errorText);
            Alert.alert(
              "Error",
              "Logout processed locally but server reported an issue."
            );
          }
        } catch (err) {
          console.log("🔥 Network error during backend logout:", err);
        }
      }

      // Clear local storage (userId + token)
      await AsyncStorage.removeItem("userId");
      await AsyncStorage.removeItem("token");

      // Notify parent (App) to clear its state
      if (onLogout) {
        try {
          onLogout();
        } catch (e) {
          console.warn("onLogout error", e);
        }
      }
    } catch (error) {
      console.log("🔥 Logout error caught:", error);
      Alert.alert("Error", "Network or server issue during logout.");
      if (onLogout) {
        try {
          onLogout();
        } catch (e) {
          console.warn("onLogout error", e);
        }
      }
    }
    // Only close sidebar if visible
    if (onClose && visible) onClose();
  };

  return (
    <>
      {/* 🩵 Dim background overlay */}
      {visible && (
        <TouchableOpacity
          style={styles.overlay}
          activeOpacity={1}
          onPress={onClose}
        />
      )}

      {/* 🧭 Animated drawer */}
      <Animated.View
        style={[styles.sidebar, { transform: [{ translateX: slideAnim }] }]}
      >
        <View style={styles.headerRow}>
          <View style={styles.titleContainer}>
            <Icon name="lightning-bolt" size={24} color={colors.primary} />
            <Text style={styles.title}>Marco Menu</Text>
          </View>
          <TouchableOpacity
            style={styles.newChatBtn}
            onPress={() => {
              navigation.navigate("Home", { newChat: true });
              onClose?.();
            }}
          >
            <Icon name="plus" size={16} color={colors.buttonText} />
            <Text style={styles.newChatText}>New</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => {
            navigation.navigate("Home");
            onClose?.();
          }}
        >
          <Icon
            name="home"
            size={20}
            color={colors.icon}
            style={styles.menuIcon}
          />
          <Text style={styles.menuText}>Home</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => {
            navigation.navigate("Memories");
            onClose?.();
          }}
        >
          <Icon
            name="brain"
            size={20}
            color={colors.icon}
            style={styles.menuIcon}
          />
          <Text style={styles.menuText}>Memories</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => {
            navigation.navigate("Chats");
            onClose?.();
          }}
        >
          <Icon
            name="chat-outline"
            size={20}
            color={colors.icon}
            style={styles.menuIcon}
          />
          <Text style={styles.menuText}>Chats</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => {
            navigation.navigate("Settings");
            onClose?.();
          }}
        >
          <Icon
            name="cog"
            size={20}
            color={colors.icon}
            style={styles.menuIcon}
          />
          <Text style={styles.menuText}>Settings</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => {
            navigation.navigate("Profile");
            onClose?.();
          }}
        >
          <Icon
            name="account"
            size={20}
            color={colors.icon}
            style={styles.menuIcon}
          />
          <Text style={styles.menuText}>Profile</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.menuItem, { marginTop: 20 }]}
          onPress={handleLogout}
        >
          <Icon
            name="logout"
            size={20}
            color={colors.error}
            style={styles.menuIcon}
          />
          <Text style={[styles.menuText, { color: colors.error }]}>Logout</Text>
        </TouchableOpacity>
      </Animated.View>
    </>
  );
}

import { colors } from "../theme/colors";

const styles = StyleSheet.create({
  sidebar: {
    position: "absolute",
    top: 0,
    left: 0,
    width: DRAWER_WIDTH,
    height: Dimensions.get("window").height,
    backgroundColor: colors.surface,
    padding: 20,
    paddingTop: 60,
    elevation: 20,
    zIndex: 9999,
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.13,
    shadowRadius: 18,
  },
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: Dimensions.get("window").height,
    backgroundColor: "rgba(0,0,0,0.5)",
    zIndex: 9998,
  },
  titleContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: colors.primary,
    marginBottom: 0,
    marginLeft: 8,
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 20,
  },
  newChatBtn: {
    backgroundColor: colors.primary,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    flexDirection: "row",
    alignItems: "center",
  },
  newChatText: {
    color: colors.buttonText,
    fontWeight: "600",
    fontSize: 13,
    marginLeft: 4,
  },
  menuItem: {
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    flexDirection: "row",
    alignItems: "center",
  },
  menuIcon: {
    marginRight: 12,
  },
  menuText: {
    fontSize: 16,
    color: colors.textPrimary,
  },
});
