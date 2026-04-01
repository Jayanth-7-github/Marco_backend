import React, { useState, useEffect } from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Sidebar from "../components/Sidebar";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";

export default function ProfileScreen({ onLogout }) {
  const [username, setUsername] = useState("");
  const [sidebarVisible, setSidebarVisible] = useState(false);

  useEffect(() => {
    const loadProfile = async () => {
      try {
        const userId = await AsyncStorage.getItem("userId");
        const storedUsername = await AsyncStorage.getItem("username");
        if (userId) {
          setUsername(storedUsername || userId);
        }
      } catch (e) {
        console.warn("Could not load profile:", e);
      }
    };
    loadProfile();
  }, []);

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
        onLogout={onLogout}
      />

      <View style={styles.titleContainer}>
        <Icon name="account" size={24} color={colors.primary} />
        <Text style={styles.title}>Profile</Text>
      </View>

      <View style={styles.card}>
        <View style={styles.row}>
          <Text style={styles.label}>Username</Text>
          <Text style={styles.value}>{username || "Loading..."}</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Account Type</Text>
          <Text style={styles.value}>Standard</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Status</Text>
          <Text style={[styles.value, { color: "#4CAF50" }]}>Active</Text>
        </View>
      </View>
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
  card: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: 20,
    elevation: 2,
    borderWidth: 1,
    borderColor: colors.border,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  label: {
    fontSize: 16,
    color: colors.textSecondary,
    fontWeight: "500",
  },
  value: {
    fontSize: 16,
    color: colors.textPrimary,
    fontWeight: "600",
  },
});
