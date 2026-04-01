import React, { useState } from "react";
import { View, Text, StyleSheet, TouchableOpacity, Alert } from "react-native";
import Sidebar from "../components/Sidebar";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { colors as baseColors } from "../theme/colors";

export default function SettingsScreen() {
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [userId, setUserId] = useState("");

  // Dynamic theme colors
  const themeColors = darkMode
    ? {
        ...baseColors,
        background: "#181818",
        surface: "#222",
        textPrimary: "#fff",
        textSecondary: "#bbb",
        icon: "#fff",
      }
    : baseColors;

  React.useEffect(() => {
    (async () => {
      try {
        const id = await AsyncStorage.getItem("userId");
        setUserId(id || "");
      } catch {}
    })();
  }, []);

  return (
    <View style={[styles.container, { backgroundColor: themeColors.background }]}>
      {!sidebarVisible && (
        <TouchableOpacity
          onPress={() => setSidebarVisible(true)}
          style={styles.menuButton}
        >
          <Icon name="menu" size={24} color={themeColors.icon} />
        </TouchableOpacity>
      )}

      <Sidebar
        visible={sidebarVisible}
        onClose={() => setSidebarVisible(false)}
      />
      <View style={styles.titleContainer}>
        <Icon name="cog" size={24} color={themeColors.primary} />
        <Text style={[styles.title, { color: themeColors.primary }]}>Settings</Text>
      </View>

      {/* Theme toggle */}
      <View style={styles.settingRow}>
        <Text style={[styles.settingLabel, { color: themeColors.textSecondary }]}>Dark Mode</Text>
        <TouchableOpacity
          style={[styles.toggleBtn, { backgroundColor: themeColors.surface }]}
          onPress={() => setDarkMode((prev) => !prev)}
        >
          <Icon
            name={darkMode ? "weather-night" : "white-balance-sunny"}
            size={22}
            color={darkMode ? themeColors.primary : themeColors.textSecondary}
          />
        </TouchableOpacity>
      </View>

      {/* Notifications toggle */}
      <View style={styles.settingRow}>
        <Text style={[styles.settingLabel, { color: themeColors.textSecondary }]}>Notifications</Text>
        <TouchableOpacity
          style={[styles.toggleBtn, { backgroundColor: themeColors.surface }]}
          onPress={() => setNotificationsEnabled((prev) => !prev)}
        >
          <Icon
            name={notificationsEnabled ? "bell" : "bell-off"}
            size={22}
            color={notificationsEnabled ? themeColors.primary : themeColors.textSecondary}
          />
        </TouchableOpacity>
      </View>

      {/* Account info */}
      <View style={styles.settingRow}>
        <Text style={[styles.settingLabel, { color: themeColors.textSecondary }]}>Account</Text>
        <Text style={[styles.settingValue, { color: themeColors.textPrimary }]}>{userId || "Not logged in"}</Text>
      </View>

      {/* Logout button */}
      <TouchableOpacity
        style={[styles.logoutBtn, { backgroundColor: themeColors.surface, borderColor: themeColors.error }]}
        onPress={async () => {
          await AsyncStorage.removeItem("userId");
          await AsyncStorage.removeItem("token");
          setUserId("");
          Alert.alert("Logged out", "You have been logged out.");
        }}
      >
        <Icon name="logout" size={20} color={themeColors.error} />
        <Text style={[styles.logoutText, { color: themeColors.error }]}>Logout</Text>
      </TouchableOpacity>
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
    marginBottom: 15,
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: colors.primary,
    marginLeft: 8,
  },
  settingRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    marginBottom: 2,
  },
  settingLabel: {
    fontSize: 16,
    color: colors.textSecondary,
    fontWeight: "500",
  },
  settingValue: {
    fontSize: 16,
    color: colors.textPrimary,
    fontWeight: "600",
  },
  toggleBtn: {
    padding: 8,
    borderRadius: 8,
    backgroundColor: colors.surface,
  },
  logoutBtn: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "center",
    marginTop: 30,
    backgroundColor: colors.surface,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.error,
  },
  logoutText: {
    color: colors.error,
    fontWeight: "600",
    fontSize: 16,
    marginLeft: 8,
  },
});
