import React, { useState, useEffect } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { View, StatusBar, Animated, Dimensions } from "react-native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import AsyncStorage from "@react-native-async-storage/async-storage";
import HomeScreen from "./pages/HomeScreen";
import LoginScreen from "./pages/LoginScreen";
import MemoriesScreen from "./pages/MemoriesScreen";
import SettingsScreen from "./pages/SettingsScreen";
import ChatsScreen from "./pages/ChatsScreen";
import ProfileScreen from "./pages/ProfileScreen";

const Stack = createNativeStackNavigator();

import { colors } from "./theme/colors";

export default function App() {
  const [userId, setUserId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [progressAnim] = useState(new Animated.Value(0));
  const screenWidth = Dimensions.get("window").width;

  // Progress bar animation
  const startProgressAnimation = () => {
    Animated.timing(progressAnim, {
      toValue: screenWidth,
      duration: 1500,
      useNativeDriver: true,
    }).start(() => {
      setLoading(false);
    });
  };

  // 🧠 Load user session from AsyncStorage
  useEffect(() => {
    const loadUserSession = async () => {
      try {
        const storedId = await AsyncStorage.getItem("userId");
        if (storedId) {
          console.log("🔁 Restored user session:", storedId);
          setUserId(storedId);
        }
        // Start progress animation after data is loaded
        startProgressAnimation();
      } catch (err) {
        console.error("❌ Failed to load session:", err);
        startProgressAnimation();
      }
    };
    loadUserSession();
  }, []);

  // 🔑 Handle successful login (now accepts optional token)
  const handleLogin = async (id, token) => {
    try {
      console.log("✅ Logged in as:", id);
      setUserId(id);
      await AsyncStorage.setItem("userId", id);
      if (token) {
        await AsyncStorage.setItem("token", token);
        console.log("🔐 Token saved to AsyncStorage");
      }
    } catch (err) {
      console.error("❌ Error saving login:", err);
    }
  };

  // 🚪 Handle logout
  const handleLogout = async () => {
    try {
      console.log("🔒 Logging out...");
      await AsyncStorage.removeItem("userId");
      await AsyncStorage.removeItem("token");
      setUserId(null);
      console.log("✅ Logout successful");
    } catch (err) {
      console.error("❌ Logout Error:", err);
    }
  };

  if (loading) {
    return (
      <View style={{ flex: 1, backgroundColor: colors.background }}>
        <StatusBar
          barStyle="light-content"
          backgroundColor={colors.background}
        />
        <Animated.View
          style={{
            width: 120,
            height: 3,
            backgroundColor: colors.primary,
            transform: [{ translateX: progressAnim }],
            position: "absolute",
            top: 0,
            left: -120, // Start off-screen
          }}
        />
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: colors.background }}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background} />
      <NavigationContainer>
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          {!userId ? (
            <Stack.Screen name="Login">
              {(props) => <LoginScreen {...props} onLogin={handleLogin} />}
            </Stack.Screen>
          ) : (
            <>
              <Stack.Screen name="Home">
                {(props) => (
                  <HomeScreen
                    {...props}
                    userId={userId}
                    onLogout={handleLogout}
                  />
                )}
              </Stack.Screen>

              <Stack.Screen name="Memories" component={MemoriesScreen} />
              <Stack.Screen name="Chats">
                {(props) => <ChatsScreen {...props} onLogout={handleLogout} />}
              </Stack.Screen>
              <Stack.Screen name="Settings" component={SettingsScreen} />
              <Stack.Screen name="Profile">
                {(props) => (
                  <ProfileScreen {...props} onLogout={handleLogout} />
                )}
              </Stack.Screen>
            </>
          )}
        </Stack.Navigator>
      </NavigationContainer>
    </View>
  );
}
