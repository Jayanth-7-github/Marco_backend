import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Alert,
  TouchableOpacity,
  Animated,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Ionicons } from "@expo/vector-icons";
import { colors } from "../theme/colors";

const API_URL = "https://marco-backend-u19w.onrender.com";
const BRAND_NAME = "Marco";

export default function LoginScreen({ onLogin }) {
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [isNew, setIsNew] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [cardAnim] = useState(new Animated.Value(0));
  const [buttonAnim] = useState(new Animated.Value(1));
  const [loadingSpinValue] = useState(new Animated.Value(0));
  const passwordInput = React.useRef();

  // Animate card in on mount
  React.useEffect(() => {
    Animated.timing(cardAnim, {
      toValue: 1,
      duration: 600,
      useNativeDriver: true,
    }).start();
  }, []);

  // Loading animation
  React.useEffect(() => {
    if (isLoading) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(loadingSpinValue, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      ).start();

      // Animate button scale
      Animated.sequence([
        Animated.timing(buttonAnim, {
          toValue: 0.95,
          duration: 150,
          useNativeDriver: true,
        }),
        Animated.timing(buttonAnim, {
          toValue: 1,
          duration: 150,
          useNativeDriver: true,
        }),
      ]).start();
    } else {
      loadingSpinValue.setValue(0);
    }
  }, [isLoading]);

  const handleAuth = async () => {
    if (!userId || !password) {
      Alert.alert("Please enter both user ID and password.");
      return;
    }
    setIsLoading(true);
    try {
      const endpoint = isNew ? "/register" : "/login";
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId, password }),
      });
      const data = await response.json();
      if (response.ok) {
        await AsyncStorage.setItem("userId", userId);
        if (data.token) {
          await AsyncStorage.setItem("token", data.token);
        }
        if (isNew && !data.token) {
          const loginRes = await fetch(`${API_URL}/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ userId, password }),
          });
          const loginData = await loginRes.json();
          if (loginRes.ok && loginData.token) {
            await AsyncStorage.setItem("token", loginData.token);
            onLogin(userId, loginData.token);
            Alert.alert("✅ Account created and logged in!");
            return;
          }
        }
        Alert.alert("✅ Success", isNew ? "Account created!" : "Logged in!");
        onLogin(userId, data.token);
      } else {
        Alert.alert("⚠️ Error", data.message || "Login failed");
      }
    } catch (error) {
      console.error("❌ Auth error:", error);
      Alert.alert("Error", "Something went wrong. Try again later.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.gradientBg}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <Animated.View
        style={[
          styles.card,
          {
            opacity: cardAnim,
            transform: [
              {
                scale: cardAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [0.95, 1],
                }),
              },
            ],
          },
        ]}
      >
        <View style={styles.brandContainer}>
          <Ionicons
            name="planet"
            size={38}
            color={colors.primary}
            style={{ marginBottom: 6 }}
          />
          <Text style={styles.brandName}>{BRAND_NAME}</Text>
        </View>
        <Text style={styles.title}>{isNew ? "Sign Up" : "Log In"}</Text>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>User ID</Text>
          <TextInput
            placeholder="Enter your user ID"
            style={styles.input}
            value={userId}
            onChangeText={setUserId}
            autoCapitalize="none"
            returnKeyType="next"
            onSubmitEditing={() => passwordInput.current?.focus()}
            placeholderTextColor={colors.textSecondary}
          />
        </View>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Password</Text>
          <View style={styles.passwordContainer}>
            <TextInput
              placeholder="Enter your password"
              style={[styles.input, styles.passwordInput]}
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!showPassword}
              returnKeyType="send"
              onSubmitEditing={handleAuth}
              ref={passwordInput}
              placeholderTextColor={colors.textSecondary}
            />
            <TouchableOpacity
              style={styles.eyeIcon}
              onPress={() => setShowPassword(!showPassword)}
              activeOpacity={0.7}
            >
              <Ionicons
                name={showPassword ? "eye-off" : "eye"}
                size={22}
                color={colors.primary}
              />
            </TouchableOpacity>
          </View>
        </View>
        <Animated.View
          style={{
            transform: [{ scale: buttonAnim }],
            width: "100%",
          }}
        >
          <TouchableOpacity
            style={[styles.ctaBtn, isLoading && styles.ctaBtnLoading]}
            onPress={handleAuth}
            activeOpacity={0.85}
            disabled={isLoading}
          >
            {isLoading ? (
              <Animated.View
                style={{
                  transform: [
                    {
                      rotate: loadingSpinValue.interpolate({
                        inputRange: [0, 1],
                        outputRange: ["0deg", "360deg"],
                      }),
                    },
                  ],
                }}
              >
                <Ionicons name="refresh" size={24} color="#fff" />
              </Animated.View>
            ) : (
              <Text style={styles.ctaBtnText}>
                {isNew ? "Sign Up" : "Log In"}
              </Text>
            )}
          </TouchableOpacity>
        </Animated.View>
        <TouchableOpacity
          style={styles.secondaryAction}
          onPress={() => setIsNew((prev) => !prev)}
          activeOpacity={0.7}
        >
          <Text style={styles.secondaryText}>
            {isNew
              ? "Already have an account? Log In"
              : "New user? Create account"}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.forgotAction}
          onPress={() =>
            Alert.alert(
              "Forgot Password",
              "Password recovery is not implemented yet."
            )
          }
          activeOpacity={0.7}
        >
          <Text style={styles.forgotText}>Forgot Password?</Text>
        </TouchableOpacity>
      </Animated.View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  gradientBg: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: colors.background,
    // Soft gradient background
    ...Platform.select({
      ios: { backgroundColor: colors.background },
      android: { backgroundColor: colors.background },
      default: { backgroundColor: colors.background },
    }),
  },
  card: {
    width: "92%",
    maxWidth: 400,
    paddingVertical: 32,
    paddingHorizontal: 28,
    backgroundColor: colors.surface,
    borderRadius: 22,
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.13,
    shadowRadius: 18,
    elevation: 8,
    alignItems: "center",
  },
  brandContainer: {
    alignItems: "center",
    marginBottom: 8,
  },
  brandName: {
    fontSize: 22,
    fontWeight: "700",
    color: colors.primary,
    letterSpacing: 1.2,
    fontFamily: "Inter",
    marginBottom: 2,
  },
  title: {
    fontSize: 20,
    fontWeight: "600",
    color: colors.textPrimary,
    textAlign: "center",
    marginBottom: 22,
    fontFamily: "Inter",
  },
  inputGroup: {
    width: "100%",
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    color: colors.textSecondary,
    marginBottom: 6,
    fontFamily: "Inter",
  },
  input: {
    backgroundColor: colors.background,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 10,
    padding: 13,
    fontSize: 16,
    color: colors.textPrimary,
    fontFamily: "Inter",
  },
  passwordContainer: {
    position: "relative",
    justifyContent: "center",
  },
  passwordInput: {
    paddingRight: 44,
  },
  eyeIcon: {
    position: "absolute",
    right: 10,
    top: "50%",
    transform: [{ translateY: -11 }],
    padding: 4,
  },
  ctaBtn: {
    width: "100%",
    backgroundColor: colors.primary,
    borderRadius: 10,
    paddingVertical: 14,
    marginTop: 10,
    marginBottom: 6,
    alignItems: "center",
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 6,
    elevation: 2,
    transition: "background-color 0.2s",
  },
  ctaBtnLoading: {
    opacity: 0.8,
  },
  ctaBtnText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
    fontFamily: "Inter",
    letterSpacing: 0.5,
  },
  secondaryAction: {
    marginTop: 10,
    alignItems: "center",
  },
  secondaryText: {
    color: colors.primary,
    fontWeight: "500",
    fontSize: 15,
    fontFamily: "Inter",
  },
  forgotAction: {
    marginTop: 8,
    alignItems: "center",
  },
  forgotText: {
    color: colors.textSecondary,
    fontSize: 14,
    fontFamily: "Inter",
    textDecorationLine: "underline",
  },
});
