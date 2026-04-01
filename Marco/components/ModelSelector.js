import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  TextInput,
  Alert,
  Linking,
} from "react-native";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";
import AIService from "../services/AIService";

export default function ModelSelector({ style }) {
  const [modalVisible, setModalVisible] = useState(false);
  const [apiKey, setApiKey] = useState("");

  const hasApiKey = AIService.hasApiKey();

  const handleApiKeySubmit = () => {
    if (!apiKey.trim()) {
      Alert.alert("Error", "Please enter an API key");
      return;
    }

    AIService.setApiKey(apiKey.trim());
    setModalVisible(false);
    setApiKey("");
  };

  const handleGetApiKey = () => {
    Linking.openURL("https://makersuite.google.com/app/apikey");
  };

  return (
    <View style={[styles.container, style]}>
      <TouchableOpacity
        style={styles.selector}
        onPress={() => setModalVisible(true)}
      >
        <View style={styles.modelSelectorContent}>
          <Icon name="brain" size={20} color={colors.buttonText} />
          <Text style={styles.currentModel}>
            {hasApiKey ? "Gemini Pro" : "Set API Key"}
          </Text>
        </View>
      </TouchableOpacity>

      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Gemini Pro Setup</Text>

            <View style={styles.apiKeyContainer}>
              <Text style={styles.apiKeyTitle}>Enter Gemini API Key</Text>
              <TextInput
                style={styles.apiKeyInput}
                value={apiKey}
                onChangeText={setApiKey}
                placeholder="Enter your API key"
                placeholderTextColor={colors.textTertiary}
                secureTextEntry
              />
              <TouchableOpacity onPress={handleGetApiKey}>
                <Text style={styles.getApiKeyLink}>
                  Get API Key from Google AI Studio
                </Text>
              </TouchableOpacity>
              <View style={styles.apiKeyButtons}>
                <TouchableOpacity
                  style={[styles.button, styles.cancelButton]}
                  onPress={() => setModalVisible(false)}
                >
                  <Text style={styles.buttonText}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.button, styles.submitButton]}
                  onPress={handleApiKeySubmit}
                >
                  <Text style={styles.buttonText}>Submit</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

import { colors } from "../theme/colors";

const styles = StyleSheet.create({
  container: {
    padding: 10,
  },
  selector: {
    backgroundColor: colors.primary,
    padding: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  modelSelectorContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
  },
  currentModel: {
    color: colors.buttonText,
    fontWeight: "600",
    marginLeft: 8,
  },
  getApiKeyLink: {
    color: colors.primary,
    textAlign: "center",
    textDecorationLine: "underline",
    marginBottom: 15,
  },
  modalContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.6)",
  },
  modalContent: {
    backgroundColor: colors.surface,
    borderRadius: 15,
    padding: 20,
    width: "80%",
    maxHeight: "80%",
    borderWidth: 1,
    borderColor: colors.border,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: "700",
    marginBottom: 20,
    textAlign: "center",
    color: colors.primary,
  },
  button: {
    padding: 12,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 10,
  },
  buttonText: {
    color: colors.buttonText,
    fontWeight: "600",
  },
  apiKeyContainer: {
    padding: 10,
  },
  apiKeyTitle: {
    fontSize: 16,
    marginBottom: 10,
    color: colors.textPrimary,
  },
  apiKeyInput: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
    color: colors.textPrimary,
    backgroundColor: colors.background,
  },
  apiKeyButtons: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  cancelButton: {
    backgroundColor: colors.disabled,
    flex: 1,
    marginRight: 5,
  },
  submitButton: {
    backgroundColor: colors.primary,
    flex: 1,
    marginLeft: 5,
  },
});
