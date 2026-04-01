import React, { useState } from "react";
import {
  View,
  TouchableOpacity,
  StyleSheet,
  Modal,
  Text,
  Pressable,
  Alert,
  TextInput,
  ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";
import FontAwesome from "react-native-vector-icons/FontAwesome";
import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import { colors } from "../theme/colors";

const getMediaType = () => {
  return (
    ImagePicker.MediaType?.Image ||
    ImagePicker.MediaTypeOptions?.Images ||
    "images"
  );
};

const MediaActions = ({
  isProcessing,
  onImageCapture,
  onFilesPicked,
  onSendMessage,
  onRecordingChange = () => {},
  input,
  setInput,
  hasAttachments = false,
}) => {
  const [pickerVisible, setPickerVisible] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const takePhoto = async () => {
    setPickerVisible(false);
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync?.();
      if (status !== "granted") {
        Alert.alert(
          "Permission required",
          "Camera permission is required to take photos."
        );
        return;
      }
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: getMediaType(),
        quality: 0.8,
      });
      if (!result.canceled && (result.assets?.length || result.uri)) {
        const uri = result.assets?.[0]?.uri || result.uri;
        if (uri) {
          onImageCapture([{ uri }]);
        }
      }
    } catch (e) {
      console.warn("expo-image-picker not available", e);
      Alert.alert(
        "Missing dependency",
        "To use photos, please install `expo-image-picker` or run this in Expo."
      );
    }
  };

  const pickFromLibrary = async () => {
    setPickerVisible(false);
    try {
      const { status } =
        await ImagePicker.requestMediaLibraryPermissionsAsync?.();
      if (status !== "granted") {
        Alert.alert(
          "Permission required",
          "Media library permission is required to choose photos."
        );
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: getMediaType(),
        quality: 0.8,
        allowsMultipleSelection: true,
      });

      if (
        !result.canceled &&
        (result.selected?.length || result.assets?.length || result.uri)
      ) {
        const uris = result.selected
          ? result.selected.map((a) => ({ uri: a.uri }))
          : result.assets
          ? result.assets.map((a) => ({ uri: a.uri }))
          : [{ uri: result.uri }];

        onImageCapture(uris);
      }
    } catch (e) {
      console.warn("expo-image-picker not available", e);
      Alert.alert(
        "Missing dependency",
        "To use photo library, please install `expo-image-picker` or run this in Expo."
      );
    }
  };

  const pickDocument = async () => {
    try {
      setPickerVisible(false);
      const result = await DocumentPicker.getDocumentAsync({
        type: ["*/*"],
        multiple: true,
        copyToCacheDirectory: true,
      });

      if (result.assets) {
        const files = result.assets.map((file) => ({
          uri: file.uri,
          type: "file",
          name: file.name,
          size: file.size,
          mimeType: file.mimeType,
        }));
        onFilesPicked(files);
      } else if (!result.canceled && result.uri) {
        onFilesPicked([
          {
            uri: result.uri,
            type: "file",
            name: result.name,
            size: result.size,
            mimeType: result.mimeType,
          },
        ]);
      }
    } catch (err) {
      console.warn("Error picking document:", err);
      Alert.alert("Error", "Could not pick the document. Please try again.");
    }
  };

  const handleMicPress = () => {
    const newRecordingState = !isRecording;
    setIsRecording(newRecordingState);
    if (onRecordingChange) {
      onRecordingChange(newRecordingState);
    }
    setInput(newRecordingState ? "Recording..." : "🎤 Voice message recorded");
  };

  return (
    <>
      <View style={styles.inputContainer}>
        <TouchableOpacity
          style={styles.plusButton}
          onPress={() => setPickerVisible(true)}
        >
          <FontAwesome name="plus" size={18} color={colors.icon} />
        </TouchableOpacity>

        <TextInput
          style={[styles.input, isProcessing && styles.inputDisabled]}
          value={input}
          onChangeText={setInput}
          placeholder="Add a message..."
          placeholderTextColor={colors.textTertiary}
          onSubmitEditing={() => onSendMessage(input)}
          editable={!isProcessing}
        />

        <Pressable
          style={[
            styles.micButton,
            isRecording && { backgroundColor: "rgba(255, 59, 48, 0.2)" },
          ]}
          onPress={handleMicPress}
          disabled={isProcessing}
        >
          <Ionicons
            name={isRecording ? "mic-off" : "mic"}
            size={22}
            color={isRecording ? colors.error : colors.textPrimary}
          />
        </Pressable>

        <TouchableOpacity
          style={[
            styles.sendButton,
            ((!input.trim() && !hasAttachments) || isProcessing) &&
              styles.sendButtonDisabled,
          ]}
          onPress={() => onSendMessage(input)}
          disabled={isProcessing || (!input.trim() && !hasAttachments)}
        >
          {isProcessing ? (
            <ActivityIndicator size="small" color={colors.textPrimary} />
          ) : (
            <Text style={styles.sendButtonIcon}>➤</Text>
          )}
        </TouchableOpacity>
      </View>

      <Modal
        visible={pickerVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setPickerVisible(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setPickerVisible(false)}
        >
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Upload Media</Text>
              <TouchableOpacity
                style={styles.modalCloseButton}
                onPress={() => setPickerVisible(false)}
              >
                <Icon name="close" size={24} color={colors.textPrimary} />
              </TouchableOpacity>
            </View>

            <View style={styles.modalOptionsGrid}>
              <TouchableOpacity
                style={styles.modalOptionCard}
                onPress={takePhoto}
              >
                <Icon name="camera" size={24} color={colors.primary} />
                <Text style={styles.modalOptionText}>Take Photo</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.modalOptionCard}
                onPress={pickFromLibrary}
              >
                <Icon name="image" size={24} color={colors.primary} />
                <Text style={styles.modalOptionText}>Photo Library</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.modalOptionCard}
                onPress={pickDocument}
              >
                <Icon name="file-upload" size={24} color={colors.primary} />
                <Text style={styles.modalOptionText}>Upload File</Text>
              </TouchableOpacity>
            </View>
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  inputContainer: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.secondaryBg,
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 4,
    marginBottom: 15,
  },
  plusButton: {
    width: 36,
    height: 36,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 8,
  },
  input: {
    flex: 1,
    height: 40,
    fontSize: 16,
    color: colors.text,
    marginRight: 8,
  },
  inputDisabled: {
    opacity: 0.6,
  },
  micButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 8,
  },
  sendButton: {
    width: 36,
    height: 36,
    backgroundColor: colors.primary,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
  },
  sendButtonDisabled: {
    opacity: 0.6,
    backgroundColor: colors.textSecondary,
  },
  sendButtonIcon: {
    color: "#FFFFFF",
    fontSize: 18,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: colors.overlay,
    justifyContent: "center",
    alignItems: "center",
  },
  modalContent: {
    backgroundColor: colors.surface,
    padding: 24,
    borderRadius: 16,
    width: "90%",
    maxWidth: 400,
    elevation: 5,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 20,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: "600",
    color: colors.textPrimary,
  },
  modalCloseButton: {
    padding: 4,
  },
  modalOptionsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-around",
    gap: 16,
  },
  modalOptionCard: {
    width: "30%",
    aspectRatio: 1,
    backgroundColor: colors.background,
    borderRadius: 12,
    padding: 12,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.border,
  },
  modalOptionText: {
    color: colors.textPrimary,
    fontSize: 14,
    marginTop: 8,
    textAlign: "center",
    fontWeight: "500",
  },
  // Input text placeholder color is set via placeholderTextColor prop
  input: {
    flex: 1,
    height: 40,
    fontSize: 16,
    color: colors.text, // Main input text color
    marginRight: 8,
  },
  inputPlaceholder: {
    color: colors.textTertiary,
  },
  sendButtonIcon: {
    color: colors.textPrimary, // White text on primary color background
    fontSize: 18,
  },
  micButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 8,
  },
  // Icon colors are set via color prop in the components:
  // - Plus icon: colors.icon
  // - Mic icon: "#FFFFFF" when not recording, "#FF3B30" when recording
  // - Close icon in modal: colors.textPrimary
  // - Modal option icons: colors.primary
});

export default MediaActions;
