import React from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Image,
  StyleSheet,
  Linking,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as Clipboard from "expo-clipboard";
import { colors } from "../theme/colors";

export const MessageBubble = ({
  message,
  isUser,
  onCopy,
  index,
  copiedMessage,
}) => {
  if (isUser) {
    return (
      <View style={styles.userBubble}>
        <Text style={styles.userLabel}>You</Text>
        {message.user && <Text style={styles.userText}>{message.user}</Text>}

        {message.user && (
          <TouchableOpacity
            style={styles.messageCopyButton}
            onPress={() => onCopy(message.user)}
          >
            <Ionicons
              name={copiedMessage === index ? "checkmark" : "copy-outline"}
              size={16}
              color="#fff"
            />
          </TouchableOpacity>
        )}

        {message.imageUris && message.imageUris.length > 0 && (
          <View style={styles.imageGrid}>
            {message.imageUris.map((uri, idx) => (
              <Image key={idx} source={{ uri }} style={styles.sentImage} />
            ))}
          </View>
        )}

        {message.imageUri && (
          <Image source={{ uri: message.imageUri }} style={styles.sentImage} />
        )}
      </View>
    );
  }

  return (
    <View
      style={[
        styles.botBubble,
        message.bot === "Processing your input..." && styles.loadingBubble,
      ]}
    >
      <Text style={styles.botLabel}>Marco</Text>
      <Text style={styles.botText}>{message.bot}</Text>

      <TouchableOpacity
        style={styles.messageCopyButton}
        onPress={() => onCopy(message.bot)}
      >
        <Ionicons
          name={copiedMessage === index ? "checkmark" : "copy-outline"}
          size={16}
          color={colors.primary}
        />
      </TouchableOpacity>

      {message.bot === "Processing your input..." && (
        <ActivityIndicator
          size="small"
          color={colors.primary}
          style={styles.loadingIndicator}
        />
      )}

      {message.links && message.links.length > 0 && (
        <View style={styles.linksContainer}>
          {message.links.map((link, linkIndex) => (
            <TouchableOpacity
              key={linkIndex}
              onPress={() => Linking.openURL(link)}
              style={styles.linkButton}
            >
              <Text style={styles.linkText}>Read article {linkIndex + 1}</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}

      {message.model && (
        <Text style={styles.modelLabel}>
          via {message.model === "local" ? "Local Processing" : message.model}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  userBubble: {
    alignSelf: "flex-end",
    backgroundColor: colors.primary,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 18,
    maxWidth: "80%",
    borderTopRightRadius: 0,
    marginBottom: 5,
    elevation: 2,
  },
  userLabel: {
    fontSize: 12,
    color: colors.textPrimary,
    marginBottom: 3,
    textAlign: "right",
    marginRight: 28,
  },
  userText: {
    color: colors.textPrimary,
    fontSize: 16,
    lineHeight: 22,
  },
  botBubble: {
    alignSelf: "stretch",
    backgroundColor: "transparent",
    paddingVertical: 10,
    paddingHorizontal: 14,
    width: "100%",
    marginTop: 8,
  },
  botLabel: {
    fontSize: 12,
    color: colors.primary,
    marginBottom: 3,
    fontWeight: "600",
  },
  botText: {
    color: colors.textPrimary,
    fontSize: 16,
    lineHeight: 22,
  },
  messageCopyButton: {
    position: "absolute",
    right: 8,
    top: 6,
    width: 24,
    height: 24,
    backgroundColor: "rgba(0,0,0,0.2)",
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    zIndex: 1,
  },
  imageGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 8,
  },
  sentImage: {
    width: 180,
    height: 120,
    borderRadius: 12,
    resizeMode: "cover",
  },
  loadingBubble: {
    backgroundColor: "transparent",
  },
  loadingIndicator: {
    marginTop: 8,
  },
  linksContainer: {
    marginTop: 8,
    paddingTop: 8,
  },
  linkButton: {
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    padding: 8,
    borderRadius: 8,
    marginVertical: 4,
  },
  linkText: {
    color: colors.textPrimary,
    fontSize: 14,
    textAlign: "center",
    fontWeight: "600",
  },
  modelLabel: {
    fontSize: 10,
    color: "rgba(255, 255, 255, 0.6)",
    marginTop: 4,
    fontStyle: "italic",
  },
});

export default MessageBubble;
