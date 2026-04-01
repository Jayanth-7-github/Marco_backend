import React from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Image,
  StyleSheet,
  Alert,
} from "react-native";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";
import { colors } from "../theme/colors";

const AttachmentPreview = ({
  attachments,
  onRemoveAttachment,
  onPreviewPress,
}) => {
  if (!attachments || attachments.length === 0) return null;

  return (
    <View style={styles.attachmentPreviewContainer}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        {attachments.map((attachment, index) => (
          <View key={index} style={styles.attachmentWrapper}>
            <TouchableOpacity
              onPress={() => {
                if (attachment.type === "file") {
                  Alert.alert(
                    "File Details",
                    `Name: ${attachment.name}\nSize: ${(
                      attachment.size / 1024
                    ).toFixed(2)} KB\nType: ${attachment.mimeType}`
                  );
                } else {
                  onPreviewPress(index);
                }
              }}
            >
              {attachment.type === "file" ? (
                <View style={styles.filePreview}>
                  <Icon
                    name="file-document-outline"
                    size={32}
                    color={colors.primary}
                  />
                  <Text style={styles.fileNameText} numberOfLines={2}>
                    {attachment.name}
                  </Text>
                </View>
              ) : (
                <Image
                  source={{ uri: attachment.uri }}
                  style={styles.attachmentPreview}
                />
              )}
            </TouchableOpacity>
            <View style={styles.attachmentControls}>
              <TouchableOpacity
                style={styles.removeAttachment}
                onPress={() => onRemoveAttachment(index)}
              >
                <Icon name="close" size={14} color={colors.textPrimary} />
              </TouchableOpacity>
            </View>
          </View>
        ))}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  attachmentPreviewContainer: {
    marginBottom: 8,
    backgroundColor: colors.surface,
  },
  attachmentWrapper: {
    marginRight: 8,
    borderRadius: 12,
    overflow: "hidden",
    position: "relative",
  },
  attachmentPreview: {
    width: 150,
    height: 150,
    resizeMode: "cover",
    borderRadius: 12,
  },
  filePreview: {
    width: 150,
    height: 150,
    backgroundColor: colors.surface,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    padding: 12,
    borderWidth: 1,
    borderColor: colors.border,
  },
  fileNameText: {
    color: colors.textPrimary,
    fontSize: 12,
    textAlign: "center",
    marginTop: 8,
    flexWrap: "wrap",
  },
  attachmentControls: {
    position: "absolute",
    top: 8,
    right: 8,
    flexDirection: "row",
    gap: 8,
  },
  removeAttachment: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "rgba(0,0,0,0.6)",
    justifyContent: "center",
    alignItems: "center",
  },
});

export default AttachmentPreview;
