import React from "react";
import { View, TouchableOpacity, Image, StyleSheet, Modal } from "react-native";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";
import { colors } from "../theme/colors";

const ImagePreviewModal = ({ visible, onClose, imageUri }) => {
  return (
    <Modal
      visible={visible}
      transparent={false}
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <TouchableOpacity style={styles.closeButton} onPress={onClose}>
          <Icon name="close" size={28} color={colors.textPrimary} />
        </TouchableOpacity>
        {imageUri && (
          <Image source={{ uri: imageUri }} style={styles.fullPreviewImage} />
        )}
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  closeButton: {
    position: "absolute",
    top: 40,
    right: 20,
    zIndex: 20,
  },
  fullPreviewImage: {
    width: "100%",
    height: "100%",
    resizeMode: "contain",
    marginBottom: 20,
  },
});

export default ImagePreviewModal;
