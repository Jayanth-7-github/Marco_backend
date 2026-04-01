import React from "react";
import { View, TouchableOpacity, StyleSheet } from "react-native";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";
import { colors } from "../theme/colors";

const ChatHeader = ({ onMenuPress, visible }) => {
  if (visible) return null;

  return (
    <View style={styles.topBar}>
      <TouchableOpacity onPress={onMenuPress} style={styles.menuButton}>
        <Icon name="menu" size={24} color={colors.icon} />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  topBar: {
    width: "100%",
    height: 36,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "flex-start",
    marginBottom: 0,
  },
  menuButton: {
    marginLeft: 10,
    backgroundColor: "transparent",
    borderRadius: 25,
    paddingVertical: 6,
    paddingHorizontal: 10,
    elevation: 0,
    borderWidth: 0,
  },
});

export default ChatHeader;
