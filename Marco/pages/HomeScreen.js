// pages/HomeScreen.js

import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  TouchableWithoutFeedback,
  Keyboard,
  Platform,
} from "react-native";
import { colors } from "../theme/colors";
import Sidebar from "../components/Sidebar";
import { initializeMarco } from "../marcoCore";
import ChatService from "../services/ChatService";
import MediaActions from "../components/MediaActions";
import MessageBubble from "../components/MessageBubble";
import AttachmentPreview from "../components/AttachmentPreview";
import ImagePreviewModal from "../components/ImagePreviewModal";
import ChatHeader from "../components/ChatHeader";
// import ModelSelector from "../components/ModelSelector";
/* Lines 29-33 omitted */

import * as Clipboard from "expo-clipboard";
import { Ionicons } from "@expo/vector-icons";
export default function HomeScreen({ userId: propUserId, onLogout, route }) {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([]);
  const [memories, setMemories] = useState([]);
  const [chatId, setChatId] = useState(null);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [copiedMessage, setCopiedMessage] = useState(null);
  const scrollViewRef = useRef(null);
  const [pendingAttachments, setPendingAttachments] = useState([]);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [selectedPreviewIndex, setSelectedPreviewIndex] = useState(0);
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    const init = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Initialize core systems
        await Promise.all([initializeMarco(), ChatService.initialize()]);

        // Load initial data
        const [memories, prevChat] = await Promise.all([
          ChatService.loadMemories(),
          ChatService.loadPreviousChat(propUserId),
        ]);

        setMemories(memories);
        if (prevChat) {
          setChat(prevChat.messages);
          setChatId(prevChat.id);
        }
      } catch (error) {
        console.error("Error during initialization:", error);
        setError("Failed to initialize the app");
        Alert.alert(
          "Initialization Error",
          "There was a problem starting the app. Some features might be limited."
        );
      } finally {
        setIsLoading(false);
      }
    };
    init();
  }, [propUserId]);

  // Handle route params (open existing chat or start new)
  useEffect(() => {
    const params = route?.params;
    if (!params) return;

    const handleParams = async () => {
      try {
        if (params.openChatId) {
          const loadedChat = await ChatService.loadChatById(
            propUserId,
            params.openChatId
          );
          if (loadedChat) {
            setChat(loadedChat.messages);
            setChatId(loadedChat.id);
          }
        } else if (params.newChat) {
          setChat([]);
          setChatId(null);
        }
      } catch (e) {
        console.warn("Error handling route params", e);
      }
    };

    handleParams();
  }, [route?.params, propUserId]);

  // Persist chats on update
  useEffect(() => {
    const persistChats = async () => {
      try {
        const newChatId = await ChatService.persistChat(
          propUserId,
          chatId,
          chat
        );
        if (newChatId && !chatId) {
          setChatId(newChatId);
        }
      } catch (e) {
        console.warn("Could not persist chats", e);
      }
    };

    if (chat && chat.length > 0 && propUserId) {
      persistChats();
    }
  }, [chat, propUserId, chatId]);

  // Ensure ScrollView scrolls correctly when keyboard shows/hides
  useEffect(() => {
    const onKeyboardShow = () => {
      // allow layout to settle then scroll to end
      setTimeout(
        () => scrollViewRef.current?.scrollToEnd({ animated: true }),
        50
      );
    };

    const onKeyboardHide = () => {
      // when keyboard hides, ensure no leftover gap — scroll to end after layout
      setTimeout(
        () => scrollViewRef.current?.scrollToEnd({ animated: true }),
        50
      );
    };

    const showSub = Keyboard.addListener("keyboardDidShow", onKeyboardShow);
    const hideSub = Keyboard.addListener("keyboardDidHide", onKeyboardHide);

    return () => {
      try {
        showSub.remove();
        hideSub.remove();
      } catch (e) {
        // some RN versions use remove() or removeSubscription
      }
    };
  }, []);

  const handleMemoryUpdate = async () => {
    try {
      const updatedMemories = await ChatService.loadMemories();
      setMemories(updatedMemories);
    } catch (error) {
      console.error("Error updating memories:", error);
      Alert.alert("Error", "Failed to update memories");
    }
  };

  const handleUserInput = async () => {
    const userText = input.trim();
    if (!userText) return;

    try {
      setIsProcessing(true);
      setInput("");

      // Process user input through ChatService
      const { response, memory } = await ChatService.processUserInput(
        userText,
        memories
      );

      // First add user message
      setChat((prev) => [
        ...prev,
        {
          user: userText,
        },
      ]);

      // Then add bot response
      setChat((prev) => [
        ...prev,
        {
          bot: response.content,
          model: response.model,
          links: response.links,
        },
      ]);

      // If memory was updated, refresh memories
      if (memory) {
        await handleMemoryUpdate();
      }
    } catch (error) {
      console.error("Error processing input:", error);
      Alert.alert(
        "Oops!",
        "I had trouble processing that. Could you try again?"
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSend = async () => {
    if (isProcessing) return;

    if (pendingAttachments.length > 0) {
      try {
        setIsProcessing(true);
        const userText = input.trim();

        // First add the user's message with images
        setChat((prev) => [
          ...prev,
          {
            user: userText || "",
            imageUris: pendingAttachments.map((att) => att.uri),
          },
        ]);

        // Then add a temporary loading message from Marco
        setChat((prev) => [
          ...prev,
          {
            bot: "Processing your input...",
            model: "local",
          },
        ]);

        // Process the text input through ChatService if provided
        if (userText) {
          try {
            const { response, memory } = await ChatService.processUserInput(
              userText,
              memories
            );
            // Replace the loading message with the actual response
            setChat((prev) =>
              prev.slice(0, -1).concat({
                bot: response.content,
                model: response.model,
                links: response.links,
              })
            );
            // If memory was updated, refresh memories
            if (memory) {
              await handleMemoryUpdate();
            }
          } catch (error) {
            console.error("Error processing text with images:", error);
            // Replace loading message with error message
            setChat((prev) =>
              prev.slice(0, -1).concat({
                bot: "Sorry, I had trouble processing your message.",
                model: "local",
              })
            );
          }
        } else {
          // If no text, just acknowledge the images
          setChat((prev) =>
            prev.slice(0, -1).concat({
              bot: "I received your images.",
              model: "local",
            })
          );
        }

        // Clear pending attachments and input
        setPendingAttachments([]);
        setInput("");
        setIsProcessing(false);

        // scroll to bottom
        setTimeout(
          () => scrollViewRef.current?.scrollToEnd({ animated: true }),
          100
        );
      } catch (e) {
        console.error("Error sending attachments:", e);
      } finally {
        setIsProcessing(false);
      }

      return;
    }

    // otherwise fallback to normal text send
    await handleUserInput();
  };

  // Placeholder for any image/file helper functions needed in the future

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={styles.container}
      keyboardVerticalOffset={Platform.OS === "ios" ? 60 : 0}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <View style={styles.innerContainer}>
          {isLoading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
              <Text style={styles.loadingText}>Loading...</Text>
            </View>
          ) : (
            <>
              {/* Top header with menu button */}
              <ChatHeader
                visible={sidebarVisible}
                onMenuPress={() => setSidebarVisible(true)}
              />

              {/* Sidebar Drawer */}
              <Sidebar
                visible={sidebarVisible}
                onClose={() => setSidebarVisible(false)}
                onLogout={onLogout}
              />

              {/* Removed Marco title from top */}
              {error && <Text style={styles.errorText}>{error}</Text>}

              {/* <ModelSelector style={styles.modelSelector} /> */}

              <ScrollView
                style={styles.chat}
                contentContainerStyle={styles.chatContent}
                ref={scrollViewRef}
                onContentSizeChange={() =>
                  scrollViewRef.current?.scrollToEnd({ animated: true })
                }
              >
                {chat.map((message, index) => (
                  <View key={index} style={styles.messageContainer}>
                    <MessageBubble
                      message={message}
                      isUser={message.user || message.imageUris?.length > 0}
                      onCopy={(text) => {
                        Clipboard.setStringAsync(text);
                        if (message.user) {
                          setInput(text);
                        } else {
                          setCopiedMessage(index);
                          setTimeout(() => setCopiedMessage(null), 2000);
                        }
                      }}
                      index={index}
                      copiedMessage={copiedMessage}
                    />
                  </View>
                ))}
              </ScrollView>

              {/* Image Picker Modal moved to MediaActions component */}

              {/* Fullscreen image preview */}
              <ImagePreviewModal
                visible={previewVisible}
                onClose={() => setPreviewVisible(false)}
                imageUri={pendingAttachments[selectedPreviewIndex]?.uri}
              />

              {/* Attachment preview row */}
              <AttachmentPreview
                attachments={pendingAttachments}
                onRemoveAttachment={(index) => {
                  setPendingAttachments((prev) =>
                    prev.filter((_, i) => i !== index)
                  );
                }}
                onPreviewPress={(index) => {
                  setSelectedPreviewIndex(index);
                  setPreviewVisible(true);
                }}
              />

              {/* Input area */}
              <MediaActions
                isProcessing={isProcessing}
                input={input}
                setInput={setInput}
                onImageCapture={(images) => {
                  setPendingAttachments((prev) => [...prev, ...images]);
                  setTimeout(
                    () =>
                      scrollViewRef.current?.scrollToEnd({ animated: true }),
                    100
                  );
                }}
                onFilesPicked={(files) => {
                  setPendingAttachments((prev) => [...prev, ...files]);
                  setTimeout(
                    () =>
                      scrollViewRef.current?.scrollToEnd({ animated: true }),
                    100
                  );
                }}
                onSendMessage={handleSend}
              />
            </>
          )}
        </View>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  innerContainer: {
    flex: 1,
    paddingTop: 60,
    paddingHorizontal: 15,
    paddingBottom: 10,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  loadingText: {
    marginTop: 10,
    color: colors.primary,
    fontSize: 16,
  },
  errorText: {
    color: colors.error,
    fontSize: 14,
    marginTop: 5,
  },
  chat: {
    flex: 1,
    marginBottom: 15,
    paddingVertical: 10,
  },
  chatContent: {
    flexGrow: 1,
    justifyContent: "flex-end",
    paddingBottom: 0,
  },
  messageContainer: {
    marginBottom: 24,
    width: "100%",
  },
});
