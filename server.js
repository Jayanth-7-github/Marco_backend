import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import cookieParser from "cookie-parser";
import User from "./models/User.js";
import Memory from "./models/Memory.js";
import Chat from "./models/Chat.js";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cookieParser());

// ✅ Allow mobile requests with credentials
app.use(
  cors({
    origin: "*", // Add your React Native app's URL
    methods: ["GET", "POST", "PUT", "DELETE"],
    allowedHeaders: ["Content-Type", "Authorization"],
    credentials: true, // Allow credentials
  })
);

// ✅ Connect to MongoDB
const MONGO_URI = process.env.MONGO_URI;
if (!MONGO_URI) {
  console.error("❌ MONGO_URI missing in .env file");
  process.exit(1);
}

mongoose
  .connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log("✅ MongoDB connected successfully"))
  .catch((err) => console.error("❌ MongoDB connection error:", err));

// =========================
// 🧍 User Authentication
// =========================

// 🔹 Register
app.post("/register", async (req, res) => {
  try {
    const { userId, password } = req.body;
    if (!userId || !password)
      return res
        .status(400)
        .json({ message: "Please provide userId and password." });

    const existingUser = await User.findOne({ userId });
    if (existingUser)
      return res
        .status(400)
        .json({ message: "User ID already exists. Try another!" });

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ userId, password: hashedPassword });
    await newUser.save();

    res.json({ success: true, message: "User registered successfully 🎉" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: "Registration failed" });
  }
});

// 🔹 Login
app.post("/login", async (req, res) => {
  try {
    const { userId, password } = req.body;
    const user = await User.findOne({ userId });
    if (!user) return res.status(404).json({ message: "User not found!" });

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch)
      return res.status(401).json({ message: "Incorrect password!" });

    const token = jwt.sign({ userId: user.userId }, process.env.JWT_SECRET, {
      expiresIn: "7d",
    });

    // Set JWT in cookie
    res.cookie("token", token, {
      httpOnly: true, // Prevents JavaScript access to cookie
      secure: process.env.NODE_ENV === "production", // Use HTTPS in production
      sameSite: "strict", // CSRF protection
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days in milliseconds
    });

    // ✅ Send success response
    res.json({
      success: true,
      message: `Welcome back, ${userId}!`,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: "Login failed" });
  }
});

// 🔹 Logout
app.post("/logout", (req, res) => {
  res.clearCookie("token"); // Clear the token cookie
  res.json({ success: true, message: "Logged out successfully 🚪" });
});

// =========================
// 🔐 JWT Middleware
// =========================
const verifyToken = (req, res, next) => {
  // Check for token in cookies first, then in Authorization header
  const token = req.cookies.token || req.headers.authorization?.split(" ")[1];

  if (!token) {
    return res
      .status(403)
      .json({ message: "No token provided. Please login." });
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {
    if (err) {
      // Clear invalid cookie if present
      if (req.cookies.token) {
        res.clearCookie("token");
      }
      return res.status(401).json({ message: "Invalid or expired token." });
    }
    req.userId = decoded.userId;
    next();
  });
};

// =========================
// 🧠 Memory Routes (Protected)
// =========================

// 🔹 Save or update memory
app.post("/memory", verifyToken, async (req, res) => {
  try {
    const { type, content } = req.body;
    const userId = req.userId;

    let memory = await Memory.findOne({ userId });
    if (!memory) memory = new Memory({ userId, memories: [] });

    memory.memories.push({ type, content });
    await memory.save();

    res.json({ success: true, message: "Memory saved!", data: memory });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: "Error saving memory" });
  }
});

// 🔹 Retrieve all memories for logged-in user
app.get("/memory", verifyToken, async (req, res) => {
  try {
    const memory = await Memory.findOne({ userId: req.userId });
    res.json(memory || { memories: [] });
  } catch (error) {
    res
      .status(500)
      .json({ success: false, message: "Error retrieving memory" });
  }
});

// 🔹 Delete all memories
app.delete("/memory", verifyToken, async (req, res) => {
  try {
    await Memory.deleteOne({ userId: req.userId });
    res.json({ success: true, message: "🧹 All memories cleared!" });
  } catch (error) {
    res.status(500).json({ success: false, message: "Error clearing memory" });
  }
});

// =========================
// 💬 Chat Routes (Protected)
// =========================

// 🔹 Store new chat conversation (generates chatId automatically)
app.post("/chat", verifyToken, async (req, res) => {
  try {
    const { conversation } = req.body;
    const userId = req.userId;

    if (!Array.isArray(conversation)) {
      return res.status(400).json({
        success: false,
        message: "Conversation must be an array of messages",
      });
    }

    // Generate a unique chatId
    const chatId =
      "chat_" +
      new Date().getTime() +
      "_" +
      Math.random().toString(36).substring(2, 15);

    // Create new chat
    const chat = new Chat({
      chatId,
      userId,
      conversation,
      timestamp: new Date(),
    });

    await chat.save();

    res.json({
      success: true,
      message: "Chat conversation saved!",
      data: chat,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: "Error saving chat conversation",
    });
  }
});

// 🔹 Retrieve chat by chatId
app.get("/chat/:chatId", verifyToken, async (req, res) => {
  try {
    const { chatId } = req.params;
    const chat = await Chat.findOne({ chatId });

    if (!chat) {
      return res.status(404).json({
        success: false,
        message: "Chat not found",
      });
    }

    // Verify chat ownership
    if (chat.userId !== req.userId) {
      return res.status(403).json({
        success: false,
        message: "You don't have permission to access this chat",
      });
    }

    res.json({
      success: true,
      data: chat,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: "Error retrieving chat",
    });
  }
});

// 🔹 Delete chat by chatId
app.delete("/chat/:chatId", verifyToken, async (req, res) => {
  try {
    const { chatId } = req.params;
    const chat = await Chat.findOne({ chatId });

    if (!chat) {
      return res.status(404).json({
        success: false,
        message: "Chat not found",
      });
    }

    // Verify chat ownership
    if (chat.userId !== req.userId) {
      return res.status(403).json({
        success: false,
        message: "You don't have permission to delete this chat",
      });
    }

    await Chat.deleteOne({ chatId });
    res.json({
      success: true,
      message: "💫 Chat deleted successfully!",
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: "Error deleting chat",
    });
  }
});

// 🔹 Get all chats for current user
app.get("/chats", verifyToken, async (req, res) => {
  try {
    const chats = await Chat.find({ userId: req.userId });
    res.json({
      success: true,
      data: chats,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: "Error retrieving chats",
    });
  }
});

// =========================
// 🌍 Root Route
// =========================
app.get("/", (req, res) => {
  res.send("✅ Marco Backend is running on Render!");
});

// ✅ Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () =>
  console.log(`🚀 Server running on https://marco-backend-u19w.onrender.com`)
);
