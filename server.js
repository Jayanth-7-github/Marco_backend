import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import User from "./models/User.js";
import Memory from "./models/Memory.js";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

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

// Register
// Register
app.post("/register", async (req, res) => {
  try {
    const { userId, password } = req.body;
    if (!userId || !password)
      return res.status(400).json({ message: "Please provide userId and password." });

    const existingUser = await User.findOne({ userId });
    if (existingUser)
      return res.status(400).json({ message: "User ID already exists. Try another!" });

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ userId, password: hashedPassword });
    await newUser.save();

    res.json({ success: true, message: "User registered successfully 🎉" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: "Registration failed" });
  }
});

// Login
app.post("/login", async (req, res) => {
  try {
    const { userId, password } = req.body;
    const user = await User.findOne({ userId });
    if (!user) return res.status(404).json({ message: "User not found!" });

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(401).json({ message: "Incorrect password!" });

    const token = jwt.sign({ userId: user.userId }, process.env.JWT_SECRET, { expiresIn: "7d" });
    res.json({ success: true, message: `Welcome back, ${userId}!`, token });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: "Login failed" });
  }
});

// Logout



// Middleware to verify JWT
const verifyToken = (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader)
    return res.status(403).json({ message: "No token provided. Please login." });

  const token = authHeader.split(" ")[1];
  jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {
    if (err) return res.status(401).json({ message: "Invalid or expired token." });
    req.userId = decoded.userId;
    next();
  });
};

app.post("/logout", verifyToken, async (req, res) => {
  try {
    // Future: Add token blacklist or revocation here if needed
    res.json({ success: true, message: "Logged out successfully 🚪" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: "Logout failed" });
  }
});

// =========================
// 🧠 Memory Routes (Protected)
// =========================

// Save or update memory
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

// Retrieve all memories for logged-in user
app.get("/memory", verifyToken, async (req, res) => {
  try {
    const memory = await Memory.findOne({ userId: req.userId });
    res.json(memory || { memories: [] });
  } catch (error) {
    res.status(500).json({ success: false, message: "Error retrieving memory" });
  }
});

// Delete all memories
app.delete("/memory", verifyToken, async (req, res) => {
  try {
    await Memory.deleteOne({ userId: req.userId });
    res.json({ success: true, message: "🧹 All memories cleared!" });
  } catch (error) {
    res.status(500).json({ success: false, message: "Error clearing memory" });
  }
});

// ✅ Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));
