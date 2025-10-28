import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import Memory from "./models/Memory.js";

dotenv.config(); // ✅ Load variables from .env

const app = express();
app.use(express.json());
app.use(cors());

// ✅ Connect to MongoDB using .env variable
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
  console.error("❌ MONGO_URI not found in .env file");
  process.exit(1);
}

mongoose
  .connect(MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("✅ MongoDB connected successfully"))
  .catch((err) => console.error("❌ MongoDB connection error:", err));

// ✅ Memory routes

// Save or update memory
app.post("/memory", async (req, res) => {
  try {
    const { userId, type, content } = req.body;
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

// Retrieve memories
app.get("/memory/:userId", async (req, res) => {
  try {
    const memory = await Memory.findOne({ userId: req.params.userId });
    res.json(memory || {});
  } catch (error) {
    res.status(500).json({ success: false, message: "Error retrieving memory" });
  }
});

// Delete or forget
app.delete("/memory/:userId", async (req, res) => {
  try {
    await Memory.deleteOne({ userId: req.params.userId });
    res.json({ success: true, message: "Memory cleared!" });
  } catch (error) {
    res.status(500).json({ success: false, message: "Error clearing memory" });
  }
});

// ✅ Run Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
