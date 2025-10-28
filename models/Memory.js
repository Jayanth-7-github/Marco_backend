// models/Memory.js
import mongoose from "mongoose";

const memorySchema = new mongoose.Schema({
  userId: { type: String, required: true },
  memories: [
    {
      type: { type: String, enum: ["fact", "preference", "note", "goal"], default: "fact" },
      content: { type: String, required: true },
      timestamp: { type: Date, default: Date.now },
      embedding: { type: [Number], default: [] } // optional for semantic search
    }
  ]
});

export default mongoose.model("Memory", memorySchema);
