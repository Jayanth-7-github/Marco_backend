import mongoose from "mongoose";

const MemorySchema = new mongoose.Schema({
  userId: { type: String, required: true },
  memories: [
    {
      type: { type: String },
      content: { type: String },
      date: { type: Date, default: Date.now },
    },
  ],
});

export default mongoose.model("Memory", MemorySchema);
