import AsyncStorage from "@react-native-async-storage/async-storage";
import EnhancedReasoner from "./components/LocalReasoner";
import IntentProcessor from "./utils/intentProcessor";
import EnhancedMemorySystem from "./utils/enhancedMemory";
import AIResponseService from "./services/AIResponseService";

// Initialize systems
const setupMarco = async () => {
  try {
    console.log("🤖 Initializing Marco...");

    // Initialize components in sequence
    const memory = new EnhancedMemorySystem();
    await memory.initialize();
    console.log("✅ Memory system initialized");

    const intentProcessor = new IntentProcessor();
    await intentProcessor.initialize();
    console.log("✅ Intent processor initialized");

    const aiService = new AIResponseService();
    await aiService.initialize();
    console.log("✅ AI Response Service initialized");

    const reasoner = new EnhancedReasoner();
    // Pass initialized components to reasoner
    reasoner.memorySystem = memory;
    reasoner.intentProcessor = intentProcessor;
    reasoner.aiService = aiService;
    await reasoner.initialize();
    console.log("✅ Reasoning system initialized");

    return {
      memory,
      intentProcessor,
      reasoner,
    };
  } catch (error) {
    console.error("❌ Error initializing Marco:", error);
    throw error;
  }
};

// Main interaction loop
export const processUserInput = async (input, context = []) => {
  try {
    // Initialize if not done
    if (!global.marcoSystems) {
      global.marcoSystems = await setupMarco();
      console.log("✅ Marco systems initialized successfully");
    }

    const { reasoner, memory } = global.marcoSystems;

    // Ensure memory is refreshed
    await memory.initialize();

    // Process the input
    console.log("🤖 Processing input:", input);
    const response = await reasoner.reason(context, input);
    console.log("✅ Generated response:", response);

    return response;
  } catch (error) {
    console.error("❌ Error processing input:", error);
    return "I encountered an error while processing that. Please try again.";
  }
};

// Export the setup function for manual initialization
export const initializeMarco = setupMarco;

// Export individual components for direct access
export { default as Reasoner } from "./components/LocalReasoner";
export { default as IntentProcessor } from "./utils/intentProcessor";
export { default as MemorySystem } from "./utils/enhancedMemory";
