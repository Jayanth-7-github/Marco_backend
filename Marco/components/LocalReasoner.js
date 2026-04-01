import commandData from "../assets/command.json";
import IntentProcessor from "../utils/intentProcessor";
import EnhancedMemorySystem from "../utils/enhancedMemory";
import NewsService from "../services/news_service";

class EnhancedReasoner {
  constructor() {
    this.intentProcessor = new IntentProcessor();
    this.memorySystem = new EnhancedMemorySystem();
    this.commands = commandData;
  }

  async initialize() {
    await this.intentProcessor.initialize();
    await this.memorySystem.initialize();
  }

  async reason(context, input) {
    try {
      const text = input.toLowerCase().trim();
      console.log("LocalReasoner: Processing input:", text);

      // Extract entities from input
      const entities = this.intentProcessor.extractEntities(text);
      console.log("LocalReasoner: Extracted entities:", entities);

      // Check for name introduction first
      if (entities.isNameIntro && entities.person) {
        return await this.handleCommand("remember_name", entities, text);
      }

      // Check for specific commands in command.json first
      let matchedCommand = null;

      // First try exact matches from commands
      for (let cmd of this.commands) {
        for (let pattern of cmd.patterns) {
          if (text === pattern.toLowerCase()) {
            matchedCommand = cmd.intent;
            break;
          }
        }
        if (matchedCommand) break;
      }

      // If we found a matching command, handle it
      if (matchedCommand) {
        let reply = await this.handleCommand(matchedCommand, entities, text);
        if (reply) {
          return reply;
        }
      }

      // If no exact command match, try AI-powered response
      if (this.aiService) {
        const aiResponse = await this.aiService.generateResponse(text, context);
        if (aiResponse) {
          // Store the interaction in memory
          await this.memorySystem.addMemory({
            type: "interaction",
            content: `User: ${text}\nAI: ${aiResponse}`,
            timestamp: new Date().toISOString(),
          });
          return aiResponse;
        }
      }

      // If we haven't found a match yet, try fuzzy matching
      if (!matchedCommand) {
        // Special handling for search commands first
        if (text.toLowerCase().startsWith("search ")) {
          console.log("LocalReasoner: Detected search command");
          matchedCommand = "search";
        } else {
          // Try to match other commands with fuzzy matching
          let highestConfidence = 0;

          for (let cmd of this.commands) {
            if (!cmd.patterns || !Array.isArray(cmd.patterns)) continue;

            for (const pattern of cmd.patterns) {
              const { confidence } = this.intentProcessor.findBestMatch(text, [
                pattern,
              ]);
              if (confidence > highestConfidence) {
                highestConfidence = confidence;
                if (confidence > 0.6) {
                  matchedCommand = cmd.intent;
                }
              }
            }
          }
        }

        // If we found a fuzzy match, handle it
        if (matchedCommand) {
          let reply = await this.handleCommand(matchedCommand, entities, text);
          if (reply) return reply;
        }
      }

      // If no command matches, try semantic search in memories
      const memories = await this.memorySystem.searchMemories(text);
      if (memories.length > 0) {
        return `I remember something related:\n${memories
          .map((m) => m.content)
          .join("\n")}`;
      }

      // Learning mode - ask user to teach new command
      if (text.startsWith("teach")) {
        const teachMatch = text.match(/teach\s+(.+)/i);
        if (teachMatch) {
          const newPattern = teachMatch[1];
          await this.intentProcessor.savePattern("custom", newPattern);
          return "Thank you for teaching me! I'll remember that for next time 📚";
        }
        return "What would you like to teach me? Start with 'teach' followed by what you want me to learn.";
      }

      return "I'm not sure what you mean. Would you like to teach me how to respond to this?";
    } catch (error) {
      console.error("Error in EnhancedReasoner:", error);
      return "I encountered an error while processing that.";
    }
  }

  async handleCommand(intent, entities, text) {
    switch (intent) {
      case "greet": {
        const nameMemory = await this.memorySystem.searchMemories("name is");
        // Get the most recent name by taking the last memory with a name
        const name =
          nameMemory.length > 0
            ? nameMemory[nameMemory.length - 1].content.split("is")[1].trim()
            : "friend";
        return `Hey ${name}! How are you doing today? 🌟`;
      }

      case "name_query": {
        const nameMemories = await this.memorySystem.searchMemories("name is");
        if (nameMemories.length === 0) {
          return "I don't know your name yet. Would you like to introduce yourself?";
        }
        // Get the most recent name
        const latestMemory = nameMemories[nameMemories.length - 1];
        const name = latestMemory.content.split("is")[1].trim();
        return `Your name is ${name} 😊`;
      }

      case "remember_name": {
        if (entities.person) {
          const name = entities.person.trim();

          // Clear specifically name-related memories
          const memories = await this.memorySystem.getMemoriesByType(
            "personal"
          );
          await Promise.all(
            memories.map(async (memory) => {
              if (memory.content.toLowerCase().includes("name is")) {
                await this.memorySystem.clearMemory(memory.id);
              }
            })
          );

          // Save new name
          await this.memorySystem.addMemory("personal", `name is ${name}`, {
            type: "identity",
          });

          return `Nice to meet you, ${name}! I'll remember your name. 😊`;
        }
        return "I didn't quite catch your name — could you try saying 'my name is...' or 'I am...'?";
      }

      case "remind": {
        const reminderText = entities.subject;
        const reminderTime = entities.datetime;

        if (reminderText) {
          const reminder = reminderTime
            ? `${reminderText} at ${reminderTime}`
            : reminderText;

          await this.memorySystem.addMemory("reminder", reminder, {
            due: reminderTime || "unspecified",
            task: reminderText,
          });
          return `I'll remind you to ${reminder} 📝`;
        }
        return "What would you like me to remind you about?";
      }

      case "show_memory": {
        const allMemories = await this.memorySystem.getMemoriesByType(
          "personal"
        );
        return allMemories.length > 0
          ? `Here's what I remember:\n${allMemories
              .map((m) => m.content)
              .join("\n")}`
          : "I don't have any memories stored yet!";
      }

      case "show_reminders": {
        const reminders = await this.memorySystem.getMemoriesByType("reminder");
        if (reminders.length === 0) {
          return "You don't have any reminders set! 📝";
        }

        // Sort reminders by due date if available
        reminders.sort((a, b) => {
          const dateA = a.metadata?.due
            ? new Date(a.metadata.due)
            : new Date(0);
          const dateB = b.metadata?.due
            ? new Date(b.metadata.due)
            : new Date(0);
          return dateA - dateB;
        });

        // Format reminders nicely
        const formattedReminders = reminders.map((r, index) => {
          const dueStr =
            r.metadata?.due !== "unspecified" ? ` (${r.metadata.due})` : "";
          return `${index + 1}. ${r.content}${dueStr}`;
        });

        return `Here are your reminders 📝:\n${formattedReminders.join("\n")}`;
      }

      case "clear_memory": {
        await this.memorySystem.clearMemories();
        return "I've cleared all my memories as requested. 🧹";
      }

      case "note_add": {
        const noteMatch = text.match(
          /(?:remember this|save this note|take a note)\s*(.*)/i
        );
        const note =
          noteMatch && noteMatch[1] ? noteMatch[1].trim() : entities.subject;

        if (note) {
          await this.memorySystem.addMemory("note", note, {
            timestamp: Date.now(),
          });
          return `📝 Got it! I saved your note: "${note}"`;
        }
        return "What would you like me to note down?";
      }

      case "note_list": {
        const notes = await this.memorySystem.getMemoriesByType("note");
        return notes.length > 0
          ? `Here are your saved notes 🗒️:\n${notes
              .map((n, i) => `${i + 1}. ${n.content}`)
              .join("\n")}`
          : "You don't have any notes yet!";
      }

      case "time":
        return `🕒 Current time: ${new Date().toLocaleTimeString()}`;

      case "date":
        return `📅 Today's date: ${new Date().toLocaleDateString()}`;

      case "motivation": {
        const quotes = [
          "Believe in yourself — you're unstoppable! 💪",
          "Every day is a fresh start 🌅",
          "You've got this. Keep going! 🚀",
          "Your potential is limitless 🌠",
        ];
        return `Here's some inspiration:\n${
          quotes[Math.floor(Math.random() * quotes.length)]
        }`;
      }

      case "joke": {
        const jokes = [
          "Why don't skeletons fight each other? They don't have the guts! 😂",
          "Parallel lines have so much in common. It's a shame they'll never meet. 😅",
          "I told my computer I needed a break, and it said: 'No problem, I'll go to sleep.' 😴",
        ];
        return `Here's a joke to brighten your mood:\n${
          jokes[Math.floor(Math.random() * jokes.length)]
        }`;
      }

      case "weather": {
        const weatherReplies = [
          "Looks like it's sunny and calm today ☀️",
          "A bit cloudy, but perfect for a walk ☁️",
          "I think it's raining somewhere near you 🌧️",
        ];
        return `${
          weatherReplies[Math.floor(Math.random() * weatherReplies.length)]
        }`;
      }

      case "fact": {
        const facts = [
          "Honey never spoils — archaeologists found 3000-year-old honey still edible! 🍯",
          "Octopuses have three hearts 💖",
          "Bananas are berries, but strawberries aren't! 🍓",
          "A day on Venus is longer than a year on Venus 🌌",
        ];
        return `Here's an interesting fact:\n${
          facts[Math.floor(Math.random() * facts.length)]
        }`;
      }

      case "mood": {
        await this.memorySystem.addMemory("mood", text, {
          timestamp: Date.now(),
        });
        return "I understand how you feel. I'm here to listen and support you 💕";
      }

      case "call": {
        if (entities.subject && entities.datetime) {
          const callReminder = `call ${entities.subject} at ${entities.datetime}`;
          await this.memorySystem.addMemory("reminder", callReminder, {
            due: entities.datetime,
            type: "call",
            person: entities.subject,
          });
          return `I'll remind you to ${callReminder} 📞`;
        }
        return "Please specify who to call and when.";
      }

      case "thanks":
        return "You're always welcome 🌸";

      case "who_are_you":
        return "I'm Marco, your friendly AI buddy 🤖 — I'm here to chat, remember things for you, set reminders, search news, and help you stay organized!";

      case "search": {
        console.log("Handling search command...");
        // Extract query from text directly if starts with search
        const query = text.toLowerCase().startsWith("search ")
          ? text.slice(7).trim() // Remove "search " prefix
          : entities.subject || text.replace(/search|find|look up/i, "").trim();

        console.log("Search query:", query);

        if (!query) {
          return "What would you like me to search for?";
        }

        try {
          // For product searches like "Amazon products"
          if (
            query.toLowerCase().includes("amazon") ||
            query.toLowerCase().includes("product")
          ) {
            const searchTerm = query.replace(/amazon|products?/gi, "").trim();
            console.log("Searching for product:", searchTerm);
            const result = await NewsService.searchNews(
              `${searchTerm} product review`
            );
            if (!result.success) {
              return `Sorry, I couldn't find any product information about "${searchTerm}". ${result.error}`;
            }
            return `Here's what I found about "${searchTerm}":\n\n${NewsService.formatArticles(
              result.articles
            )}`;
          }

          // For general searches, use news search
          console.log("Performing general search for:", query);
          const result = await NewsService.searchNews(query);
          if (!result.success) {
            return `Sorry, I couldn't find any information about "${query}". ${result.error}`;
          }
          return `Here's what I found about "${query}":\n\n${NewsService.formatArticles(
            result.articles
          )}`;
        } catch (error) {
          console.error("Search error:", error);
          return `Sorry, I encountered an error while searching: ${error.message}`;
        }
      }

      case "get_headlines": {
        const category = entities.subject?.toLowerCase() || "";
        const validCategories = [
          "business",
          "entertainment",
          "general",
          "health",
          "science",
          "sports",
          "technology",
        ];

        if (category && !validCategories.includes(category)) {
          return `Please specify a valid news category: ${validCategories.join(
            ", "
          )}`;
        }

        const result = await NewsService.getTopHeadlines(category);
        if (!result.success) {
          return `Sorry, I couldn't fetch the headlines. ${result.error}`;
        }

        const categoryText = category ? ` in ${category}` : "";
        return `Here are the top headlines${categoryText}:\n\n${NewsService.formatArticles(
          result.articles
        )}`;
      }

      case "exit": {
        const nameMemory = await this.memorySystem.searchMemories("name is");
        const name =
          nameMemory.length > 0
            ? nameMemory[0].content.split("is")[1].trim()
            : "friend";
        return `Goodbye, ${name}! 👋 Take care!`;
      }

      default:
        return null;
    }
  }
}

export default EnhancedReasoner;
