import AsyncStorage from "@react-native-async-storage/async-storage";

// Enhanced intent processor with learning capabilities
export class IntentProcessor {
  constructor() {
    this.threshold = 0.7; // Confidence threshold
    this.learnedPatterns = new Map();
  }

  // Load learned patterns from storage
  async initialize() {
    try {
      const stored = await AsyncStorage.getItem("learned_patterns");
      if (stored) {
        this.learnedPatterns = new Map(JSON.parse(stored));
      }
    } catch (error) {
      console.error("Error loading learned patterns:", error);
    }
  }

  // Save new learned patterns
  async savePattern(intent, pattern) {
    try {
      this.learnedPatterns.set(intent, pattern);
      await AsyncStorage.setItem(
        "learned_patterns",
        JSON.stringify(Array.from(this.learnedPatterns.entries()))
      );
    } catch (error) {
      console.error("Error saving pattern:", error);
    }
  }

  // Calculate similarity between two strings
  calculateSimilarity(str1, str2) {
    const s1 = str1.toLowerCase();
    const s2 = str2.toLowerCase();

    // Use Levenshtein distance for similarity
    const matrix = Array(s2.length + 1)
      .fill(null)
      .map(() => Array(s1.length + 1).fill(null));

    for (let i = 0; i <= s1.length; i += 1) {
      matrix[0][i] = i;
    }

    for (let j = 0; j <= s2.length; j += 1) {
      matrix[j][0] = j;
    }

    for (let j = 1; j <= s2.length; j += 1) {
      for (let i = 1; i <= s1.length; i += 1) {
        const indicator = s1[i - 1] === s2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,
          matrix[j - 1][i] + 1,
          matrix[j - 1][i - 1] + indicator
        );
      }
    }

    const maxLength = Math.max(s1.length, s2.length);
    return 1 - matrix[s2.length][s1.length] / maxLength;
  }

  // Match input against learned patterns
  findBestMatch(input, patterns) {
    let bestMatch = null;
    let highestSimilarity = 0;

    const normalizedInput = input.toLowerCase().trim();

    for (const pattern of patterns) {
      const normalizedPattern = pattern.toLowerCase();
      // Handle wildcard patterns (ending with *)
      if (pattern.endsWith("*")) {
        const prefix = normalizedPattern.slice(0, -1);
        if (normalizedInput.startsWith(prefix)) {
          return {
            match: pattern,
            confidence: 1.0,
          };
        }
      }

      // Handle exact includes
      if (normalizedInput.includes(normalizedPattern)) {
        return {
          match: pattern,
          confidence: 1.0,
        };
      }

      // For search patterns, be more lenient
      if (
        normalizedPattern.includes("search") &&
        normalizedInput.includes("search")
      ) {
        const similarity = this.calculateSimilarity(
          normalizedInput,
          normalizedPattern
        );
        if (similarity > 0.5) {
          // More lenient threshold for search
          return {
            match: pattern,
            confidence: similarity,
          };
        }
      }

      // Otherwise calculate similarity
      const similarity = this.calculateSimilarity(
        normalizedInput,
        normalizedPattern
      );
      if (similarity > highestSimilarity) {
        highestSimilarity = similarity;
        bestMatch = pattern;
      }
    }

    return {
      match: bestMatch,
      confidence: highestSimilarity,
    };
  }

  // Extract entities from user input
  extractEntities(input) {
    const entities = {
      datetime: null,
      location: null,
      person: null,
      action: null,
      subject: null,
    };

    // For search queries, extract everything after "search"/"find"/"look up"
    const searchMatch = input.match(/^(?:search|find|look up)\s+(.+)/i);
    if (searchMatch) {
      entities.subject = searchMatch[1].trim();
      entities.action = "search";
      console.log("Search entities extracted:", entities);
      return entities;
    }
    console.log("Input did not match search pattern:", input);

    // Time extraction
    const timeMatch = input.match(/\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b/i);
    if (timeMatch) entities.datetime = timeMatch[1];

    // Date extraction
    const dateMatch = input.match(
      /\b(today|tomorrow|yesterday|sunday|monday|tuesday|wednesday|thursday|friday|saturday|\d{1,2}\/\d{1,2}(?:\/\d{4})?)\b/i
    );
    if (dateMatch) {
      // If we already have a time, combine it with the date
      entities.datetime = entities.datetime
        ? `${dateMatch[1]} at ${entities.datetime}`
        : dateMatch[1];
    }

    // Location extraction
    const locationMatch = input.match(
      /\b(?:at|in|near|to)\s+([A-Za-z\s]+(?:\s*,\s*[A-Za-z\s]+)*)\b/i
    );
    if (locationMatch) entities.location = locationMatch[1];

    // Person extraction - Check for name introduction patterns first
    const nameIntroMatch = input.match(
      /^(?:i\s*am|i'm|i\s*a\s*m|iam|my\s*name\s*is|call\s*me)\s*([A-Za-z\s]+)(?:\s|$)/i
    );
    if (nameIntroMatch) {
      entities.person = nameIntroMatch[1].trim();
      // Flag this as a name introduction
      entities.isNameIntro = true;
    } else {
      // Fallback to general person extraction
      const personMatch = input.match(
        /\b(?:with|to|from)\s+([A-Za-z\s]+)(?:\s|$)/i
      );
      if (personMatch) entities.person = personMatch[1].trim();
    }

    // Action & subject extraction
    const reminderMatch = input.match(/(?:remind|remaind) me to\s+([^.!?,]+)/i);

    const actionMatch = input.match(
      /\b(?:call|message|schedule|buy|get|do|make)\s+([A-Za-z\s]+?)(?:\s+(?:at|on|by)\s|$)/i
    );

    if (reminderMatch) {
      entities.action = "remind";
      entities.subject = reminderMatch[1].trim();
    } else if (actionMatch) {
      if (input.toLowerCase().startsWith("call")) {
        entities.action = "call";
        entities.subject = actionMatch[1];
      } else {
        entities.action = actionMatch[1];
        entities.subject = actionMatch[2] || "";
      }
    }

    return entities;
  }
}

export default IntentProcessor;
