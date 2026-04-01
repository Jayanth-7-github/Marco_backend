// utils/responseFormatter.js

class ResponseFormatter {
  // Emoji mapping for different types of content
  static contentEmojis = {
    search: "🔍",
    article: "📰",
    news: "📣",
    fact: "💡",
    science: "🔬",
    tech: "💻",
    health: "🏥",
    sports: "⚽",
    entertainment: "🎬",
    business: "💼",
    education: "📚",
    weather: "🌤️",
    travel: "✈️",
    food: "🍽️",
    music: "🎵",
    art: "🎨",
    nature: "🌿",
    error: "❌",
    success: "✅",
    warning: "⚠️",
    question: "❓",
    answer: "💭",
  };

  // Sentiment emojis for different tones
  static sentimentEmojis = {
    positive: ["😊", "🎉", "👍", "✨", "🌟"],
    negative: ["😔", "🤔", "💭", "📝", "🔧"],
    neutral: ["💡", "📌", "🔍", "💬", "ℹ️"],
  };

  /**
   * Format a search response with appropriate emojis and structure
   */
  static formatSearchResponse(response, query) {
    if (!response || typeof response !== "string") {
      return `${this.contentEmojis.error} Sorry, I couldn't find any information about that.`;
    }

    // Add professional header
    let formatted = `${this.contentEmojis.search} Research Results | Topic: "${query}"\n\n`;

    // Add introduction
    const intros = [
      "Based on comprehensive analysis, here are the relevant findings:",
      "After thorough research, I can provide the following information:",
      "Here's a detailed overview of the topic:",
      "Drawing from reliable sources, I can share these insights:",
    ];
    formatted += `${this.contentEmojis.info || "ℹ️"} ${
      intros[Math.floor(Math.random() * intros.length)]
    }\n\n`;

    // Split response into paragraphs
    const paragraphs = response.split("\n\n");

    // Format each paragraph with professional structure
    paragraphs.forEach((paragraph, index) => {
      if (paragraph.trim()) {
        const emoji = this.getRelevantEmoji(paragraph);
        // Add section numbering for better organization
        formatted += `${emoji} Section ${index + 1}: ${paragraph.trim()}\n\n`;
      }
    });

    return formatted.trim();
  }

  /**
   * Format factual information with emojis
   */
  static formatFact(fact) {
    return `${this.contentEmojis.fact} Important Information:\n\n📝 Details:\n${fact}\n\n✨ Context:\nThis information has been verified and is presented for your reference.`;
  }

  /**
   * Format a news article summary
   */
  static formatArticle(article) {
    if (!article) return "";

    return `${this.contentEmojis.article} Article Summary\n
📑 Title: ${article.title || "Untitled"}
${
  article.date
    ? `� Published: ${new Date(article.date).toLocaleDateString()}`
    : ""
}
${article.description ? `\n� Abstract:\n${article.description}` : ""}
${article.url ? `\n� Source Reference:\n${article.url}` : ""}
${article.keywords ? `\n🏷️ Key Topics: ${article.keywords.join(", ")}` : ""}
`;
  }

  /**
   * Format error messages with appropriate emoji
   */
  static formatError(error) {
    return `${this.contentEmojis.error} ${error}`;
  }

  /**
   * Add appropriate emojis based on content type
   */
  static getRelevantEmoji(content) {
    const contentLower = content.toLowerCase();

    // Check content keywords to determine appropriate emoji
    if (contentLower.includes("error") || contentLower.includes("failed")) {
      return this.contentEmojis.error;
    }
    if (contentLower.includes("warning")) {
      return this.contentEmojis.warning;
    }
    if (contentLower.match(/study|research|scientist/)) {
      return this.contentEmojis.science;
    }
    if (contentLower.match(/tech|software|computer|app|digital/)) {
      return this.contentEmojis.tech;
    }
    if (contentLower.match(/health|medical|doctor|patient/)) {
      return this.contentEmojis.health;
    }
    if (contentLower.match(/sport|game|player|team/)) {
      return this.contentEmojis.sports;
    }
    if (contentLower.match(/movie|film|actor|entertainment/)) {
      return this.contentEmojis.entertainment;
    }
    if (contentLower.match(/business|company|market|economy/)) {
      return this.contentEmojis.business;
    }
    // Default to a general info emoji if no specific category is found
    return this.contentEmojis.fact;
  }

  /**
   * Format a list of search results
   */
  static formatSearchResults(results) {
    if (!results || !Array.isArray(results)) {
      return this.formatError("No results found");
    }

    return results
      .map((result) => {
        return this.formatArticle(result);
      })
      .join("\n\n");
  }

  /**
   * Add sentiment emojis based on content tone
   */
  static addSentimentEmoji(content) {
    // Simple sentiment analysis based on keywords
    const positivePhrases = [
      "success",
      "great",
      "good",
      "happy",
      "best",
      "solved",
    ];
    const negativePhrases = [
      "error",
      "fail",
      "bad",
      "wrong",
      "issue",
      "problem",
    ];

    const contentLower = content.toLowerCase();
    let sentiment = "neutral";

    if (positivePhrases.some((phrase) => contentLower.includes(phrase))) {
      sentiment = "positive";
    } else if (
      negativePhrases.some((phrase) => contentLower.includes(phrase))
    ) {
      sentiment = "negative";
    }

    const emojis = this.sentimentEmojis[sentiment];
    const emoji = emojis[Math.floor(Math.random() * emojis.length)];

    return `${emoji} ${content}`;
  }

  /**
   * Format code snippets or technical content
   */
  static formatTechnicalContent(content) {
    return `${this.contentEmojis.tech} Technical Analysis\n\n📋 Overview:\n${content}\n\n💡 Key Points:\n• Ensure you understand each concept thoroughly\n• Reference official documentation when needed\n• Consider best practices for implementation`;
  }

  /**
   * Format a list of items with appropriate emojis
   */
  static formatList(items, listType = "general") {
    if (!items || !Array.isArray(items)) return "";

    const emoji = this.contentEmojis[listType] || "•";
    return items.map((item) => `${emoji} ${item}`).join("\n");
  }

  /**
   * Format a chat message response
   */
  static formatChatResponse(response) {
    if (!response) return this.formatError("No response available");

    // If response is already formatted with emojis, return as is
    if (response.match(/[\u{1F300}-\u{1F6FF}]/u)) {
      return response;
    }

    // Add appropriate emojis based on content
    return this.addSentimentEmoji(response);
  }
}

export default ResponseFormatter;
