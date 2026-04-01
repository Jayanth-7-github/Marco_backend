import knowledgeBase, {
  searchTopics,
  getRelatedTopics,
  getTopicsByCategory,
} from "./knowledgeBase";
import NewsService from "../services/news_service";

class AdvancedSearch {
  constructor() {
    this.newsService = NewsService;
  }

  async search(query, options = {}) {
    const {
      includeNews = true,
      includeKnowledgeBase = true,
      category = null,
      limit = 10,
      sortBy = "relevance",
    } = options;

    const results = {
      knowledgeBase: [],
      news: [],
      related: [],
      category: null,
    };

    try {
      // Search local knowledge base
      if (includeKnowledgeBase) {
        const kbResults = searchTopics(query);
        results.knowledgeBase = kbResults.slice(0, limit);

        // Get related topics for the top result
        if (kbResults.length > 0) {
          results.related = getRelatedTopics(kbResults[0].topic);
        }

        // If category is specified, filter by category
        if (category) {
          const categoryTopics = getTopicsByCategory(category);
          results.knowledgeBase = results.knowledgeBase.filter((r) =>
            categoryTopics.includes(r.topic)
          );
        }
      }

      // Search news if requested
      if (includeNews) {
        const newsResults = await this.newsService.searchNews(query, {
          sortBy: sortBy === "date" ? "publishedAt" : "relevancy",
          pageSize: limit,
        });

        if (newsResults.success) {
          results.news = newsResults.articles;
        }
      }

      return {
        success: true,
        results,
        query,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error("Advanced search error:", error);
      return {
        success: false,
        error: error.message,
        query,
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Format search results into a readable response
  formatResults(searchResponse) {
    if (!searchResponse.success) {
      return `Sorry, I encountered an error while searching: ${searchResponse.error}`;
    }

    const { results, query } = searchResponse;
    let response = `Here's what I found about "${query}":\n\n`;

    // Knowledge base results
    if (results.knowledgeBase.length > 0) {
      response += "📚 From my knowledge base:\n";
      results.knowledgeBase.forEach((result) => {
        response += result.content + "\n\n";
      });
    }

    // News results
    if (results.news.length > 0) {
      response += "📰 Recent news:\n";
      response += this.newsService.formatArticles(results.news) + "\n";
    }

    // Related topics
    if (results.related.length > 0) {
      response += "🔄 Related topics you might be interested in:\n";
      response += results.related.map((topic) => `• ${topic}`).join("\n");
    }

    return response;
  }

  // Search with specific filters
  async searchByCategory(query, category) {
    return this.search(query, { category });
  }

  async searchNewsOnly(query) {
    return this.search(query, { includeKnowledgeBase: false });
  }

  async searchKnowledgeBaseOnly(query) {
    return this.search(query, { includeNews: false });
  }
}

export default new AdvancedSearch();
