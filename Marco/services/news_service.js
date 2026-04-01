// news_service.js
import axios from "axios";

class NewsService {
  constructor() {
    // We'll use NewsAPI for news content
    this.newsApiKey = "92dded37c1214d1ab59cc5c406b73d1e"; // Replace with your actual API key
    this.newsApiBaseUrl = "https://newsapi.org/v2";
  }

  async searchNews(query, options = {}) {
    try {
      console.log("NewsService: Searching for:", query);
      const params = {
        q: query,
        apiKey: this.newsApiKey,
        language: "en",
        sortBy: options.sortBy || "relevancy",
        pageSize: options.pageSize || 10,
        ...options,
      };
      console.log("NewsService: Request params:", { ...params, apiKey: "***" });

      const response = await axios.get(`${this.newsApiBaseUrl}/everything`, {
        params,
      });

      if (response.data.status !== "ok") {
        throw new Error(response.data.message || "Failed to fetch news");
      }

      return {
        success: true,
        articles: response.data.articles.map((article) => ({
          title: article.title,
          description: article.description,
          url: article.url,
          source: article.source.name,
          publishedAt: new Date(article.publishedAt).toLocaleDateString(),
          imageUrl: article.urlToImage,
        })),
      };
    } catch (error) {
      console.error("News API error:", error);
      return {
        success: false,
        error: error.message || "Failed to fetch news",
      };
    }
  }

  async getTopHeadlines(category = "", country = "us") {
    try {
      const params = {
        apiKey: this.newsApiKey,
        country,
        pageSize: 5,
      };

      if (category) {
        params.category = category;
      }

      const response = await axios.get(`${this.newsApiBaseUrl}/top-headlines`, {
        params,
      });

      if (response.data.status !== "ok") {
        throw new Error(response.data.message || "Failed to fetch headlines");
      }

      return {
        success: true,
        articles: response.data.articles.map((article) => ({
          title: article.title,
          description: article.description,
          url: article.url,
          source: article.source.name,
          publishedAt: new Date(article.publishedAt).toLocaleDateString(),
          imageUrl: article.urlToImage,
        })),
      };
    } catch (error) {
      console.error("News API error:", error);
      return {
        success: false,
        error: error.message || "Failed to fetch headlines",
      };
    }
  }

  formatArticles(articles) {
    return articles
      .map(
        (article, index) =>
          `${article.title}\n` +
          `${article.description || "No description available."}\n` +
          `Source: ${article.source} | ${article.publishedAt}\n` +
          `Link ${index + 1}: Read more\n`
      )
      .join("\n\n");
  }
}

export default new NewsService();
