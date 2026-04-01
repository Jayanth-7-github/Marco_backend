const knowledgeBase = {
  // Technology
  "elon musk": {
    content: `Elon Musk is a technology entrepreneur and CEO known for:
    • Tesla - Electric vehicle and clean energy company
    • SpaceX - Private space exploration company
    • X (formerly Twitter) - Social media platform
    • Neuralink - Brain-computer interface company
    • The Boring Company - Infrastructure and tunnel construction
    Key achievements include revolutionizing electric cars, developing reusable rockets, and advancing space exploration.`,
    category: "technology, business",
  },
  spacex: {
    content: `SpaceX (Space Exploration Technologies Corp.) is:
    • Founded by Elon Musk in 2002
    • Known for developing reusable rockets (Falcon 9)
    • Created the Starship spacecraft for Mars missions
    • Operates Starlink satellite internet constellation
    • First private company to send humans to the ISS
    Major goals include reducing space transportation costs and enabling Mars colonization.`,
    category: "technology, space",
  },
  tesla: {
    content: `Tesla, Inc. is a leading electric vehicle and clean energy company:
    • Products: Model S, Model 3, Model X, Model Y, Cybertruck
    • Also produces solar panels and battery storage systems
    • Known for advanced autopilot technology
    • Focuses on sustainable energy solutions
    • Market leader in electric vehicle technology`,
    category: "technology, automotive",
  },
  "artificial intelligence": {
    content: `Artificial Intelligence (AI) encompasses:
    • Machine Learning - Systems that learn from data
    • Neural Networks - Brain-inspired computing models
    • Natural Language Processing - Understanding human language
    • Computer Vision - Visual information processing
    • Robotics - Automated physical systems
    Current applications include virtual assistants, autonomous vehicles, and medical diagnosis.`,
    category: "technology",
  },

  // Science & Environment
  "climate change": {
    content: `Climate change refers to long-term shifts in global weather patterns:
    • Causes: Greenhouse gas emissions, deforestation, industrial activities
    • Effects: Rising temperatures, sea levels, extreme weather
    • Solutions: Renewable energy, carbon reduction, conservation
    • Global Impact: Affects ecosystems, agriculture, and human health
    • International Efforts: Paris Agreement and climate accords`,
    category: "science, environment",
  },

  // Health
  "covid-19": {
    content: `COVID-19 (Coronavirus Disease 2019):
    • Caused by SARS-CoV-2 virus
    • Global pandemic began in 2019
    • Symptoms: Respiratory issues, fever, loss of taste/smell
    • Prevention: Vaccination, masks, social distancing
    • Long-term impacts on global health and economy`,
    category: "health",
  },

  // Finance & Technology
  cryptocurrency: {
    content: `Cryptocurrency is digital or virtual currency:
    • Bitcoin - First and most well-known cryptocurrency
    • Blockchain - Underlying technology for security
    • Digital Wallets - For storing and trading
    • Decentralized Finance (DeFi) - New financial systems
    • NFTs - Digital assets and ownership`,
    category: "finance, technology",
  },

  // Additional Technology Topics
  "quantum computing": {
    content: `Quantum Computing uses quantum mechanics for computation:
    • Qubits - Quantum bits for processing
    • Superposition - Multiple states simultaneously
    • Applications: Cryptography, drug discovery, optimization
    • Major Players: IBM, Google, Microsoft
    • Current Status: Early development stage`,
    category: "technology",
  },
  "5g technology": {
    content: `5G is the fifth generation of mobile networks:
    • Faster speeds than 4G (up to 20 Gbps)
    • Lower latency for real-time applications
    • Enables IoT and smart cities
    • Network slicing capabilities
    • Improved mobile broadband`,
    category: "technology",
  },

  // Science Topics
  "renewable energy": {
    content: `Renewable Energy includes sustainable power sources:
    • Solar Power - Converting sunlight to electricity
    • Wind Energy - Harnessing wind power
    • Hydroelectric - Water-based power generation
    • Geothermal - Earth's heat for power
    • Benefits: Clean, sustainable, cost-effective`,
    category: "science, environment",
  },

  // Business & Technology
  metaverse: {
    content: `The Metaverse is a virtual shared space:
    • Virtual Reality (VR) and Augmented Reality (AR)
    • Digital economies and virtual assets
    • Social interaction and entertainment
    • Business and education applications
    • Major investments by tech companies`,
    category: "technology, business",
  },

  // Popular Culture
  "social media": {
    content: `Social Media platforms for digital communication:
    • Major Platforms: Instagram, Facebook, X, TikTok
    • Features: Content sharing, messaging, networking
    • Impact: Society, business, politics
    • Concerns: Privacy, mental health, misinformation
    • Trends: Short-form video, AR filters`,
    category: "technology, society",
  },
};

// Categories for organizing knowledge
const categories = {
  technology: [
    "artificial intelligence",
    "quantum computing",
    "5g technology",
    "social media",
  ],
  business: ["elon musk", "tesla", "metaverse"],
  science: ["climate change", "renewable energy"],
  health: ["covid-19"],
  finance: ["cryptocurrency"],
  space: ["spacex"],
};

// Helper function to get topics by category
export const getTopicsByCategory = (category) => {
  return categories[category] || [];
};

// Helper function to get related topics
export const getRelatedTopics = (topic) => {
  const topicData = knowledgeBase[topic.toLowerCase()];
  if (!topicData) return [];

  const topicCategories = topicData.category.split(", ");
  const related = new Set();

  topicCategories.forEach((category) => {
    if (categories[category]) {
      categories[category].forEach((relatedTopic) => {
        if (relatedTopic !== topic.toLowerCase()) {
          related.add(relatedTopic);
        }
      });
    }
  });

  return Array.from(related);
};

// Helper function to search topics
export const searchTopics = (query) => {
  query = query.toLowerCase();
  const results = [];

  // Direct match
  if (knowledgeBase[query]) {
    results.push({
      topic: query,
      content: knowledgeBase[query].content,
      relevance: 1,
    });
  }

  // Partial matches
  Object.keys(knowledgeBase).forEach((topic) => {
    if (
      topic !== query &&
      (topic.includes(query) ||
        knowledgeBase[topic].content.toLowerCase().includes(query))
    ) {
      results.push({
        topic,
        content: knowledgeBase[topic].content,
        relevance: topic.includes(query) ? 0.8 : 0.5,
      });
    }
  });

  return results.sort((a, b) => b.relevance - a.relevance);
};

export default knowledgeBase;
