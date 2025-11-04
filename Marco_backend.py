import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from pymongo import MongoClient
import networkx as nx
import faiss
import spacy
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from scipy.stats import pearsonr
import joblib
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from collections import deque
import random

# Import neural network components
from neural_components import PatternNetwork, VectorMemoryManager, DecisionNetwork, SelfLearningModule

# Load spaCy model for enhanced NLP
nlp = spacy.load('en_core_web_sm')

# Initialize sentence transformer for better embeddings
sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Load environment variables
load_dotenv()

app = Flask(__name__)

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_concept(self, concept: str, attributes: Dict = None):
        self.graph.add_node(concept, **attributes if attributes else {})
        
    def add_relation(self, concept1: str, concept2: str, relation_type: str):
        self.graph.add_edge(concept1, concept2, relation=relation_type)
        
    def find_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        if concept not in self.graph:
            return []
        
        related = []
        for node in nx.descendants(self.graph, concept):
            path_length = len(nx.shortest_path(self.graph, concept, node)) - 1
            if path_length <= max_depth:
                related.append(node)
        return related

class MarcoBrain:
    def __init__(self):
        # Initialize the brain's components
        self.memories = []
        self.knowledge_graph = KnowledgeGraph()
        self.thought_history = []
        self.emotion_state = {"valence": 0.0, "arousal": 0.0}
        
        # Load language model for understanding and generation
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Initialize neural components
        self.embedding_dim = 384  # SentenceTransformer default
        self.pattern_recognizer = PatternNetwork(self.embedding_dim)
        self.vector_memory = VectorMemoryManager(self.embedding_dim)
        self.self_learning = SelfLearningModule(self.embedding_dim)
        
        # Initialize MongoDB connection
        self.mongo_client = MongoClient(os.getenv('MONGO_URI'))
        self.db = self.mongo_client['Marco']
        self.memory_collection = self.db['brain_memories']
        
        # Load existing memories
        self.load_memories_from_db()
        
    def process_input(self, input_text: str, context: Dict = None) -> Dict:
        """Process input with enhanced cognitive capabilities"""
        # Encode input
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        thought_embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Multi-step reasoning
        reasoning_steps = self._multi_step_reasoning(input_text, context)
        
        # Find patterns in recent interactions
        recent_patterns = self._detect_patterns([m["content"] for m in self.memories[-10:]])
        
        # Get knowledge graph associations
        graph_associations = self._get_knowledge_graph_insights(input_text)
        
        # Make predictions based on patterns and knowledge
        predictions = self._make_predictions(input_text, context)
        
        # Generate comprehensive response
        response = {
            "thoughts": reasoning_steps,
            "patterns_detected": recent_patterns,
            "knowledge_graph_insights": graph_associations,
            "predictions": predictions,
            "emotional_state": self.emotion_state,
            "memory_associations": self._find_associations(thought_embedding)
        }
        
        # Save interaction to memory
        self.learn(input_text, response)
        
        return response

    def _find_associations(self, current_thought: torch.Tensor) -> List[Dict]:
        """Find related memories using vector similarity search"""
        associations = []
        
        # Use vector memory for fast similarity search
        similar_memories = self.vector_memory.search_similar(current_thought.numpy(), k=5)
        
        for memory_text, similarity in similar_memories:
            if similarity < 20.0:  # FAISS uses L2 distance, lower is better
                associations.append({
                    "type": "memory",
                    "content": memory_text,
                    "similarity": 1.0 / (1.0 + similarity)  # Convert distance to similarity score
                })
        
        # Add pattern-based associations
        predicted_pattern = self.self_learning.predict_pattern(current_thought)
        pattern_based_memories = self.vector_memory.search_similar(predicted_pattern.numpy(), k=3)
        
        for memory_text, similarity in pattern_based_memories:
            if similarity < 20.0:
                associations.append({
                    "type": "pattern_based",
                    "content": memory_text,
                    "similarity": 1.0 / (1.0 + similarity)
                })
        
        return associations

    def _generate_response(self, input_text: str, associations: List[Dict]) -> Dict:
        """Generate a thoughtful response based on input and associations"""
        # Combine input understanding with associations
        response = {
            "thoughts": [],
            "emotions": self.emotion_state.copy(),
            "associations": associations,
            "decision": None
        }
        
        # Add analytical thoughts
        response["thoughts"].append({
            "type": "analytical",
            "content": f"Analyzing input: {input_text}"
        })
        
        # Add associative thoughts
        for assoc in associations:
            response["thoughts"].append({
                "type": "associative",
                "content": f"This reminds me of: {assoc['content']}"
            })
        
        # Make a decision based on analysis
        response["decision"] = self._make_decision(input_text, associations)
        
        return response

    def _update_emotional_state(self, input_text: str):
        """Update the emotional state based on input"""
        # Simple sentiment analysis (could be enhanced with proper NLP)
        positive_words = ["good", "happy", "excellent", "wonderful"]
        negative_words = ["bad", "sad", "terrible", "awful"]
        
        words = input_text.lower().split()
        
        # Adjust valence based on words
        for word in words:
            if word in positive_words:
                self.emotion_state["valence"] += 0.1
            elif word in negative_words:
                self.emotion_state["valence"] -= 0.1
        
        # Keep emotions in bounds
        self.emotion_state["valence"] = max(min(self.emotion_state["valence"], 1.0), -1.0)

    def _make_decision(self, input_text: str, associations: List[Dict]) -> str:
        """Make a decision using neural decision network and emotional context"""
        # Get input embedding
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Get decision probabilities
        decision_probs = self.self_learning.make_decision(embedding)
        decision_idx = decision_probs.argmax().item()
        confidence = decision_probs[0][decision_idx].item()
        
        # Map decision index to action
        decision_map = {
            0: "proceed with confidence",
            1: "proceed with caution",
            2: "gather more information",
            3: "suggest alternative approach",
            4: "recommend against proceeding"
        }
        
        # Combine neural decision with emotional state
        base_decision = decision_map[decision_idx]
        if self.emotion_state["valence"] > 0.5 and decision_idx < 2:
            confidence += 0.1
        elif self.emotion_state["valence"] < -0.5 and decision_idx > 2:
            confidence += 0.1
            
        return f"Based on analysis (confidence: {confidence:.2f}), I suggest to {base_decision}"

    def learn(self, information: str, response: Dict = None):
        """Enhanced learning with metadata and neural network training"""
        # Encode new information
        inputs = self.tokenizer(information, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Train neural networks on new information
        self.self_learning.train_on_batch(embedding)
        predicted_pattern = self.self_learning.predict_pattern(embedding)
        
        # Create memory entry with metadata
        memory = {
            "content": information,
            "embedding": embedding,
            "timestamp": datetime.now(),
            "response": response,
            "metadata": {
                "emotional_state": self.emotion_state.copy(),
                "patterns_detected": self._detect_patterns([information]),
                "predicted_patterns": predicted_pattern.numpy().tolist()
            }
        }
        
        # Store in vector memory and database
        self.vector_memory.add_memory(information, embedding.numpy())
        self.memories.append(memory)
        self.save_memory_to_db(memory)
        
        # Update knowledge graph
        self.knowledge_graph.add_concept(information, attributes={
            "type": "learned_info",
            "pattern_prediction": predicted_pattern.numpy().tolist()
        })
        
    def save_memory_to_db(self, memory: Dict):
        """Save memory to MongoDB"""
        memory_doc = {
            "content": memory["content"],
            "embedding": memory["embedding"].numpy().tolist(),
            "timestamp": memory["timestamp"],
            "metadata": memory["metadata"]
        }
        if memory.get("response"):
            memory_doc["response"] = memory["response"]
            
        self.memory_collection.insert_one(memory_doc)
        
    def load_memories_from_db(self):
        """Load memories from MongoDB"""
        stored_memories = self.memory_collection.find({})
        for mem in stored_memories:
            self.memories.append({
                "content": mem["content"],
                "embedding": torch.tensor(mem["embedding"]),
                "timestamp": mem["timestamp"],
                "metadata": mem.get("metadata", {}),
                "response": mem.get("response", None)
            })

    def introspect(self) -> Dict:
        """Return current internal state"""
        return {
            "emotional_state": self.emotion_state,
            "memory_count": len(self.memories),
            "recent_thoughts": self.thought_history[-5:] if self.thought_history else []
        }

    def _detect_patterns(self, data: List[str]) -> List[Dict]:
        """Detect patterns in a sequence of text data"""
        patterns = []
        
        if not data:
            return patterns
            
        # Detect repetition patterns
        word_frequencies = {}
        for text in data:
            words = text.lower().split()
            for word in words:
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
                
        # Find frequently occurring terms
        common_terms = {word: freq for word, freq in word_frequencies.items() 
                       if freq > 1 and len(word) > 3}
        if common_terms:
            patterns.append({
                "type": "repetition",
                "pattern": common_terms
            })
            
        # Detect sentiment patterns
        sentiment_scores = []
        for text in data:
            score = sum(1 for word in text.lower().split() 
                       if word in ["good", "great", "happy", "excellent"]) - \
                    sum(1 for word in text.lower().split() 
                        if word in ["bad", "poor", "sad", "terrible"])
            sentiment_scores.append(score)
            
        if all(score > 0 for score in sentiment_scores[-3:]):
            patterns.append({
                "type": "sentiment",
                "pattern": "consistently positive"
            })
        elif all(score < 0 for score in sentiment_scores[-3:]):
            patterns.append({
                "type": "sentiment",
                "pattern": "consistently negative"
            })
            
        return patterns
        
    def _multi_step_reasoning(self, input_text: str, context: Dict = None) -> List[Dict]:
        """Implement multi-step reasoning process"""
        steps = []
        
        # Step 1: Initial understanding
        steps.append({
            "step": 1,
            "type": "understanding",
            "content": f"Processing input: {input_text}"
        })
        
        # Step 2: Context integration
        if context:
            steps.append({
                "step": 2,
                "type": "context_integration",
                "content": "Integrating context with current input"
            })
        
        # Step 3: Pattern matching
        patterns = self._detect_patterns([input_text])
        if patterns:
            steps.append({
                "step": 3,
                "type": "pattern_recognition",
                "content": f"Detected patterns: {patterns}"
            })
        
        # Step 4: Knowledge application
        related_concepts = self.knowledge_graph.find_related_concepts(input_text)
        if related_concepts:
            steps.append({
                "step": 4,
                "type": "knowledge_application",
                "content": f"Applied related concepts: {related_concepts}"
            })
        
        return steps
        
    def _get_knowledge_graph_insights(self, input_text: str) -> List[Dict]:
        """Get insights from knowledge graph"""
        insights = []
        
        # Add input as a concept if new
        self.knowledge_graph.add_concept(input_text)
        
        # Find related concepts
        related = self.knowledge_graph.find_related_concepts(input_text)
        
        if related:
            insights.append({
                "type": "related_concepts",
                "concepts": related
            })
        
        return insights
        
    def _make_predictions(self, input_text: str, context: Dict = None) -> List[Dict]:
        """Make predictions based on patterns and knowledge"""
        predictions = []
        
        # Analyze recent memory patterns
        recent_memories = [m["content"] for m in self.memories[-5:]]
        patterns = self._detect_patterns(recent_memories)
        
        if patterns:
            predictions.append({
                "type": "pattern_based",
                "prediction": f"Based on recent patterns, expect: {patterns[-1]}"
            })
        
        # Use knowledge graph for conceptual predictions
        related_concepts = self.knowledge_graph.find_related_concepts(input_text)
        if related_concepts:
            predictions.append({
                "type": "concept_based",
                "prediction": f"Likely related topics: {', '.join(related_concepts[:3])}"
            })
        
        return predictions

# Initialize brain
marco_brain = MarcoBrain()

# API Routes
@app.route('/think', methods=['POST'])
def think():
    try:
        data = request.json
        response = marco_brain.process_input(
            data['input'],
            context=data.get('context')
        )
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/learn', methods=['POST'])
def learn():
    try:
        data = request.json
        marco_brain.learn(data['information'])
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/introspect', methods=['GET'])
def introspect():
    try:
        memory_count = len(marco_brain.memories)
        recent_thoughts = [m["content"] for m in marco_brain.memories[-5:]]
        knowledge_nodes = len(marco_brain.knowledge_graph.graph.nodes)
        
        return jsonify({
            "memory_count": memory_count,
            "recent_thoughts": recent_thoughts,
            "knowledge_graph_size": knowledge_nodes,
            "emotional_state": marco_brain.emotion_state
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    port = int(os.getenv('PYTHON_PORT', 5001))
    app.run(host='0.0.0.0', port=port)