import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import json
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process and chunk documents for RAG system"""
    
    def __init__(self):
        self.chunks = []
        self.metadata = []
    
    def process_loan_data(self, df: pd.DataFrame) -> List[Dict]:
        """Convert loan data into searchable document chunks"""
        documents = []
        
        # Create comprehensive loan insights document
        loan_insights = self._generate_loan_insights(df)
        documents.extend(loan_insights)
        
        # Create feature explanations
        feature_docs = self._generate_feature_explanations()
        documents.extend(feature_docs)
        
        # Create approval criteria documents
        criteria_docs = self._generate_approval_criteria(df)
        documents.extend(criteria_docs)
        
        # Create statistical summaries
        stats_docs = self._generate_statistical_summaries(df)
        documents.extend(stats_docs)
        
        return documents
    
    def _generate_loan_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Generate comprehensive loan insights"""
        insights = []
        
        # Overall approval statistics
        approval_rate = df['Loan_Status'].value_counts(normalize=True).get('Y', 0) * 100
        insights.append({
            'content': f"Overall loan approval rate is {approval_rate:.1f}%. "
                      f"Out of {len(df)} loan applications, {df['Loan_Status'].value_counts().get('Y', 0)} were approved.",
            'type': 'statistics',
            'category': 'approval_rates'
        })
        
        # Gender-based analysis
        if 'Gender' in df.columns:
            gender_approval = df.groupby('Gender')['Loan_Status'].apply(lambda x: (x == 'Y').mean() * 100)
            for gender, rate in gender_approval.items():
                insights.append({
                    'content': f"Loan approval rate for {gender} applicants is {rate:.1f}%.",
                    'type': 'demographics',
                    'category': 'gender_analysis'
                })
        
        # Income-based insights
        if 'ApplicantIncome' in df.columns:
            income_stats = df['ApplicantIncome'].describe()
            insights.append({
                'content': f"Applicant income statistics: Average income is ${income_stats['mean']:.0f}, "
                          f"median income is ${income_stats['50%']:.0f}, "
                          f"ranging from ${income_stats['min']:.0f} to ${income_stats['max']:.0f}.",
                'type': 'financial',
                'category': 'income_analysis'
            })
        
        # Property area analysis
        if 'Property_Area' in df.columns:
            area_approval = df.groupby('Property_Area')['Loan_Status'].apply(lambda x: (x == 'Y').mean() * 100)
            for area, rate in area_approval.items():
                insights.append({
                    'content': f"Loan approval rate in {area} areas is {rate:.1f}%.",
                    'type': 'geography',
                    'category': 'property_area'
                })
        
        return insights
    
    def _generate_feature_explanations(self) -> List[Dict]:
        """Generate explanations for each feature"""
        explanations = [
            {
                'content': "Gender affects loan approval with historical data showing varying approval rates between male and female applicants.",
                'type': 'feature_explanation',
                'category': 'gender'
            },
            {
                'content': "Marital status is considered as married applicants often have dual income sources, affecting loan approval probability.",
                'type': 'feature_explanation',
                'category': 'marital_status'
            },
            {
                'content': "Number of dependents impacts loan approval as it affects the applicant's financial obligations and repayment capacity.",
                'type': 'feature_explanation',
                'category': 'dependents'
            },
            {
                'content': "Education level influences loan approval with graduates typically having better approval rates due to higher income potential.",
                'type': 'feature_explanation',
                'category': 'education'
            },
            {
                'content': "Employment status (self-employed vs not self-employed) affects loan approval based on income stability and verification.",
                'type': 'feature_explanation',
                'category': 'employment'
            },
            {
                'content': "Applicant income is a primary factor in loan approval, with higher income generally leading to better approval chances.",
                'type': 'feature_explanation',
                'category': 'income'
            },
            {
                'content': "Co-applicant income adds to the total household income and improves loan approval probability.",
                'type': 'feature_explanation',
                'category': 'co_applicant_income'
            },
            {
                'content': "Loan amount requested affects approval based on the applicant's repayment capacity and income-to-loan ratio.",
                'type': 'feature_explanation',
                'category': 'loan_amount'
            },
            {
                'content': "Loan amount term (duration) impacts approval as longer terms mean lower monthly payments but higher total interest.",
                'type': 'feature_explanation',
                'category': 'loan_term'
            },
            {
                'content': "Credit history is crucial for loan approval, with good credit history significantly improving approval chances.",
                'type': 'feature_explanation',
                'category': 'credit_history'
            },
            {
                'content': "Property area (Urban/Semi-urban/Rural) affects loan approval based on property values and market conditions.",
                'type': 'feature_explanation',
                'category': 'property_area'
            }
        ]
        return explanations
    
    def _generate_approval_criteria(self, df: pd.DataFrame) -> List[Dict]:
        """Generate loan approval criteria documents"""
        criteria = [
            {
                'content': "Loan approval depends on multiple factors including credit history, income stability, debt-to-income ratio, and employment status.",
                'type': 'criteria',
                'category': 'general_criteria'
            },
            {
                'content': "Credit history is the most important factor in loan approval. Applicants with good credit history have significantly higher approval rates.",
                'type': 'criteria',
                'category': 'credit_importance'
            },
            {
                'content': "Income verification is crucial. Both applicant and co-applicant income are considered to assess repayment capacity.",
                'type': 'criteria',
                'category': 'income_verification'
            },
            {
                'content': "Debt-to-income ratio should typically be below 40% for loan approval. This includes existing debts and the new loan payment.",
                'type': 'criteria',
                'category': 'debt_ratio'
            },
            {
                'content': "Employment stability is important. Self-employed applicants may face additional scrutiny compared to salaried employees.",
                'type': 'criteria',
                'category': 'employment_stability'
            }
        ]
        return criteria
    
    def _generate_statistical_summaries(self, df: pd.DataFrame) -> List[Dict]:
        """Generate statistical summaries"""
        summaries = []
        
        # Correlation insights
        summaries.append({
            'content': "Strong positive correlation exists between applicant income and loan amount, indicating higher income applicants request larger loans.",
            'type': 'correlation',
            'category': 'income_loan_correlation'
        })
        
        # Distribution insights
        summaries.append({
            'content': "Loan amount distribution shows most applications are for medium-range loans, with few applications for very high amounts.",
            'type': 'distribution',
            'category': 'loan_amount_distribution'
        })
        
        return summaries

class VectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store"""
        self.documents = documents
        self.metadata = [doc for doc in documents]
        
        # Generate embeddings
        texts = [doc['content'] for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for relevant documents"""
        if self.index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}/faiss_index.bin")
        
        # Save metadata
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str):
        """Load vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}/faiss_index.bin")
        
        # Load metadata
        with open(f"{path}/metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
        
        logger.info(f"Vector store loaded from {path}")

class LLMInterface:
    """Interface for different LLM providers"""
    
    def __init__(self, provider: str = "huggingface", model_name: str = "microsoft/DialoGPT-medium"):
        self.provider = provider
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM based on provider"""
        if self.provider == "huggingface":
            self._initialize_huggingface()
        elif self.provider == "openai":
            self._initialize_openai()
        # Add more providers as needed
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face model"""
        try:
            # Use a lightweight model for demonstration
            self.model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                tokenizer="microsoft/DialoGPT-small",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Hugging Face model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face model: {e}")
            # Fallback to a simpler approach
            self.model = None
    
    def _initialize_openai(self):
        """Initialize OpenAI model"""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            openai.api_key = api_key
            logger.info("OpenAI API initialized")
        else:
            logger.warning("OpenAI API key not found")
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using the LLM"""
        if self.provider == "huggingface":
            return self._generate_huggingface(prompt, max_length)
        elif self.provider == "openai":
            return self._generate_openai(prompt, max_length)
        else:
            return self._generate_fallback(prompt)
    
    def _generate_huggingface(self, prompt: str, max_length: int) -> str:
        """Generate response using Hugging Face model"""
        if self.model is None:
            return self._generate_fallback(prompt)
        
        try:
            response = self.model(prompt, max_length=max_length, num_return_sequences=1)
            return response[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
            return self._generate_fallback(prompt)
    
    def _generate_openai(self, prompt: str, max_length: int) -> str:
        """Generate response using OpenAI"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return self._generate_fallback(prompt)
    
    def _generate_fallback(self, prompt: str) -> str:
        """Fallback response generation"""
        return "I understand your question about loan approval. Based on the available data, I can provide insights about loan approval factors, but I'm currently unable to generate a detailed response. Please try rephrasing your question or check the system configuration."

class RAGSystem:
    """Main RAG system combining retrieval and generation"""
    
    def __init__(self, llm_provider: str = "huggingface", embedding_model: str = "all-MiniLM-L6-v2"):
        self.vector_store = VectorStore(embedding_model)
        self.llm = LLMInterface(llm_provider)
        self.conversation_history = []
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the RAG system"""
        return """You are an intelligent loan approval assistant. You help users understand loan approval processes, 
        analyze loan data, and provide insights about loan approval factors. 

        You have access to loan approval data and can provide detailed explanations about:
        - Loan approval criteria and factors
        - Statistical insights from loan data
        - Feature importance in loan decisions
        - Recommendations for loan applicants

        Always provide accurate, helpful, and detailed responses based on the retrieved context.
        If you don't have specific information, acknowledge this and provide general guidance.
        """
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the RAG system"""
        self.vector_store.add_documents(documents)
    
    def query(self, user_query: str, include_history: bool = True) -> Dict[str, Any]:
        """Process user query and return response"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(user_query, k=3)
        
        # Prepare context
        context = self._prepare_context(relevant_docs)
        
        # Generate response
        response = self._generate_response(user_query, context, include_history)
        
        # Update conversation history
        self.conversation_history.append({
            'user_query': user_query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'retrieved_docs': len(relevant_docs)
        })
        
        return {
            'response': response,
            'sources': relevant_docs,
            'context_used': context,
            'timestamp': datetime.now().isoformat()
        }
    
    def _prepare_context(self, relevant_docs: List[Tuple[Dict, float]]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for doc, score in relevant_docs:
            context_parts.append(f"- {doc['content']} (Relevance: {score:.2f})")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, include_history: bool) -> str:
        """Generate response using LLM"""
        # Prepare prompt
        prompt = f"{self.system_prompt}\n\n"
        
        if include_history and self.conversation_history:
            prompt += "Previous conversation:\n"
            for entry in self.conversation_history[-2:]:  # Last 2 exchanges
                prompt += f"User: {entry['user_query']}\n"
                prompt += f"Assistant: {entry['response']}\n\n"
        
        prompt += f"Context from loan data:\n{context}\n\n"
        prompt += f"User question: {query}\n\n"
        prompt += "Provide a helpful, detailed response based on the context above:"
        
        # Generate response
        response = self.llm.generate_response(prompt)
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def save_system(self, path: str):
        """Save RAG system state"""
        os.makedirs(path, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(f"{path}/vector_store")
        
        # Save conversation history
        with open(f"{path}/conversation_history.json", 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        logger.info(f"RAG system saved to {path}")
    
    def load_system(self, path: str):
        """Load RAG system state"""
        # Load vector store
        self.vector_store.load(f"{path}/vector_store")
        
        # Load conversation history
        history_path = f"{path}/conversation_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.conversation_history = json.load(f)
        
        logger.info(f"RAG system loaded from {path}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem(llm_provider="huggingface")
    
    # Load sample data (this would be your actual loan data)
    sample_data = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male'],
        'Loan_Status': ['Y', 'N', 'Y'],
        'ApplicantIncome': [5000, 4000, 6000],
        'Property_Area': ['Urban', 'Rural', 'Urban']
    })
    
    # Process documents
    processor = DocumentProcessor()
    documents = processor.process_loan_data(sample_data)
    
    # Add documents to RAG system
    rag.add_documents(documents)
    
    # Test queries
    test_queries = [
        "What factors affect loan approval?",
        "What is the approval rate for different genders?",
        "How does income affect loan approval?"
    ]
    
    for query in test_queries:
        result = rag.query(query)
        print(f"Query: {query}")
        print(f"Response: {result['response']}")
        print(f"Sources: {len(result['sources'])}")
        print("-" * 50)
