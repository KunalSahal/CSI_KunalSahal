import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, FileText, BarChart3, TrendingUp, AlertTriangle, CheckCircle, XCircle, Brain, Database, MessageSquare, Upload, Download, Settings, Info, Users } from 'lucide-react';

const LoanRAGChatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: "Hello! I'm your AI Loan Approval Assistant. I have comprehensive knowledge about loan approval patterns, risk factors, and can help you understand loan eligibility criteria. Ask me anything about loan approvals, risk assessment, or data insights!",
      timestamp: new Date().toISOString()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [loanData, setLoanData] = useState(null);
  const [selectedModel, setSelectedModel] = useState('gpt-3.5-turbo');
  const messagesEndRef = useRef(null);

  // Simulated loan dataset insights
  const datasetInsights = {
    totalRecords: 614,
    approvalRate: 68.5,
    avgLoanAmount: 146000,
    topRiskFactors: [
      { factor: 'Credit History', impact: 85 },
      { factor: 'Income Stability', impact: 78 },
      { factor: 'Debt-to-Income Ratio', impact: 72 },
      { factor: 'Employment Status', impact: 65 }
    ],
    demographics: {
      male: 498,
      female: 116,
      married: 398,
      unmarried: 216,
      graduates: 480,
      nonGraduates: 134
    },
    modelPerformance: {
      accuracy: 0.847,
      precision: 0.823,
      recall: 0.795,
      f1Score: 0.809
    }
  };

  // Knowledge base for RAG responses
  const knowledgeBase = {
    'loan approval factors': {
      content: "Key factors affecting loan approval include: Credit History (most important with 85% impact), Income Stability (78% impact), Debt-to-Income Ratio (72% impact), Employment Status (65% impact), Loan Amount, Property Area, and Educational Background. Our analysis shows that applicants with good credit history have an 89% approval rate compared to 32% for those with poor credit.",
      confidence: 0.95
    },
    'credit history importance': {
      content: "Credit history is the most critical factor in loan approval decisions. From our dataset analysis: 89% of applicants with good credit history get approved, while only 32% with poor credit history are approved. This represents a 57% difference in approval rates, making it the strongest predictor of loan approval.",
      confidence: 0.98
    },
    'income requirements': {
      content: "Income analysis shows that the average approved loan amount is ₹146,000. Applicants with higher income levels (>₹5000) have a 78% approval rate. Self-employed individuals face slightly lower approval rates (65%) compared to salaried employees (72%) due to income stability concerns.",
      confidence: 0.92
    },
    'demographics impact': {
      content: "Demographic analysis reveals: Male applicants (81%) dominate the dataset, married applicants (65%) have higher approval rates than unmarried (58%), and graduates (78%) have significantly better approval rates than non-graduates (52%). Urban areas show 71% approval rates vs 65% in rural areas.",
      confidence: 0.89
    },
    'risk assessment': {
      content: "Our ML model identifies high-risk applicants with 84.7% accuracy. Key risk indicators include: Missing credit history, high debt-to-income ratios (>50%), unstable employment, and loan amounts exceeding 80% of annual income. The model uses ensemble methods combining Random Forest and XGBoost for optimal performance.",
      confidence: 0.93
    }
  };

  // Simulate RAG retrieval and response generation
  const generateRAGResponse = async (query) => {
    setIsTyping(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const queryLower = query.toLowerCase();
    let response = "I apologize, but I don't have specific information about that topic in my knowledge base. However, I can help you with loan approval factors, credit history analysis, income requirements, demographic impacts, or risk assessment strategies.";
    let confidence = 0.3;
    
    // Simple keyword matching for demo purposes
    for (const [key, value] of Object.entries(knowledgeBase)) {
      if (queryLower.includes(key.split(' ')[0]) || queryLower.includes(key)) {
        response = value.content;
        confidence = value.confidence;
        break;
      }
    }
    
    // Add model-specific insights
    if (queryLower.includes('model') || queryLower.includes('prediction') || queryLower.includes('accuracy')) {
      response += ` Our current ML model achieves ${datasetInsights.modelPerformance.accuracy * 100}% accuracy with ${datasetInsights.modelPerformance.precision * 100}% precision and ${datasetInsights.modelPerformance.recall * 100}% recall.`;
      confidence = Math.max(confidence, 0.91);
    }
    
    setIsTyping(false);
    return { response, confidence };
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;
    
    const userMessage = {
      id: messages.length + 1,
      type: 'user',
      text: inputText,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    
    const { response, confidence } = await generateRAGResponse(inputText);
    
    const botMessage = {
      id: messages.length + 2,
      type: 'bot',
      text: response,
      confidence: confidence,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, botMessage]);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const quickQuestions = [
    "What are the key factors for loan approval?",
    "How important is credit history?",
    "What are the income requirements?",
    "How does the ML model work?",
    "What are the main risk factors?"
  ];

  const handleQuickQuestion = (question) => {
    setInputText(question);
  };

  const DataInsights = () => (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 font-medium">Total Records</p>
              <p className="text-2xl font-bold text-blue-800">{datasetInsights.totalRecords}</p>
            </div>
            <Database className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 font-medium">Approval Rate</p>
              <p className="text-2xl font-bold text-green-800">{datasetInsights.approvalRate}%</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>
        
        <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-yellow-600 font-medium">Avg Loan Amount</p>
              <p className="text-2xl font-bold text-yellow-800">₹{datasetInsights.avgLoanAmount / 1000}K</p>
            </div>
            <TrendingUp className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-purple-600 font-medium">Model Accuracy</p>
              <p className="text-2xl font-bold text-purple-800">{Math.round(datasetInsights.modelPerformance.accuracy * 100)}%</p>
            </div>
            <Brain className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-orange-500" />
            Top Risk Factors
          </h3>
          <div className="space-y-3">
            {datasetInsights.topRiskFactors.map((factor, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm font-medium">{factor.factor}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-orange-500 h-2 rounded-full" 
                      style={{ width: `${factor.impact}%` }}
                    ></div>
                  </div>
                  <span className="text-sm text-gray-600">{factor.impact}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-blue-500" />
            Model Performance
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Accuracy</span>
              <span className="text-sm font-bold text-green-600">{Math.round(datasetInsights.modelPerformance.accuracy * 100)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Precision</span>
              <span className="text-sm font-bold text-blue-600">{Math.round(datasetInsights.modelPerformance.precision * 100)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Recall</span>
              <span className="text-sm font-bold text-purple-600">{Math.round(datasetInsights.modelPerformance.recall * 100)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">F1-Score</span>
              <span className="text-sm font-bold text-orange-600">{Math.round(datasetInsights.modelPerformance.f1Score * 100)}%</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Users className="w-5 h-5 mr-2 text-indigo-500" />
          Dataset Demographics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{datasetInsights.demographics.male}</div>
            <div className="text-sm text-gray-600">Male</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-pink-600">{datasetInsights.demographics.female}</div>
            <div className="text-sm text-gray-600">Female</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{datasetInsights.demographics.married}</div>
            <div className="text-sm text-gray-600">Married</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{datasetInsights.demographics.unmarried}</div>
            <div className="text-sm text-gray-600">Unmarried</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{datasetInsights.demographics.graduates}</div>
            <div className="text-sm text-gray-600">Graduates</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-indigo-600">{datasetInsights.demographics.nonGraduates}</div>
            <div className="text-sm text-gray-600">Non-Graduates</div>
          </div>
        </div>
      </div>
    </div>
  );

  const ChatInterface = () => (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xs lg:max-w-md xl:max-w-lg ${message.type === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-800'} rounded-lg p-3`}>
              <div className="flex items-start space-x-2">
                {message.type === 'bot' && <Bot className="w-5 h-5 mt-0.5 text-blue-500" />}
                {message.type === 'user' && <User className="w-5 h-5 mt-0.5 text-white" />}
                <div className="flex-1">
                  <p className="text-sm">{message.text}</p>
                  <div className="flex items-center justify-between mt-2">
                    <p className={`text-xs ${message.type === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                      {formatTimestamp(message.timestamp)}
                    </p>
                    {message.confidence && (
                      <div className="flex items-center space-x-1">
                        <div className={`w-2 h-2 rounded-full ${message.confidence > 0.9 ? 'bg-green-500' : message.confidence > 0.7 ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
                        <span className="text-xs text-gray-500">{Math.round(message.confidence * 100)}%</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-gray-100 text-gray-800 rounded-lg p-3 max-w-xs">
              <div className="flex items-center space-x-2">
                <Bot className="w-5 h-5 text-blue-500" />
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="p-4 border-t border-gray-200">
        <div className="mb-3">
          <div className="flex items-center space-x-2 mb-2">
            <Info className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-700">Quick Questions:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {quickQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleQuickQuestion(question)}
                className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
        
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Ask about loan approval patterns, risk factors, or data insights..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        
        <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
          <span>Powered by RAG + {selectedModel}</span>
          <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            className="text-xs border border-gray-200 rounded px-2 py-1"
          >
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            <option value="gpt-4">GPT-4</option>
            <option value="claude-3-sonnet">Claude 3 Sonnet</option>
            <option value="gemini-pro">Gemini Pro</option>
          </select>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Brain className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Intelligent Loan Approval Assistant</h1>
                <p className="text-sm text-gray-600">RAG-powered Q&A system with ML insights</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Online</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              <button
                onClick={() => setActiveTab('chat')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'chat' 
                    ? 'border-blue-500 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <MessageSquare className="w-4 h-4 inline mr-2" />
                Chat Assistant
              </button>
              <button
                onClick={() => setActiveTab('insights')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'insights' 
                    ? 'border-blue-500 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <BarChart3 className="w-4 h-4 inline mr-2" />
                Data Insights
              </button>
            </nav>
          </div>

          <div className="h-96">
            {activeTab === 'chat' ? <ChatInterface /> : <DataInsights />}
          </div>
        </div>

        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <div className="flex items-center space-x-2 mb-2">
              <FileText className="w-4 h-4 text-blue-500" />
              <span className="font-medium text-blue-800">RAG Technology</span>
            </div>
            <p className="text-blue-700">Retrieval-Augmented Generation for accurate, context-aware responses</p>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <div className="flex items-center space-x-2 mb-2">
              <Brain className="w-4 h-4 text-green-500" />
              <span className="font-medium text-green-800">ML Insights</span>
            </div>
            <p className="text-green-700">Advanced machine learning models with 84.7% accuracy</p>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="w-4 h-4 text-purple-500" />
              <span className="font-medium text-purple-800">Real-time Analysis</span>
            </div>
            <p className="text-purple-700">Live data processing and intelligent risk assessment</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoanRAGChatbot;