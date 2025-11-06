import React, { useState, useEffect } from 'react';
import { User, DollarSign, TrendingUp, Mail, Settings, Brain, BarChart3, AlertCircle, CheckCircle, Clock, RefreshCw } from 'lucide-react';
import './finadvisor.css';

// Define TypeScript interfaces for your data structures
interface UserData {
  email: string;
  age: number;
  income?: number;
  expenses?: number;
  debt?: number;
  assets?: number;
  risk_tolerance: string;
  financial_goals?: string;
  plaid_access_token?: string;
}
interface Message {
  type: 'success' | 'error' | '';
  text: string;
}

interface MessageAlertProps {
  message: Message;
}


interface FinancialPlan {
  recommendations?: {
    stocks?: Array<{
      symbol: string;
      name: string;
      current_price: number;
      '5day_change_percent': number;
    }>;
    crypto?: Array<{
      symbol: string;
      name: string;
      current_price: number;
      '24h_change_percent': number;
    }>;
    sectors?: Array<{
      sector: string;
      etf_symbol: string;
      '30day_performance': number;
    }>;
  };
}
type StatusType = 'operational' | 'running' | 'healthy' | 'down' | 'error'| 'warning';
interface StatusBadgeProps {
  status: StatusType;
  label: string;
}
const API_BASE = 'http://127.0.0.1:8000';

const App = () => {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [user, setUser] = useState<UserData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<Message>({ type: '', text: '' });  
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [trendingData, setTrendingData] = useState<FinancialPlan | null>(null);
  const [financialPlan, setFinancialPlan] = useState<any>(null);

  // User registration/login form
  const [userForm, setUserForm] = useState<UserData>({
    email: '',
    age: 0,
    income: 0,
    expenses: 0,
    debt: 0,
    assets: 0,
    risk_tolerance: 'moderate',
    financial_goals: '',
    plaid_access_token: '',
  });

  // Auto-extract form
  const [extractForm, setExtractForm] = useState<Partial<UserData>>({
    email: '',
    age: 0,
    plaid_access_token: '',
    risk_tolerance: 'moderate',
    financial_goals: '',
  });

  // Investment alert form
  const [alertForm, setAlertForm] = useState({
    user_email: '',
    risk_tolerance: '',
    hour:0
  });

  useEffect(() => {
    checkHealthStatus();
  }, [activeTab]);

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 5000);
  };
  const checkHealthStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      const data = await response.json();
      setHealthStatus(data);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

 

const registerUser = async (e: React.FormEvent) => {
  e.preventDefault();
  setLoading(true);
  try {
    const response = await fetch(`${API_BASE}/users/register/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...userForm,
        age: Number(userForm.age),
        income: Number(userForm.income),
        expenses: Number(userForm.expenses),
        debt: Number(userForm.debt),
        assets: Number(userForm.assets),
      }),
    });
    
    const data = await response.json();
    if (response.ok) {
      setUser(data.user_profile);
      showMessage('success', 'User registered successfully!');
      setActiveTab('dashboard');
    } else {
      showMessage('error', data.detail || 'Registration failed');
    }
  } catch (error) {
    showMessage('error', 'Network error during registration');
  }finally{
    setLoading(false);
  }
  
};

  const autoExtractProfile = async (e: React.FormEvent) => {
  e.preventDefault();
  setLoading(true);
  try {
    const response = await fetch(`${API_BASE}/auto-extract-profile/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...extractForm,
        age: Number(extractForm.age), // Convert to number
      }),
    });
    
    const data = await response.json();
    if (response.ok) {
      setUser(data.user_profile);
      showMessage('success', 'Profile extracted successfully!');
      setActiveTab('dashboard');
    } else {
      showMessage('error', data.detail || 'Extraction failed');
    }
  } catch (error) {
    showMessage('error', 'Network error during extraction');
  }finally{
    setLoading(false);
  }
  
};
  const getTrendingInvestments = async () => {
    setLoading(true);
    try {
      const riskTolerance = user?.risk_tolerance || 'moderate';
      const response = await fetch(`${API_BASE}/trending-investments/?risk_tolerance=${riskTolerance}`);
      const data = await response.json();
      
      if (response.ok) {
        setTrendingData(data.trending_investments);
        showMessage('success', 'Trending investments loaded!');
      } else {
        showMessage('error', data.detail || 'Failed to load trending investments');
      }
    } catch (error) {
      showMessage('error', 'Network error loading trending investments');
    }
    setLoading(false);
  };

  const generateFinancialPlan = async (planType = 'enhanced') => {
    if (!user?.email) {
      showMessage('error', 'Please register or load a user profile first');
      return;
    }

    setLoading(true);
    try {
      let endpoint = '';
      switch (planType) {
        case 'enhanced':
          endpoint = `/enhanced-hybrid-plan/${user.email}`;
          break;
        case 'hybrid':
          endpoint = '/hybrid-plan/';
          break;
        default:
          endpoint = '/rl-only-plan/';
      }

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: planType === 'enhanced' ? undefined : JSON.stringify(user)
      });
      
      const data = await response.json();
      if (response.ok) {
        setFinancialPlan(data);
        showMessage('success', `${planType} financial plan generated!`);
      } else {
        showMessage('error', data.detail || 'Plan generation failed');
      }
    } catch (error) {
      showMessage('error', 'Network error generating plan');
    }
    setLoading(false);
  };

  const sendInvestmentAlert = async (e : React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/setup-daily-alerts/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(alertForm)
      });
      
      const data = await response.json();
      if (response.ok) {
        showMessage('success', 'Investment alert sent successfully!');
      } else {
        showMessage('error', data.detail || 'Alert sending failed');
      }
    } catch (error) {
      showMessage('error', 'Network error sending alert');
    }
    setLoading(false);
  };

  const MessageAlert: React.FC<MessageAlertProps> = ({ message }) => {
  if (!message.text) return null;
  
  const isSuccess = message.type === 'success';
  const Icon = isSuccess ? CheckCircle : AlertCircle;
  
  return (
    <div className={`message-alert ${isSuccess ? 'success' : 'error'}`}>
      <div className="message-content">
        <Icon className="message-icon" />
        {message.text}
      </div>
    </div>
  );
};


  const StatusBadge: React.FC<StatusBadgeProps> = ({ status, label }) => {
  const isOperational = status === 'operational' || status === 'running' || status === 'healthy';
  
  return (
    <span className={`status-badge ${isOperational ? 'operational' : 'error'}`}>
      {label}: {status}
    </span>
  );
};


  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-content">
          <div className="nav-header">
            <div className="nav-brand">
              <Brain className="brand-icon" />
              <span className="brand-text">AI Financial Advisor</span>
            </div>
            <div className="nav-user">
              {user && (
                <span className="user-welcome">Welcome, {user.email}</span>
              )}
            </div>
          </div>
        </div>
      </nav>

      <div className="main-content">
        <MessageAlert message={message} />

        {/* Tab Navigation */}
        <div className="tab-container">
          <div className="tab-nav">
            <nav className="tab-list">
              {[
                { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
                { id: 'register', label: 'Register', icon: User },
                { id: 'extract', label: 'Auto Extract', icon: DollarSign },
                { id: 'investments', label: 'Trending', icon: TrendingUp },
                { id: 'alerts', label: 'Alerts', icon: Mail },
                { id: 'status', label: 'System Status', icon: Settings }
              ].map(tab => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                  >
                    <Icon className="tab-icon" />
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div className="content-container">
          {activeTab === 'dashboard' && (
            <div>
              <h2 className="page-title">Financial Dashboard</h2>
              
              {user ? (
                <div className="dashboard-grid">
                  <div className="card profile-card">
                    <h3 className="card-title">Profile Summary</h3>
                    <p className="card-text">Age: {user.age}</p>
                    <p className="card-text">Risk: {user.risk_tolerance}</p>
                    <p className="card-text">Income: ${user.income?.toLocaleString()}</p>
                  </div>
                  
                  <div className="card assets-card">
                    <h3 className="card-title">Assets & Debt</h3>
                    <p className="card-text">Assets: ${user.assets?.toLocaleString()}</p>
                    <p className="card-text">Debt: ${user.debt?.toLocaleString()}</p>
                    <p className="card-text">Expenses: ${user.expenses?.toLocaleString()}</p>
                  </div>
                  
                  <div className="card actions-card">
                    <h3 className="card-title">Quick Actions</h3>
                    <div className="action-buttons">
                      <button 
                        onClick={() => generateFinancialPlan('enhanced')}
                        className="action-button"
                        disabled={loading}
                      >
                        Generate Enhanced Plan
                      </button>
                      <button 
                        onClick={getTrendingInvestments}
                        className="action-button"
                        disabled={loading}
                      >
                        Get Trending Investments
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <User className="empty-icon" />
                  <h3 className="empty-title">No user profile</h3>
                  <p className="empty-description">Get started by registering or auto-extracting your financial data.</p>
                </div>
              )}

              {financialPlan && (
                <div className="financial-plan-section">
                  <h3 className="section-title">Latest Financial Plan</h3>
                  <div className="plan-container">
                    <pre className="plan-content">
                      {JSON.stringify(financialPlan, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'register' && (
            <div>
              <h2 className="page-title">Register New User</h2>
              <div className="form-container">
                <div className="form-group">
                  <label className="form-label">Email</label>
                  <input
                    type="email"
                    required
                    value={userForm.email}
                    onChange={(e) => setUserForm({...userForm, email: e.target.value})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Age</label>
                  <input
                    type="number"
                    required
                    value={userForm.age}
                    onChange={(e) => setUserForm({...userForm, age: Number(e.target.value)})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Monthly Income</label>
                  <input
                    type="number"
                    step="0.01"
                    value={userForm.income}
                    onChange={(e) => setUserForm({...userForm, income: Number(e.target.value)})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Monthly Expenses</label>
                  <input
                    type="number"
                    step="0.01"
                    value={userForm.expenses}
                    onChange={(e) => setUserForm({...userForm, expenses: Number(e.target.value)})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Total Debt</label>
                  <input
                    type="number"
                    step="0.01"
                    value={userForm.debt}
                    onChange={(e) => setUserForm({...userForm, debt: Number(e.target.value)})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Assets</label>
                  <input
                    type="number"
                    step="0.01"
                    value={userForm.assets}
                    onChange={(e) => setUserForm({...userForm, assets: Number(e.target.value)})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Risk Tolerance</label>
                  <select
                    value={userForm.risk_tolerance}
                    onChange={(e) => setUserForm({...userForm, risk_tolerance: e.target.value})}
                    className="form-select"
                  >
                    <option value="conservative">Conservative</option>
                    <option value="moderate">Moderate</option>
                    <option value="aggressive">Aggressive</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label className="form-label">Plaid Access Token (Optional)</label>
                  <input
                    type="text"
                    value={userForm.plaid_access_token}
                    onChange={(e) => setUserForm({...userForm, plaid_access_token: e.target.value})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group full-width">
                  <label className="form-label">Financial Goals</label>
                  <textarea
                    value={userForm.financial_goals}
                    onChange={(e) => setUserForm({...userForm, financial_goals: e.target.value})}
                    className="form-textarea"
                    rows={3}
                  />
                </div>
                
                <div className="form-group full-width">
                  <button
                    type="button"
                    onClick={registerUser}
                    disabled={loading}
                    className="submit-button register-button"
                  >
                    {loading ? 'Registering...' : 'Register User'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'extract' && (
            <div>
              <h2 className="page-title">Auto-Extract Financial Data</h2>
              <div  className="form-container">
                <div className="form-group">
                  <label className="form-label">Email</label>
                  <input
                    type="email"
                    required
                    value={extractForm.email}
                    onChange={(e) => setExtractForm({...extractForm, email: e.target.value})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Age</label>
                  <input
                    type="number"
                    required
                    value={extractForm.age}
                    onChange={(e) => setExtractForm({...extractForm, age: Number(e.target.value)})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Plaid Access Token</label>
                  <input
                    type="text"
                    required
                    value={extractForm.plaid_access_token}
                    onChange={(e) => setExtractForm({...extractForm, plaid_access_token: e.target.value})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Risk Tolerance</label>
                  <select
                    value={extractForm.risk_tolerance}
                    onChange={(e) => setExtractForm({...extractForm, risk_tolerance: e.target.value})}
                    className="form-select"
                  >
                    <option value="conservative">Conservative</option>
                    <option value="moderate">Moderate</option>
                    <option value="aggressive">Aggressive</option>
                  </select>
                </div>
                
                <div className="form-group full-width">
                  <label className="form-label">Financial Goals</label>
                  <textarea
                    value={extractForm.financial_goals}
                    onChange={(e) => setExtractForm({...extractForm, financial_goals: e.target.value})}
                    className="form-textarea"
                    rows={3}
                  />
                </div>
                
                <div className="form-group full-width">
                  <button
                    type="submit"
                    onClick={autoExtractProfile}
                    disabled={loading}
                    className="submit-button extract-button"
                  >
                    {loading ? 'Extracting...' : 'Auto-Extract Profile'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'investments' && (
            <div>
              <h2 className="page-title">Trending Investments</h2>
              
              <div className="refresh-section">
                <button
                  onClick={getTrendingInvestments}
                  disabled={loading}
                  className="refresh-button"
                >
                  <RefreshCw className="refresh-icon" />
                  {loading ? 'Loading...' : 'Refresh Trending Data'}
                </button>
              </div>

              {trendingData && (
                <div className="trending-sections">
                  {trendingData.recommendations?.stocks && (
                    <div className="trending-section">
                      <h3 className="section-title">📈 Top Performing Stocks</h3>
                      <div className="investment-grid">
                        {trendingData.recommendations.stocks.map((stock, idx) => (
                          <div key={idx} className="investment-card stocks-card">
                            <h4 className="investment-symbol">{stock.symbol}</h4>
                            <p className="investment-name">{stock.name}</p>
                            <p className="investment-detail">Price: ${stock.current_price?.toFixed(2)}</p>
                            <p className="investment-detail">5d Change: {stock['5day_change_percent']?.toFixed(2)}%</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {trendingData.recommendations?.crypto && (
                    <div className="trending-section">
                      <h3 className="section-title">₿ Trending Cryptocurrencies</h3>
                      <div className="investment-grid">
                        {trendingData.recommendations.crypto.map((coin, idx) => (
                          <div key={idx} className="investment-card crypto-card">
                            <h4 className="investment-symbol">{coin.symbol}</h4>
                            <p className="investment-name">{coin.name}</p>
                            <p className="investment-detail">Price: ${coin.current_price?.toFixed(4)}</p>
                            <p className="investment-detail">24h Change: {coin['24h_change_percent']?.toFixed(2)}%</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {trendingData.recommendations?.sectors && (
                    <div className="trending-section">
                      <h3 className="section-title">🏢 Sector Performance</h3>
                      <div className="investment-grid">
                        {trendingData.recommendations.sectors.map((sector, idx) => (
                          <div key={idx} className="investment-card sectors-card">
                            <h4 className="investment-symbol">{sector.sector}</h4>
                            <p className="investment-name">ETF: {sector.etf_symbol}</p>
                            <p className="investment-detail">30d Performance: {sector['30day_performance']?.toFixed(2)}%</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === 'alerts' && (
            <div>
              <h2 className="page-title">Investment Alerts</h2>
              
              <form onSubmit={sendInvestmentAlert} className="alert-form">
                <div className="form-group">
                  <label className="form-label">Email Address</label>
                  <input
                    type="email"
                    required
                    value={alertForm.user_email}
                    onChange={(e) => setAlertForm({...alertForm, user_email: e.target.value})}
                    className="form-input"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Risk Tolerance</label>
                  <select
                    value={alertForm.risk_tolerance}
                    onChange={(e) => setAlertForm({...alertForm, risk_tolerance: e.target.value})}
                    className="form-select"
                  >
                    <option value="conservative">Conservative</option>
                    <option value="moderate">Moderate</option>
                    <option value="aggressive">Aggressive</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Hour Selection</label>
                    <select
                      value={alertForm.hour}  // Make sure your state has this field
                      onChange={(e) => setAlertForm({...alertForm, hour: Number(e.target.value)})}
                      className="form-select"
                    >
                    {Array.from({ length: 24 }, (_, i) => (
                    <option key={i} value={i}>
                      {i.toString().padStart(2, '0')}  {/* Formats as 00, 01, ..., 23 */}
                    </option>
                    ))}
                    </select>
                </div>
                
                <div className="form-group">
                  <button
                    type="submit"
                    disabled={loading}
                    className="submit-button alert-button"
                  >
                    {loading ? 'Sending...' : 'Send Alert'}
                  </button>
                </div>
              </form>
            </div>
          )}
          {activeTab === 'status' && (
            <div>
              <h2 className="page-title">System Status</h2>
              
              <div className="status-section">
                <div className="status-grid">
                  <div className="status-card">
                    <h3 className="status-card-title">API Health</h3>
                    <StatusBadge 
                      status={healthStatus?.status || 'unknown'} 
                      label="Status" 
                    />
                  </div>
                  
                  <div className="status-card">
                    <h3 className="status-card-title">Database</h3>
                    <StatusBadge 
                      status={healthStatus?.database || 'unknown'} 
                      label="Connection" 
                    />
                  </div>
                </div>

                {healthStatus && (
                  <div className="health-details">
                    <h3 className="details-title">System Details</h3>
                    <div className="details-container">
                      <pre className="details-content">
                        {JSON.stringify(healthStatus, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;