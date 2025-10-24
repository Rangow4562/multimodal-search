import streamlit as st
import requests
import json
from datetime import datetime
import re


# Page configuration
st.set_page_config(
    page_title="Tower Assist",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for Tower Assist styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-container {
        background-color: #f8f9fa;
        min-height: 100vh;
    }
    
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        background-color: #f8f9fa;
    }
    
    .login-card {
        background: white;
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        max-width: 400px;
        width: 100%;
    }
    
    .tower-logo {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #4CAF50, #2196F3);
        border-radius: 15px;
        margin: 0 auto 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    
    .login-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    .header-container {
        background: white;
        padding: 1rem 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .header-logo {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #4CAF50, #2196F3);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: white;
    }
    
    .header-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin: 0;
    }
    
    .header-right {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .user-avatar {
        width: 35px;
        height: 35px;
        background: #2196F3;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    
    .welcome-section {
        text-align: center;
        padding: 3rem 2rem;
        background: white;
        margin: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .welcome-logo {
        width: 120px;
        height: 120px;
        background: linear-gradient(135deg, #4CAF50, #2196F3);
        border-radius: 20px;
        margin: 0 auto 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        font-weight: bold;
        color: white;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        font-size: 1.1rem;
        color: #7f8c8d;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .chat-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 2rem;
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
        padding: 1rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-header-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .chat-logo {
        width: 30px;
        height: 30px;
        background: rgba(255,255,255,0.2);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: bold;
    }
    
    .chat-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .chat-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .new-chat-btn {
        background: rgba(255,255,255,0.2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chat-messages {
        padding: 1.5rem;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .user-message {
        background: #00BFFF;
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 8px rgba(0,191,255,0.3);
        position: relative;
    }
    
    .user-message::before {
        content: "👤";
        position: absolute;
        right: -25px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 0.8rem;
    }
    
    .assistant-message {
        background: #f8f9fa;
        color: #2c3e50;
        padding: 1rem 1.25rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        position: relative;
        border-left: 4px solid #2196F3;
    }
    
    .assistant-message::before {
        content: "T";
        position: absolute;
        left: -25px;
        top: 50%;
        transform: translateY(-50%);
        width: 20px;
        height: 20px;
        background: #2196F3;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: bold;
        color: white;
    }
    
    .message-time {
        font-size: 0.75rem;
        color: #95a5a6;
        margin-top: 0.5rem;
    }
    
    .chat-input-container {
        padding: 1.5rem;
        background: #f8f9fa;
        border-top: 1px solid #e9ecef;
    }
    
    .input-row {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .chat-input {
        flex: 1;
        padding: 0.75rem 1rem;
        border: 2px solid #e9ecef;
        border-radius: 25px;
        font-size: 1rem;
        outline: none;
        transition: border-color 0.3s;
    }
    
    .chat-input:focus {
        border-color: #2196F3;
    }
    
    .send-btn {
        background: #2196F3;
        color: white;
        border: none;
        width: 45px;
        height: 45px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        transition: background 0.3s;
    }
    
    .send-btn:hover {
        background: #1976D2;
    }
    
    .tips-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    .sidebar-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .help-item {
        padding: 0.75rem 0;
        border-bottom: 1px solid #f1f3f4;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: #495057;
    }
    
    .help-item:last-child {
        border-bottom: none;
    }
    
    .help-icon {
        width: 20px;
        height: 20px;
        background: #2196F3;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        color: white;
    }
    
    .zendesk-button {
        background: #2196F3;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 500;
        transition: background 0.3s;
        border: none;
        cursor: pointer;
        width: 100%;
        justify-content: center;
    }
    
    .zendesk-button:hover {
        background: #1976D2;
        color: white;
    }
    
    .image-container {
        margin: 1rem 0;
        text-align: center;
    }
    
    .response-image {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin-top: 1rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 0.75rem 1rem;
        border: 2px solid #e9ecef;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2196F3;
        box-shadow: none;
    }
    
    .stButton > button {
        border-radius: 25px;
        background: #2196F3;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: background 0.3s;
    }
    
    .stButton > button:hover {
        background: #1976D2;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "messages" not in st.session_state:
    st.session_state.messages = []


# Backend API configuration
API_BASE_URL = "http://localhost:8000"


def validate_email(email):
    """Validate email domain"""
    if not email:
        return False, "Please enter an email address"
    
    valid_domains = ["@towercoldchain.com", "@coldchaintech.com"]
    if not any(email.endswith(domain) for domain in valid_domains):
        return False, "Please use a company email (@towercoldchain.com or @coldchaintech.com)"
    
    return True, ""


def call_backend_api(question, top_k=5, include_images=True):
    """Call the FastAPI backend to get RAG response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "top_k": top_k,
                "include_images": include_images
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None


def display_user_message(message_content, timestamp=None):
    """Display user message with appropriate styling"""
    time_str = timestamp.strftime("%H:%M") if timestamp else datetime.now().strftime("%H:%M")
    
    st.markdown(f"""
    <div class="user-message">
        {message_content}
        <div class="message-time">{time_str}</div>
    </div>
    """, unsafe_allow_html=True)


def display_assistant_message(response_data, timestamp=None):
    """Display assistant response with full JSON payload handling"""
    time_str = timestamp.strftime("%H:%M") if timestamp else datetime.now().strftime("%H:%M")
    
    # Extract answer text
    answer = response_data.get("answer", "")
    answer_html = response_data.get("answer_html", answer)
    
    # Display the main answer
    st.markdown(f"""
    <div class="assistant-message">
        {answer}
        <div class="message-time">{time_str}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display images if available
    images_used = response_data.get("images_used", [])
    if images_used:
        st.markdown("**📸 Related Images:**")
        for img_data in images_used:
            data_url = img_data.get("data_url", "")
            if data_url:
                st.markdown(f"""
                <div class="image-container">
                    <img src="{data_url}" class="response-image" alt="Response Image">
                </div>
                """, unsafe_allow_html=True)
    
    # Display sources if available
    sources = response_data.get("sources", [])
    if sources:
        with st.expander("📚 Sources"):
            for i, source in enumerate(sources, 1):
                st.write(f"**Source {i}:**")
                st.write(f"- Document: {source.get('id', 'Unknown')}")
                st.write(f"- Page: {source.get('page', 'Unknown')}")
                st.write(f"- Preview: {source.get('content_preview', 'No preview available')}")
                if source.get('has_images'):
                    st.write("- Contains images")
                if source.get('has_annotations'):
                    st.write("- Contains annotations")
                st.write("---")
    
    # Display raw JSON payload in debug mode (optional)
    if st.checkbox("Show JSON Response", key=f"debug_{timestamp}"):
        st.json(response_data)
    
    # Show response structure info
    with st.expander("📊 Response Structure", expanded=False):
        st.write(f"**Answer Length:** {len(answer)} characters")
        st.write(f"**Images Count:** {len(images_used)}")
        st.write(f"**Sources Count:** {len(sources)}")
        st.write(f"**Has HTML Answer:** {'Yes' if answer_html != answer else 'No'}")
        
        # Show image details
        if images_used:
            st.write("**Image Details:**")
            for i, img in enumerate(images_used, 1):
                st.write(f"- Image {i}: {img.get('data_url', 'No URL')[:50]}...")
        
        # Show source details
        if sources:
            st.write("**Source Details:**")
            for i, source in enumerate(sources, 1):
                st.write(f"- Source {i}: {source.get('id', 'Unknown')} (Page {source.get('page', 'Unknown')})")


def login_page():
    """Display login page"""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-card">
            <div class="tower-logo">T</div>
            <h1 class="login-title">Tower Assist Access</h1>
            <p class="login-subtitle">Enter your company email to access the knowledge assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="your.email@towercoldchain.com")
            submit_button = st.form_submit_button("Access Tower Assist", use_container_width=True)
            
            if submit_button:
                is_valid, error_msg = validate_email(email)
                if is_valid:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.markdown(f'<div class="error-message">{error_msg}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def chat_page():
    """Display main chat interface"""
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-left">
            <div class="header-logo">T</div>
            <div>
                <h1 class="header-title">Tower Assist</h1>
                <p class="header-subtitle">Cold Chain Knowledge Assistant</p>
            </div>
        </div>
        <div class="header-right">
            <div class="user-avatar">A</div>
            <span>@{st.session_state.user_email.split('@')[0]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("""
    <div class="welcome-section">
        <div class="welcome-logo">T</div>
        <h1 class="welcome-title">Welcome to Tower Assist</h1>
        <p class="welcome-text">
            Your intelligent assistant for temperature-controlled shipping solutions. 
            Get instant answers about cold chain logistics, technical specifications, and best practices.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat header
        st.markdown("""
        <div class="chat-header">
            <div class="chat-header-left">
                <div class="chat-logo">T</div>
                <div>
                    <h3 class="chat-title">Tower Assist</h3>
                    <p class="chat-subtitle">Knowledge Assistant</p>
                </div>
            </div>
            <button class="new-chat-btn" onclick="window.location.reload()">
                🔄 New Chat
            </button>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat messages
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        # Display initial message if no conversation
        if not st.session_state.messages:
            initial_message = {
                "answer": "I'm here to help you quickly find accurate, up-to-date information about our temperature-controlled shippers. Just ask me a question, and I'll pull from our internal knowledge to get you the answer. I currently support technical queries only — for logistics or sales, please reach out via Zendesk. If I'm unsure or can't find the right info, I'll let you know — and you can always follow up with the tech team.",
                "answer_html": "I'm here to help you quickly find accurate, up-to-date information about our temperature-controlled shippers. Just ask me a question, and I'll pull from our internal knowledge to get you the answer. I currently support technical queries only — for logistics or sales, please reach out via Zendesk. If I'm unsure or can't find the right info, I'll let you know — and you can always follow up with the tech team.",
                "sources": [],
                "images_used": []
            }
            display_assistant_message(initial_message, datetime.now().replace(hour=16, minute=19))
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_user_message(message["content"], message.get("timestamp"))
            else:
                # For assistant messages, check if it's a full response object
                if isinstance(message["content"], dict):
                    display_assistant_message(message["content"], message.get("timestamp"))
                else:
                    # Fallback for simple text messages
                    simple_response = {
                        "answer": message["content"],
                        "answer_html": message["content"],
                        "sources": [],
                        "images_used": []
                    }
                    display_assistant_message(simple_response, message.get("timestamp"))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Ask about temperature-controlled shipping, payload specifications, duration requirements...",
                    key="user_input",
                    placeholder="Ask about temperature-controlled shipping, payload specifications, duration requirements..."
                )
            
            with col_send:
                send_button = st.form_submit_button("✈️", use_container_width=True)
        
        # Tips box
        st.markdown("""
        <div class="tips-box">
            <strong>Tips for best results:</strong> Be as specific as possible (include payload, duration, or temperature if known)<br>
            <strong>Eg:</strong> What Tinst can I use that ship US pallet at -40 to -60 temperature range?<br>
            Is there any product that can accommodate the following dimensions ...?
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Sidebar
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-title">
                ❓ What I Can Help With
            </div>
            <div class="help-item">
                <div class="help-icon">🔧</div>
                Technical specifications for temperature-controlled shippers
            </div>
            <div class="help-item">
                <div class="help-icon">📦</div>
                Payload and duration requirements
            </div>
            <div class="help-item">
                <div class="help-icon">🌡️</div>
                Temperature range guidance
            </div>
            <div class="help-item">
                <div class="help-icon">⚙️</div>
                Redirect to simulations
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-title">
                ✅ Need More Help?
            </div>
            <p>For logistics or sales inquiries, please contact our support team.</p>
            <button class="zendesk-button" onclick="window.open('https://towerassist.zendesk.com', '_blank')">
                Z Open Zendesk Support
            </button>
            <p style="font-size: 0.85rem; color: #6c757d; margin-top: 1rem;">
                Technical questions beyond my knowledge can be escalated to our engineering team.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle user input
    if send_button and user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Show loading spinner
        with st.spinner("Tower Assist is thinking..."):
            # Call backend API
            response_data = call_backend_api(user_input)
            
            if response_data:
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data,
                    "timestamp": datetime.now()
                })
            else:
                # Add error message with proper JSON structure
                error_response = {
                    "answer": "Sorry, I'm having trouble connecting to the knowledge base. Please try again later.",
                    "answer_html": "Sorry, I'm having trouble connecting to the knowledge base. Please try again later.",
                    "sources": [],
                    "images_used": []
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_response,
                    "timestamp": datetime.now()
                })
        
        # Rerun to show new messages
        st.rerun()


def main():
    """Main app function"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        login_page()
    else:
        chat_page()
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()