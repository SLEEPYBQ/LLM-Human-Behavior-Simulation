import os
from langchain_openai import ChatOpenAI

def test_api_connection():
    """Test the proxy API connection"""
    
    try:
        llm = ChatOpenAI(
            base_url='https://api.openai-proxy.org/v1',
            api_key='sk-ctf5Qjjf5cAG6fKuRn2rMEXlRd0ogCNT0ex6kn76oEAYL1eq',
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        response = llm.invoke("Hello, this is a test message. Please respond with 'API connection successful'.")
        print(f"✅ API Connection Test Successful!")
        print(f"Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ API Connection Test Failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_api_connection()