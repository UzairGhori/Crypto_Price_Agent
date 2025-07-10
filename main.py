import streamlit as st
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool
from dotenv import load_dotenv
import os
import requests
import asyncio
from typing import Dict, Any
import nest_asyncio

# Apply nest_asyncio to allow async in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Constants
BINANCE_API_URL = "https://api.binance.com/api/v3"
DEFAULT_CRYPTO = "BTCUSDT"

class CryptoDataService:
    """Service for fetching cryptocurrency data from Binance API"""
    
    @staticmethod
    @function_tool
    def get_crypto_price(symbol: str) -> str:
        """Get current price of cryptocurrency pair (e.g. BTCUSDT)"""
        try:
            response = requests.get(f"{BINANCE_API_URL}/ticker/price?symbol={symbol}")
            response.raise_for_status()
            data = response.json()
            return f"Current {symbol} price: ${float(data['price']):,.2f}"
        except Exception as e:
            return f"❌ Error fetching {symbol} price: {str(e)}"

    @staticmethod
    @function_tool
    def get_crypto_market_data(symbol: str) -> Dict[str, Any]:
        """Get detailed market data for cryptocurrency pair"""
        try:
            response = requests.get(f"{BINANCE_API_URL}/ticker?symbol={symbol}")
            response.raise_for_status()
            data = response.json()
            return {
                "symbol": symbol,
                "price": float(data['price']),
                "change": float(data['priceChangePercent']),
                "high": float(data['highPrice']),
                "low": float(data['lowPrice']),
                "volume": float(data['volume'])
            }
        except Exception as e:
            return {"error": str(e)}

def initialize_agent() -> Agent:
    """Initialize and configure the crypto agent"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",  # Fixed typo from gemini to gemini
        openai_client=client
    )

    return Agent(
        name="CryptoBot",
        instructions="""You are CryptoBot, a professional cryptocurrency assistant. 
        Provide accurate, real-time crypto data including prices, market changes, 
        and trading volume. Always:
        1. Verify data from Binance API
        2. Format numbers properly (2 decimal places, commas)
        3. Include percentage changes where relevant
        4. Keep responses concise but informative
        5. Mention if data is delayed""",
        tools=[
            CryptoDataService.get_crypto_price,
            CryptoDataService.get_crypto_market_data
        ]
    )

def format_market_data(data: Dict[str, Any]) -> str:
    """Format market data into readable string"""
    if "error" in data:
        return data["error"]
    
    return (
        f"📊 {data['symbol']} Market Data:\n"
        f"• Price: ${data['price']:,.2f}\n"
        f"• 24h Change: {data['change']:.2f}%\n"
        f"• 24h High: ${data['high']:,.2f}\n"
        f"• 24h Low: ${data['low']:,.2f}\n"
        f"• 24h Volume: {data['volume']:,.2f}"
    )

async def get_agent_response(agent: Agent, prompt: str) -> str:
    """Get response from agent asynchronously"""
    try:
        runner = await Runner.run(
            starting_agent=agent,
            input=prompt,
            run_config=RunConfig(
                model=OpenAIChatCompletionsModel(
                    model="gemini-2.0-flash",
                    openai_client=AsyncOpenAI(
                        api_key=os.getenv("GEMINI_API_KEY"),
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                    )
                ),
                tracing_disabled=True
            )
        )
        return runner.final_output
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Crypto Agent",
        page_icon="🤖",
        layout="centered"
    )
    
    st.title("💰Crypto Price Agent")
    st.markdown("Your AI cryptocurrency assistant")
    
    # Initialize agent and session state
    if "agent" not in st.session_state:
        st.session_state.agent = initialize_agent()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hi! I'm Crypto Price Agent. Ask me about cryptocurrency prices and market data."
        }]
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about crypto prices..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.spinner("Analyzing..."):
            try:
                # Run async function in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    get_agent_response(st.session_state.agent, prompt)
                )
                
                # Check if response contains market data to format
                if any(keyword in prompt.lower() for keyword in ["market", "data", "details"]):
                    try:
                        symbol = next(
                            (word for word in prompt.upper().split() 
                             if word in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]),
                            DEFAULT_CRYPTO
                        )
                        market_data = CryptoDataService.get_crypto_market_data(symbol)
                        if "error" not in market_data:
                            response = format_market_data(market_data)
                    except:
                        pass
                
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()