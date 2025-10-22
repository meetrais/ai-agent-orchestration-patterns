import sys
import warnings
import uuid
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage
from langsmith import traceable
from dotenv import load_dotenv
import os
from prompts import SHOPPING_AGENT_PROMPT, PRODUCT_CATALOG_AGENT_PROMPT, PAYMENT_AGENT_PROMPT

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.memory import ConversationBufferMemory

load_dotenv()

class MultiAgentSequentialOrchestrator:
    """Handles sequential orchestration of shopping agents"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.4
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    @traceable(run_type="llm", name="Shopping Agent")
    def call_shopping_agent(self, user_input, chat_history, config=None):
        """Call shopping agent with user input"""
        messages = [
            SystemMessage(content=f"{SHOPPING_AGENT_PROMPT}\n\nConversation History: {chat_history}"),
            AIMessage(content=f"User request: {user_input}")
        ]
        response = self.llm.invoke(messages, config=config)
        return response.content if hasattr(response, 'content') else str(response)
    
    @traceable(run_type="llm", name="Product Catalog Agent")
    def call_catalog_agent(self, shopping_output, config=None):
        """Call catalog agent with shopping agent output"""
        messages = [
            SystemMessage(content=PRODUCT_CATALOG_AGENT_PROMPT),
            AIMessage(content=f"Shopping Agent output: {shopping_output}")
        ]
        response = self.llm.invoke(messages, config=config)
        return response.content if hasattr(response, 'content') else str(response)
    
    @traceable(run_type="llm", name="Payment Agent")
    def call_payment_agent(self, catalog_output, config=None):
        """Call payment agent with catalog agent output"""
        messages = [
            SystemMessage(content=PAYMENT_AGENT_PROMPT),
            AIMessage(content=f"Catalog Agent output: {catalog_output}")
        ]
        response = self.llm.invoke(messages, config=config)
        return response.content if hasattr(response, 'content') else str(response)
    
    def orchestrate(self, user_input):
        """Main orchestration logic"""
        # Generate unique transaction ID
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create config with transaction metadata
        run_config = {
            "tags": [f"transaction:{transaction_id}"],
            "metadata": {
                "transaction_id": transaction_id,
                "timestamp": timestamp,
                "user_input": user_input[:100]  # First 100 chars
            }
        }
        
        print(f"[Transaction ID: {transaction_id}]")
        print(f"[Timestamp: {timestamp}]\n")
        
        # Step 1: Shopping agent
        print("[1/3] Shopping Agent...")
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        
        shopping_result = self.call_shopping_agent(user_input, chat_history, config=run_config)
        print(f"      {shopping_result}\n")
        
        # Step 2: Check if ready to purchase
        if "READY_TO_PURCHASE:" in shopping_result:
            print("[2/3] Product Catalog Agent...")
            
            catalog_result = self.call_catalog_agent(shopping_result, config=run_config)
            print("      Complete")
            
            print("[3/3] Payment Agent...")
            
            payment_result = self.call_payment_agent(catalog_result, config=run_config)
            print("      Complete\n")
            
            print("=" * 60)
            print("ORDER SUMMARY")
            print("=" * 60)
            print(payment_result)
            print("=" * 60)
            final_result = payment_result
        else:
            final_result = shopping_result
        
        # Save to memory
        self.memory.save_context({"input": user_input}, {"output": final_result})
        return final_result
    
    def run(self):
        """Main conversation loop"""
        print("Shopping Agent: Hello! How can I help you today?\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for shopping with us!")
                    break
                if user_input:
                    self.orchestrate(user_input)
                    print()
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    """Entry point"""
    orchestrator = MultiAgentSequentialOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
