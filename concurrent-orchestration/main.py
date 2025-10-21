import sys
import warnings
import asyncio
import uuid
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from prompts import (
    SHOPPING_AGENT_PROMPT,
    PRODUCT_CATALOG_AGENT_PROMPT,
    CUSTOMER_SERVICE_AGENT_PROMPT,
    PAYMENT_AGENT_PROMPT
)

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.memory import ConversationBufferMemory

load_dotenv()

class MultiAgentConcurrentOrchestrator:
    """Handles concurrent orchestration of shopping agents"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.4
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        parser = StrOutputParser()
        
        # Create agent chains with custom names for LangSmith tracing
        self.shopping_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SHOPPING_AGENT_PROMPT + "\n\nConversation History: {chat_history}"),
                ("human", "{input}")
            ]) | self.llm | parser
        ).with_config({"run_name": "Shopping Agent"})
        
        self.catalog_chain = (
            ChatPromptTemplate.from_messages([
                ("system", PRODUCT_CATALOG_AGENT_PROMPT),
                ("human", "{input}")
            ]) | self.llm | parser
        ).with_config({"run_name": "Product Catalog Agent"})
        
        self.customer_service_chain = (
            ChatPromptTemplate.from_messages([
                ("system", CUSTOMER_SERVICE_AGENT_PROMPT),
                ("human", "{input}")
            ]) | self.llm | parser
        ).with_config({"run_name": "Customer Service Agent"})
        
        self.payment_chain = (
            ChatPromptTemplate.from_messages([
                ("system", PAYMENT_AGENT_PROMPT),
                ("human", "{input}")
            ]) | self.llm | parser
        ).with_config({"run_name": "Payment Agent"})
    
    async def orchestrate_async(self, user_input):
        """Async orchestration logic with concurrent agent execution"""
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
        print("[1/4] Shopping Agent...")
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        shopping_result = self.shopping_chain.invoke(
            {"input": user_input, "chat_history": chat_history},
            config=run_config
        )
        print(f"      {shopping_result}\n")
        
        # Step 2 & 3: Check if ready to purchase
        if "READY_TO_PURCHASE:" in shopping_result:
            print("[2/4] Catalog Agent (running concurrently)...")
            print("[3/4] Customer Service Agent (running concurrently)...")
            
            # Run catalog and customer service agents concurrently
            catalog_task = asyncio.to_thread(
                self.catalog_chain.invoke, {"input": shopping_result}, run_config
            )
            service_task = asyncio.to_thread(
                self.customer_service_chain.invoke, {"input": shopping_result}, run_config
            )
            
            catalog_result, service_result = await asyncio.gather(catalog_task, service_task)
            print("      Complete\n")
            
            # Step 4: Payment agent with combined results
            print("[4/4] Payment Agent...")
            combined_input = f"""
Catalog Information:
{catalog_result}

Customer Service Information:
{service_result}

Original Request: {shopping_result}
"""
            payment_result = self.payment_chain.invoke(
                {"input": combined_input},
                config=run_config
            )
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
    
    def orchestrate(self, user_input):
        """Synchronous wrapper for async orchestration"""
        return asyncio.run(self.orchestrate_async(user_input))
    
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
    orchestrator = MultiAgentConcurrentOrchestrator()
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
