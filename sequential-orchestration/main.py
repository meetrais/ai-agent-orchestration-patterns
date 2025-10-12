import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from prompts import (
    SHOPPING_AGENT_PROMPT,
    PRODUCT_CATALOG_AGENT_PROMPT,
    PAYMENT_AGENT_PROMPT
)

load_dotenv()

class MultiAgentSequentialOrchestrator:
    """Handles sequential orchestration of shopping agents"""
    
    def __init__(self):
        self.llm = self._create_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.parser = StrOutputParser()
        
        # Create agent chains
        self.shopping_chain = self._create_agent_chain(SHOPPING_AGENT_PROMPT, use_memory=True)
        self.catalog_chain = self._create_agent_chain(PRODUCT_CATALOG_AGENT_PROMPT)
        self.payment_chain = self._create_agent_chain(PAYMENT_AGENT_PROMPT)
    
    def _create_llm(self):
        """Create LLM instance"""
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.4
        )
    
    def _create_agent_chain(self, system_prompt, use_memory=False):
        """Create a reusable agent chain"""
        if use_memory:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\nConversation History: {chat_history}"),
                ("human", "{input}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
        
        return prompt | self.llm | self.parser
    
    def _get_chat_history(self):
        """Get formatted chat history"""
        return self.memory.load_memory_variables({}).get("chat_history", [])
    
    def process_shopping(self, user_input):
        """Process user input through shopping agent"""
        print("[1/3] Shopping Agent...")
        
        result = self.shopping_chain.invoke({
            "input": user_input,
            "chat_history": self._get_chat_history()
        })
        
        print(f"      {result}\n")
        return result
    
    def process_catalog(self, shopping_output):
        """Process through catalog agent"""
        print("[2/3] Product Catalog Agent...")
        result = self.catalog_chain.invoke({"input": shopping_output})
        print("      Complete")
        return result
    
    def process_payment(self, catalog_output):
        """Process through payment agent"""
        print("[3/3] Payment Agent...")
        result = self.payment_chain.invoke({"input": catalog_output})
        print("      Complete\n")
        self._display_order_summary(result)
        return result
    
    def _display_order_summary(self, result):
        """Display formatted order summary"""
        print("=" * 60)
        print("ORDER SUMMARY")
        print("=" * 60)
        print(result)
        print("=" * 60)
    
    def orchestrate(self, user_input):
        """Main orchestration logic"""
        # Step 1: Shopping agent
        shopping_result = self.process_shopping(user_input)
        
        # Step 2: Check if ready to purchase
        if "READY_TO_PURCHASE:" in shopping_result:
            # Full pipeline: catalog -> payment
            catalog_result = self.process_catalog(shopping_result)
            payment_result = self.process_payment(catalog_result)
            final_result = payment_result
        else:
            # Just conversation
            final_result = shopping_result
        
        # Save to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": final_result}
        )
        
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
                
                if not user_input:
                    continue
                
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