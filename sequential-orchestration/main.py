#!/usr/bin/env python3
"""
Online Shopping Multi-Agent Sequential Orchestration
Using LangChain's native capabilities - fully agentic
"""

import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from prompts import (
    SHOPPING_AGENT_PROMPT,
    PRODUCT_CATALOG_AGENT_PROMPT,
    PAYMENT_AGENT_PROMPT
)

load_dotenv()


def create_llm():
    """Create LLM instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4
    )


def main():
    """Main application - fully agentic orchestration"""
    print("\nOnline Shopping Multi-Agent System")
    print("Powered by LangChain + Google Gemini\n")
    
    llm = create_llm()
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output"
    )
    
    # Shopping agent with memory
    shopping_prompt = ChatPromptTemplate.from_messages([
        ("system", SHOPPING_AGENT_PROMPT + "\n\nConversation History: {chat_history}"),
        ("human", "{input}")
    ])
    
    def shopping_with_memory(input_text):
        history = memory.load_memory_variables({}).get("chat_history", [])
        result = (shopping_prompt | llm | StrOutputParser()).invoke({
            "input": input_text,
            "chat_history": history
        })
        return result
    
    shopping_chain = RunnableLambda(shopping_with_memory)
    
    # Router agent - decides next step
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a router. Based on the shopping agent's response, output ONLY ONE WORD:
- Output 'CATALOG' if customer is ready to purchase (has specific product/buying intent)
- Output 'CHAT' if more conversation is needed"""),
        ("human", "{input}")
    ])
    router_chain = router_prompt | llm | StrOutputParser()
    
    # Catalog agent
    catalog_prompt = ChatPromptTemplate.from_messages([
        ("system", PRODUCT_CATALOG_AGENT_PROMPT),
        ("human", "{input}")
    ])
    catalog_chain = catalog_prompt | llm | StrOutputParser()
    
    # Payment agent
    payment_prompt = ChatPromptTemplate.from_messages([
        ("system", PAYMENT_AGENT_PROMPT),
        ("human", "{input}")
    ])
    payment_chain = payment_prompt | llm | StrOutputParser()
    
    # Build agentic sequential chain
    def route_decision(shopping_output):
        """Router decides if catalog agent should run"""
        decision = router_chain.invoke({"input": shopping_output})
        return "catalog" if "CATALOG" in decision.upper() else "chat"
    
    def run_shopping(user_input):
        print("[1/3] Shopping Agent...")
        result = shopping_chain.invoke({"input": user_input})
        print(f"      {result}\n")
        return result
    
    def run_catalog(shopping_output):
        print("[2/3] Product Catalog Agent...")
        result = catalog_chain.invoke({"input": shopping_output})
        print("      Complete")
        return result
    
    def run_payment(catalog_output):
        print("[3/3] Payment Agent...")
        result = payment_chain.invoke({"input": catalog_output})
        print("      Complete\n")
        print("="*60)
        print("ORDER SUMMARY")
        print("="*60)
        print(result)
        print("="*60)
        return result
    
    # Create branching logic using RunnableBranch
    orchestration_chain = (
        RunnableLambda(run_shopping) |
        RunnableBranch(
            (lambda x: route_decision(x) == "catalog", 
             RunnableLambda(run_catalog) | RunnableLambda(run_payment)),
            RunnableLambda(lambda x: x)  # Just return shopping response if chat
        )
    )
    
    # Main loop with LangChain memory
    print("Shopping Agent: Hello! How can I help you today?\n")
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for shopping with us!")
            break
        
        if not user_input:
            continue
        
        try:
            # Save user input to memory
            memory.save_context({"input": user_input}, {"output": "processing..."})
            
            # Execute orchestration
            result = orchestration_chain.invoke(user_input)
            
            # Update memory with actual response
            history = memory.load_memory_variables({}).get("chat_history", [])
            if history:
                history[-1].content = str(result) if isinstance(result, str) else "Order completed"
            
            print()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
