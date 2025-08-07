from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

class RAGChain:
    """RAG chain for generating answers from retrieved context"""
    
    def __init__(self, settings, project_name):
        self.settings = settings
        self.project_name = project_name
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        return ChatOpenAI(
            model_name=self.settings.LLM_MODEL,
            temperature=self.settings.LLM_TEMPERATURE,
            openai_api_key=self.settings.OPENAI_API_KEY
        )
    
    def _create_prompt_template(self):
        """Create the prompt template for the RAG chain"""
        template = f"""You are a helpful assistant for {self.project_name}.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{{context}}

Question: {{question}}
Helpful Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def generate_answer(self, query, context_docs):
        """Generate an answer from the retrieved context"""
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create and run the chain
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        
        response = llm_chain.invoke({
            "context": context,
            "question": query
        })
        
        return response["text"]
    
    def format_response_with_sources(self, answer, context_docs, project_id, project_name):
        """Format the response with source citations and project identification"""
        # Add project identification
        project_identification = f"Answer for project: {project_name} (ID: {project_id})\n\n"
        
        # Add source citations
        sources = []
        for doc in context_docs[:2]:  # Limit to top 2 sources
            if doc.metadata["source"] == "faq":
                sources.append(f"FAQ: {doc.metadata['question']}")
            else:
                sources.append(f"KB: {doc.metadata['article_title']} (Section {doc.metadata['chunk_id']+1}/{doc.metadata['total_chunks']})")
        
        formatted_answer = project_identification + answer
        
        if sources:
            formatted_answer += "\n\nSources:\n- " + "\n- ".join(sources)
        
        return formatted_answer