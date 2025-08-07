class QueryRouter:
    """Routes queries to appropriate content types (FAQ vs KB)"""
    
    def __init__(self):
        # Keywords for different content types
        self.faq_keywords = [
            "how", "what", "why", "when", "who", "explain", 
            "meaning", "definition", "tell me about", "describe",
            "is", "are", "does", "do", "can", "could", "would"
        ]
        self.kb_keywords = [
            "process", "procedure", "policy", "guideline", 
            "requirement", "step", "according to", "based on",
            "according to the", "per the", "as per", "following",
            "instructions", "guidelines", "steps", "gift"
        ]
    
    def route_query(self, query):
        """Determine if query is better suited for FAQ or KB"""
        query_lower = query.lower()
        
        # Count relevant keywords
        faq_score = sum(1 for word in self.faq_keywords if word in query_lower)
        kb_score = sum(1 for word in self.kb_keywords if word in query_lower)
        
        # Special cases for direct questions
        if any(qw in query_lower for qw in ["what is", "who is", "tell me about", "explain the", "is a", "are a"]):
            faq_score += 2
        
        # Special cases for procedural questions
        if any(qw in query_lower for qw in ["gift", "steps to", "procedure for", "process of"]):
            kb_score += 2
        
        return "faq" if faq_score > kb_score else "kb"