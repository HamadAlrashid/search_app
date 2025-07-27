# from https://medium.com/@kelvincampelo/how-ive-optimized-document-interactions-with-open-webui-and-rag-a-comprehensive-guide-65d1221729eb
RAG_PROMPT = """
**Generate Response to User Query**
**Step 1: Parse Context Information**
Extract and utilize relevant knowledge from the provided context within `<context></context>` XML tags.
**Step 2: Analyze User Query**
Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the question.
**Step 3: Determine Response**
If the answer to the user's query can be directly inferred from the context information, provide a concise and accurate response in the same language as the user's query.
**Step 4: Handle Uncertainty**
If the answer is not clear, ask the user for clarification to ensure an accurate response.
**Step 5: Avoid Context Attribution**
When formulating your response, do not indicate that the information was derived from the context.
**Step 6: Respond in User's Language**
Maintain consistency by ensuring the response is in the same language as the user's query.
**Step 7: Provide Response**
Generate a clear, concise, and informative response to the user's query, adhering to the guidelines outlined above.
User Query: {query}
<context>
{context}
</context>
"""

# prompt for generating number_of_queries different queries 
MULTI_QUERY_PROMPT = """You are an expert query generation AI assistant for a Retrieval-Augmented Generation (RAG) system. 

Your task is to reformulate the user's question into exactly {number_of_queries} diverse search queries that will help retrieve comprehensive and relevant documents.

Generate queries that:
- Use different keywords and phrasings than the original
- Cover different aspects or angles of the question
- Are semantically similar but lexically diverse
- Would retrieve complementary information

**Original Question:** {query}

Generate exactly {number_of_queries} alternative queries. Do not include the original question."""


MULTI_QUERY_PROMPT2 = """
You are an expert query generation AI assistant for a Retrieval-Augmented Generation (RAG) system. Your sole purpose is to reformulate a single user's question into a set of 3 synonymous search queries. These queries will be used to retrieve relevant documents.

Generate queries that are semantically similar to the original question but use different keywords and phrasing.

**User's Original Question:**
"{query}"

Based on the user's question above, generate a JSON array of 3 alternative queries that are rephrased or use synonyms.

**CRITICAL:**
- Your output **MUST** be a single, valid JSON array of strings.
- Do **NOT** include the original question in the output.
- Do **NOT** add any explanations, introductory text, or markdown formatting around the JSON output.

**Example:**

**User's Original Question:**
"Why is the sky blue during the day but red during sunset?"

**Your Output:**
[
  "what causes the sky's color to be blue and then red at sunset",
  "explain the reason for the sky changing color from blue to red",
  "what is the scientific reason for the sky's blue daytime color and red sunset color"
]
"""

QUERY_DECOMPOSITION_PROMPT = """
You are an expert query decomposition AI assistant for a Retrieval-Augmented Generation (RAG) system. Your sole purpose is to decompose a single user's question into a set of sub-queries. These sub-queries will be used to retrieve relevant documents.

The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.

**User's Original Question:**
"{query}"

Generate no more than {number_of_queries} sub-queries. Do not include the original question in the output.
"""



task_map = {
  "multi_query": MULTI_QUERY_PROMPT,
  "query_decomposition": QUERY_DECOMPOSITION_PROMPT,
  "rag": RAG_PROMPT,
  
}