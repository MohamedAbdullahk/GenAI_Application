from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load variables from the .env file into environment variables
load_dotenv()

# Define the template using role-based tuples
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert educator who explains complex topics using simple analogies."),
    ("human", "Explain the concept of {topic} to a {target_audience}.")
])

# Test formatting the prompt locally to see what it produces
formatted_prompt = prompt_template.invoke({
    "topic": "Photosynthesis",
    "target_audience": "5-year-old child"
})

print("Formatted Prompt Output:")
print(formatted_prompt)

# Initialize the model
# temperature=0.7 controls creativity (0 = rigid/deterministic, 1 = creative)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Test calling the model directly with our formatted prompt
response = llm.invoke(formatted_prompt)

print("\nModel Raw Response:")
print(response)

# 1. Create the Output Parser
output_parser = StrOutputParser()

# 2. Build the Chain using LCEL (Pipe Syntax)
# Data flow: Prompt Template -> LLM -> Output Parser
chain = prompt_template | llm | output_parser

# 3. Run the complete chain with a single call
final_result = chain.invoke({
    "topic": "Photosynthesis",
    "target_audience": "5-year-old child"
})

print("\n--- Final Clean Output ---")
print(final_result)

# Replace .invoke() with .stream()
for chunk in chain.stream({
    "topic": "Black Holes",
    "target_audience": "10-year-old child"
}):
    # print each chunk as it arrives
    # flush=True forces Python to display the text immediately in the terminal
    print(chunk, end="", flush=True)

print("\n\n--- Stream Complete ---")