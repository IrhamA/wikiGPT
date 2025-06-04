import wikipedia
from openai import OpenAI

# goal: take sentence input, if we get results NICE
#       otherwise we try to correct it 
#       - we can use wikipedia.suggest() but there is some weird behavior
#       - we can also use GPT



#inp = "how does a blakc hole form?"
#inp = "how does a black hole form?"

# ""

inp = "What happened in Mobutu's coup?"


search_results = wikipedia.search(inp)
if not search_results:
    query = wikipedia.suggest(inp)
    print(query)
    search_results = wikipedia.search(query)

print(search_results)

#print(wikipedia.summary(search_results[0], sentences = 5))

#summary = wikipedia.summary(titles[0], sentences = 3)

#print(summary)