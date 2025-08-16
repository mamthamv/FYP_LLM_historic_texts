import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


response = requests.get('https://celt.ucc.ie/published/G303000.html')

content = BeautifulSoup(response.content, 'html.parser')

para_content= content.find_all('p')

words_set=set()
results=[]

for p in para_content:
    text= p.get_text(separator=' ', strip=True)
    words = re.findall(r'\b[^\W\d_]+\b', text)
    words_set.update(w.lower() for w in words if len(w) > 1)


# Convert to DataFrame
df = pd.DataFrame(list(words_set), columns=["Word"])

# Export to Excel
df.to_excel("words_exported.xlsx", index=False)

print("Exported to words_exported.xlsx")