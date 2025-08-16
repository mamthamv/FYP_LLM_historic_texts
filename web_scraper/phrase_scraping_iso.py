import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

urls= ["https://iso.ucc.ie/Acallamh-senorach/Acallamh-senorach-text.html","https://iso.ucc.ie/Baile-buan/Baile-buan-text.html","https://iso.ucc.ie/Tain-cualnge/Tain-cualnge-text.html","https://iso.ucc.ie/Scel-datho/Scel-datho-text.html","https://iso.ucc.ie/Cath-maige/Cath-maige-text.html"]

medieval_irish = []
modern_irish = []
english = []

for url in urls:
    response = requests.get(url)
    content = BeautifulSoup(response.content, "html.parser")

    for table in content.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) == 3:
                old = cols[0].get_text(strip=True)
                modern = cols[1].get_text(strip=True)
                eng = cols[2].get_text(strip=True)
                if old and modern and eng:
                    medieval_irish.append(old)
                    modern_irish.append(modern)
                    english.append(eng)

df = pd.DataFrame({
    "Old_Irish": medieval_irish,
    "Modern_Irish": modern_irish,
    "English": english
})


df.to_excel("sentence_dataset.xlsx", index=False)