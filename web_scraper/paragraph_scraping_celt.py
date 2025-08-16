# Cath Maige Tuired: The Second Battle of Mag Tuired
# import requests
# from bs4 import BeautifulSoup
# import csv, re

# with open('paragraph_benchmark_dataset.csv','a', newline='',encoding='utf-8-sig') as csvfile:
#     fieldnames = ['Irish para', 'English translation']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     for i in range(2,169):
#         links=['https://celt.ucc.ie/published/G300010/text'+f"{i:03}"+'.html','https://celt.ucc.ie/published/T300010/text'+f"{i:03}"+'.html']
   
#         print(f"Processing {i-1} section")

#         irish_response = requests.get(links[0])
#         irish_content = BeautifulSoup(irish_response.content, 'html.parser')
#         irish_paras = irish_content.find_all('p')
#         if i==2:
#             irish_paras = irish_paras[1:]
#         irish_text = irish_paras[0].get_text()
#         irish_text = re.sub(r'\d+\]', '', irish_text.replace('\n', ''))
#         english_response = requests.get(links[1])
#         english_content = BeautifulSoup(english_response.content, 'html.parser')
#         english_paras = english_content.find_all('p')
#         if i==2:
#             english_paras = english_paras[1:]
#         english_text = english_paras[0].get_text().replace('\n', '')
#         no_of_sentences = english_text.split(".")
#         if len(no_of_sentences)>3:
#             writer.writerow({'Irish para':irish_text, 'English translation': english_text})
        

##The Wooing of Emer by Cú Chulainn

# import requests
# from bs4 import BeautifulSoup
# import csv, re

# with open('paragraph_dataset_wooeing.csv','w', newline='', encoding='utf-8-sig') as csvfile:
#     fieldnames = ['Irish para', 'English translation']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for i in range(2,93):
#         links=['https://celt.ucc.ie/published/G301021/text'+f"{i:03}"+'.html','https://celt.ucc.ie/published/T301021/text'+f"{i:03}"+'.html']
   
#         print(f"Processing {i-1} section")

#         irish_response = requests.get(links[0])
#         irish_content = BeautifulSoup(irish_response.content, 'html.parser')
#         irish_paras = irish_content.find_all('p')
#         irish_texts = []
#         for para in irish_paras:
       
#             irish_text = re.sub(r'\bp\.\d+\b', '', para.get_text().strip())
#             if irish_text:
#                 irish_texts.append(irish_text.replace('\n', ' '))

#         english_response = requests.get(links[1])
#         english_content = BeautifulSoup(english_response.content, 'html.parser')
#         english_paras = english_content.find_all('p')
#         english_texts=[]
#         for para in english_paras:
#             english_text = re.sub(r'\bp\.\d+\b', '', para.get_text().strip())
#             if english_text:
#                 english_texts.append(english_text.replace('\n', ' '))
        
#         for it,et in zip(irish_texts, english_texts):
            
#             writer.writerow({'Irish para':it, 'English translation': et})
    
# Buile Suibhne
import requests
from bs4 import BeautifulSoup
import csv, re

with open('paragraph_benchmark_dataset_buile.csv','w', newline='',encoding='utf-8-sig') as csvfile:
    fieldnames = ['Irish para', 'English translation']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for i in range(2,89):
        links=['https://celt.ucc.ie/published/G302018/text'+f"{i:03}"+'.html','https://celt.ucc.ie/published/T302018/text'+f"{(i-1):03}"+'.html']
   
        print(f"Processing {i-1} section")

        irish_response = requests.get(links[0])
        irish_content = BeautifulSoup(irish_response.content, 'html.parser')
        if not (irish_content.find_all('ol') or irish_content.find_all('li')):
            
            irish_paras = irish_content.find_all('p')
            irish_texts = []
            for para in irish_paras:
                irish_text = re.sub(r'\bp\.\d+\b|\d+\]', '', para.get_text().replace('\n',''))
                if irish_text:
                    irish_texts.append(irish_text)

            english_response = requests.get(links[1])
            english_content = BeautifulSoup(english_response.content, 'html.parser')
            english_paras = english_content.find_all('p')
            english_texts = []
            for para in english_paras:
                english_text = re.sub(r'\bp\.\d+\b', '', para.get_text())
                if english_text:
                    english_texts.append(english_text.replace('\n',''))
            
            for it,et in zip(irish_texts, english_texts):
                writer.writerow({'Irish para':it, 'English translation': et})
