import json, re, os, time
from dotenv import load_dotenv
from pathlib import Path
import lmstudio as lms
import pandas as pd

#load environment variables
load_dotenv()

#load required files
config = json.load(open(Path(os.getenv("CONFIG_FILE"))))
models = config["models"]
benches = config["benchmarks"]

benchPath = Path(os.getenv("BENCHMARK_PATH"))
evalOutputPath = Path(os.getenv("EVAL_OUTPUT_PATH"))
evalOutputPath.mkdir(exist_ok=True)

#set response pattern to process responses
rspnsPtrn = re.compile(r"<response>(?P<answer>.*)<\/response>") 

#parameter configuration
temps = [0.1, 0.2, 0.5, 1.0]
topks = [30, 40, 30, 40]
topps = [0.7, 0.85, 0.85, 0.9]

def evalLocalModels():
    for modelP in models:
        if modelP["skip"] or modelP["family"] != "local":
            continue

        print(f"=== Loading local model '{modelP['id']}'")
        print(modelP["id"])
        model = lms.llm(modelP["id"])

        #iterating through benhcmarks
        for bName in benches:
            dataset = bName['id']
            prompt = bName['prompt']

            df = pd.read_csv(benchPath / dataset, encoding="utf-8") 

            data_list = df['Irish'].tolist()

            print(f"=== Loaded benchmark '{bName['id']}' with prompts")

            results_model = pd.DataFrame()
            for iterator in range(4):
                results = []
                config_name = f"{modelP['id']}_T{temps[iterator]}_K{topks[iterator]}_P{topps[iterator]}"
                print(f"=== [CONFIGURATION] : {config_name}")
            
                for item in data_list:

                    #input data cleanup
                    item = item.replace("\"", "'") 
                    item = item.replace("\n", " ") 


                    chat = lms.Chat(f"/no_think")
                    chat.add_user_message(f"{prompt}{item}")
                    try:
                        response = model.respond(chat, config={
                            "temperature": temps[iterator],
                            "topKSampling": topks[iterator],
                            "topP": topps[iterator],
                            "maxTokens": 1400,
                            "stopStrings": ["</response>"] 
                        })
                        resStr = f"{response.content}</response>"

                        #response cleanup
                        resStr = resStr.replace("\n", " ") 
                        resStr = resStr.replace("\"", "'")

                
                        print(f"=== === [QUERY ITEM] : {item}")
                        print(f"=== === [RESPONSE] : {resStr}")

                        resGroups = re.search(rspnsPtrn, resStr)
                        if resGroups != None:
                            res = resGroups.groupdict()["answer"]
                            res = res.strip()
                            results.append(res)
                        else:
                            raise Exception("Can't find a matching response XML pattern.")
                    except Exception as e:
                        #just directly append response, if error in parsing
                        print(f"[ERROR] : {e}") 
                        results.append(response) 
                  
                results_model[config_name]=results

            #save to csv
            model_name = modelP['id'].replace("/", "_")
            results_model.to_csv(evalOutputPath / f'{model_name}_{bName["name"]}.csv', index=False, encoding="utf-8")
            print(f"Saved results to: {model_name}_{bName['name']}.csv")

        model.unload()

if __name__ == "__main__":
    evalLocalModels()
