import json, re, os, time
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import importlib

load_dotenv()

# Load config and paths
config = json.load(open(Path(os.getenv("CONFIG_FILE"))))
models = config["models"]

benchPath = Path(os.getenv("BENCHMARK_PATH"))
evalOutputPath = Path(os.getenv("EVAL_OUTPUT_PATH"))
evalOutputPath.mkdir(exist_ok=True)

sleepTime = int(os.getenv("API_SLEEP", 2))
rspnsPtrn = re.compile(r"<response>(?P<answer>.*)<\/response>")

def get_model_api_key(env_key_name):
    return os.getenv(env_key_name)

def get_model_call_fn(fn_name):
    return getattr(importlib.import_module("model_invocation"), fn_name)

def translate_assigned_set_to_mi():
    prompt_mi = config["prompt_mi"]

    for model in models:
        model_id = model["id"]
        
        bench_file = model["bench_name"]

        print(f"\n=== {model_id} translating {bench_file} to MI")

        api_key = get_model_api_key(model["api_env_key"])
        llm_call = get_model_call_fn(model["call_fn"])
        print(benchPath / bench_file)
        df = pd.read_csv(benchPath / bench_file, encoding="utf-8", skip_blank_lines=True)
        english_list = df["English"].tolist()
        
        
        mi_results = []

        for sentence in english_list:
            try:
                full_prompt = f"{prompt_mi} {sentence}"
                response = llm_call(prompt=full_prompt, api_key=api_key)
                resStr =response.replace("\n", " ").replace("\"", "'")
                match = re.search(rspnsPtrn, resStr)
                translation = match.group("answer").strip() if match else "ParseError"
                mi_results.append(translation)
            except Exception as e:
                mi_results.append(str(e))
          
            time.sleep(sleepTime)

        out_file = evalOutputPath / f"{model_id}_mi.csv"
        pd.DataFrame({f"{model_id}_mi": mi_results, "English_sentence": df["English"]}).to_csv(out_file, index=False, encoding="utf-8")
        print(f"Saved MI output: {out_file.name}")

def evaluate_other_sets_in_en():
    prompt_en = config["prompt_en"]
    
    # Loop through MI files generated
    for model in models:
        source_model= model["id"]
        input_path = evalOutputPath / f"{source_model}_mi.csv"
    
        print(f"\n=== Evaluating output from: {source_model}")
        df_mi = pd.read_csv(input_path, encoding="utf-8")
        column_name = f"{source_model}_mi"
        mi_sentences = df_mi[column_name].tolist()

        # Use all other models to evaluate this MI set
        evaluators = [m for m in models if m["id"] != source_model]
        config_file_name_trans=f"{source_model}_evaluated_by_"
        backtranslations = {"Mi_sentences":df_mi[column_name],"Original_English_sentence":df_mi["English_sentence"]}
        for evaluator in evaluators:
            eval_id = evaluator["id"]
            config_file_name_trans+=eval_id

            eval_api_key = get_model_api_key(evaluator["api_env_key"])
            eval_fn = get_model_call_fn(evaluator["call_fn"])
            results_en=[]

            print(f"\n>>> {eval_id} evaluating {source_model}'s MI translations")

            for mi_sentence in mi_sentences:
                try:
                    prompt = f"{prompt_en} {mi_sentence}"
                    response = eval_fn(prompt=prompt, api_key=eval_api_key)
                    resStr = response.replace("\n", " ").replace("\"", "'")
                    match = re.search(rspnsPtrn, resStr)
                    translation = match.group("answer").strip() if match else "ParseError"
                    results_en.append(translation)
                except Exception as e:
                    results_en.append(str(e))
                time.sleep(sleepTime)
                
            backtranslations[eval_id+" translation"]=results_en

        config_file_name_trans+=".csv"
        out_file = evalOutputPath / config_file_name_trans
        pd.DataFrame(backtranslations).to_csv(out_file, index=False, encoding="utf-8")
        print(f"Saved: {out_file.name}")

def evalClosedModels():
    translate_assigned_set_to_mi()
    evaluate_other_sets_in_en()

if __name__ == "__main__":
    evalClosedModels()