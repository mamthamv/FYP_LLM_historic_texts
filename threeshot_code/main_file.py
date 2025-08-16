import json
import re
import os
import time
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import importlib

load_dotenv()

# config
config_path = Path(os.getenv("CONFIG_FILE"))
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

models = config["models"]
benches = config["benchmarks"]

benchPath = Path(os.getenv("BENCHMARK_PATH"))
evalOutputPath = Path(os.getenv("EVAL_OUTPUT_PATH"))
evalOutputPath.mkdir(exist_ok=True)

sleepTime = int(os.getenv("API_SLEEP", 2))

rspnsPtrn = re.compile(r"<response>(?P<answer>.*)<\/response>", re.IGNORECASE)


temps = [0.1, 0.2, 0.5, 1.0]
topks = [30, 40, 30, 40]
topps = [0.7, 0.85, 0.85, 0.9]

def get_model_api_key(env_key_name):
    return os.getenv(env_key_name)

def get_model_call_fn(fn_name):
    return getattr(importlib.import_module("model_invocation"), fn_name)

def evalAPIModels():
    max_retries = 3  # number of times to retry API calls

    for modelP in models:
        fn_name = modelP.get("call_fn")
        model_id = modelP.get("id", "")
        api_key_env = modelP.get("api_env_key") 
        model_arg = modelP.get("model")
        supports_top_k = bool(modelP.get("supports_top_k", False)) or ("gemini" in (fn_name or "").lower())

        print(f"=== Loading API model entry '{model_id}' (fn: {fn_name})")

        try:
            call_fn = get_model_call_fn(fn_name)
        except Exception as e:
            print(f"[ERROR] Skipping model entry {model_id}: {e}")
            continue

        api_key = get_model_api_key(api_key_env)
        for bName in benches:
            dataset = bName["id"]
            prompt = bName["prompt"]

            csv_path = benchPath / dataset
            df = pd.read_csv(csv_path, encoding="utf-8")
            data_list = df["Irish"].astype(str).tolist()
            print(f"=== Loaded benchmark '{dataset}' with {len(data_list)} items")

            original_cols = df.columns[:2].tolist()
            results_model = df[original_cols].copy()

            for iterator in range(len(temps)):
                t = temps[iterator]
                k = topks[iterator]
                p = topps[iterator]

                if supports_top_k:
                    config_name = f"{model_id}_T{t}_K{k}_P{p}"
                else:
                    config_name = f"{model_id}_T{t}_P{p}"

                print(f"=== [CONFIGURATION] : {config_name}")

                results = []
                for item in data_list:
                    item_clean = item.replace("\"", "'").replace("\n", " ").strip()
                    full_prompt = f"{prompt}{item_clean}"

                    call_kwargs = {}
                    call_args = [full_prompt, api_key]

                    if model_arg:
                        call_kwargs["model"] = model_arg
                    call_kwargs["temperature"] = t
                    call_kwargs["top_p"] = p
                    if supports_top_k:
                        call_kwargs["top_k"] = k

                    # retry loop
                    for attempt in range(1, max_retries + 1):
                        try:
                            response_text = call_fn(*call_args, **call_kwargs)

                            if response_text is None:
                                raise ValueError("API returned None")

                            if not isinstance(response_text, str):
                                if hasattr(response_text, "text"):
                                    response_text = str(response_text.text)
                                elif hasattr(response_text, "content"):
                                    response_text = str(response_text.content)
                                else:
                                    response_text = str(response_text)

                            resStr = response_text.replace("\n", " ").replace("\"", "'").strip()

                            m = rspnsPtrn.search(resStr)
                            if m:
                                res = m.groupdict()["answer"].strip()
                                results.append(res)
                            else:
                                print("[WARNING] Response didn't include <response> tags; using entire response as translation.")
                                results.append(resStr)
                            break  # success, stop retrying

                        except Exception as e:
                            if "NoneType" in str(e) or "API returned None" in str(e):
                                print(f"[RETRY] Attempt {attempt} failed for item due to None response. Retrying...")
                                time.sleep(1)  # short wait before retry
                                if attempt == max_retries:
                                    results.append(f"[ERROR after retries] {e}")
                            else:
                                print(f"[ERROR] Exception calling model fn for item: {e}")
                                results.append(f"[ERROR] {e}")
                                break  # don't retry other errors

                    if sleepTime and sleepTime > 0:
                        time.sleep(sleepTime)

                final_model_name = model_id.replace("/", "_").replace(" ", "_")
                out_fname = f"{final_model_name}_{bName['name']}.csv"
                results_model[config_name] = results
                results_model.to_csv(evalOutputPath / out_fname, index=False, encoding="utf-8")
                print(f"Saved results to: {out_fname}")

    print("=== All done ===")

if __name__ == "__main__":
    evalAPIModels()
