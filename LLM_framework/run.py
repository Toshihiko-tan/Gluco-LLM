import sys, os, re, yaml, argparse, numpy as np, pandas as pd
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)
STRIDE = 6

cli = argparse.ArgumentParser()
cli.add_argument("--shots", type=int, default=2)
cli.add_argument("--rag",   action="store_true",
                 help="Enable retriever‑augmented generation")
cli.add_argument("--model", default="gpt-4o-mini",
                 help="LLM backend: deepseek-chat | gpt-4o | gpt-4o-mini …")
cli.add_argument("--metric", default="euclidean", help="Distance metric: euclidean, mape, correlation, etc.")
cli.add_argument("--length", type=int, default=144)
args = cli.parse_args()
K_NEIGHBOURS = args.shots

from datetime import datetime, timezone
from pathlib import Path
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
flag_str = f"rag-{args.rag}_metric-{args.metric}_shots-{args.shots}_len-{args.length}_model-{args.model}"
out_dir = Path("runs") / f"{ts}__{flag_str}"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "run_info.txt", "w") as f:
    f.write(f"model: {args.model}\n")
    f.write(f"rag:   {args.rag}\n")
    f.write(f"shots: {getattr(args, 'shots', 'n/a')}\n")
    f.write(f"metric:{args.metric}\n")
    f.write(f"length:{args.length}\n")

if args.model.startswith("deepseek"):
    from langchain_deepseek import ChatDeepSeek
    llm = ChatDeepSeek(model=args.model)
else:                                  
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=args.model)

from build_prompt import promptFactory
if args.rag:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from build_prompt import RAGpromptFactory
    from RAG.darts_retriever import build_retriever_from_dataset
    from RAG.darts_retriever import DartsRetriever
    from utils.darts_dataset import SamplingDatasetPast
    from darts import TimeSeries

from langchain.schema import HumanMessage
from func import calc_mae, calc_rmse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_formatter.base import DataFormatter

dataset_list = ['iglu', 'dubosson', 'hall', 'colas', 'weinstock']
diabetes_map = {"iglu": 2, "dubosson": 1, "hall": 0, "colas": 0, "weinstock": 1}

HORIZON = 12
N_RUNS = 5
HIST_LEN = args.length
PRED_COLS = [f"pred_{i+1}" for i in range(HORIZON)]
TRUE_COLS = [f"true_{i+1}" for i in range(HORIZON)]

summary_rows = []      

for dataset in dataset_list:
    diabetes_type = diabetes_map[dataset]


    with open(f'./config/{dataset}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    formatter = DataFormatter(config)
    id_test = formatter.test_data.loc[
        ~formatter.test_data.index.isin(formatter.test_idx_ood)
    ]

    metrics_rows, preds_rows = [], []
    prompt_f = open(out_dir / f"prompts_{dataset}.txt", "w")
            
    for segment_id in id_test["id_segment"].unique():
        seg = id_test[id_test["id_segment"] == segment_id]
        history   = seg["gl"].tolist()[:HIST_LEN]
        timestamp = seg["time"].iloc[0]
        future    = seg["gl"].tolist()[HIST_LEN:HIST_LEN + HORIZON]

        if args.rag:
            train_seg = formatter.train_data[formatter.train_data["id_segment"] == segment_id]
            if train_seg.empty:
                prompt_text = promptFactory(diabetes_type, history, timestamp)
            else:
                arr = train_seg["gl"].to_numpy(dtype=float)
                windows, futures = [], []
                for i in range(0, len(arr) - HIST_LEN - HORIZON + 1, STRIDE):
                    windows.append(arr[i : i + HIST_LEN])
                    futures.append(arr[i + HIST_LEN : i + HIST_LEN + HORIZON])
                retriever = DartsRetriever(windows, futures, metric=args.metric)
                neighbours = retriever(np.asarray(history), k=args.shots)
                prompt_text = RAGpromptFactory(
                    diabetes_type = diabetes_type,
                    history = history,
                    timestamp = timestamp,
                    retrieved_data = neighbours,
                )
        else:
            prompt_text = promptFactory(diabetes_type, history, timestamp)

        prompt_f.write(f"--- {segment_id} ---\n{prompt_text}\n\n")
        runs = []
        for _ in range(N_RUNS):
            raw   = llm([HumanMessage(content=prompt_text)]).content
            preds = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", raw)][:HORIZON]
            preds += [np.nan] * (HORIZON - len(preds)) 
            runs.append(preds)

        avg_preds = np.nanmean(runs, axis=0)

        rmse = calc_rmse(avg_preds, future)
        mae  = calc_mae(avg_preds,  future)

        metrics_rows.append({"segment_id": segment_id, "rmse": rmse, "mae": mae})

        preds_row = {
            "segment_id": segment_id,
            "history": ",".join(map(str, history)),
            **{c: avg_preds[i] for i, c in enumerate(PRED_COLS)},
            **{c: future[i]    for i, c in enumerate(TRUE_COLS)},
        }
        preds_rows.append(preds_row)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df   = pd.DataFrame(preds_rows)

    overall_rmse = metrics_df["rmse"].median()
    overall_mae  = metrics_df["mae"].median()

    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([{
            "segment_id": "OVERALL",
            "rmse": overall_rmse,
            "mae":  overall_mae
        }])],
        ignore_index=True
    )
    prompt_f.close()

    metrics_df.to_csv(out_dir / f"metrics_{dataset}.csv", index=False)
    preds_df.to_csv(out_dir / f"preds_{dataset}.csv",   index=False)


    print(f"{dataset.upper():<10}  RMSE(median)={overall_rmse:.2f}  "
          f"MAE(median)={overall_mae:.2f}")

    summary_rows.append({
        "dataset": dataset,
        "rmse_median": overall_rmse,
        "mae_median":  overall_mae
    })

pd.DataFrame(summary_rows).to_csv(out_dir / "metrics_summary.csv", index=False)
print("\nSaved per‑dataset files plus overall summary → metrics_summary.csv")
