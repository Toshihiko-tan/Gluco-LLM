import numpy as np

def promptFactory(diabetes_type, history, timestamp):
    match diabetes_type:
        case 1:
            diabetes_type = "type 1 diabetes patient"
        case 2:
            diabetes_type = "type 2 diabetes patient"
        case 0:
            diabetes_type = "healthy individual"
        case _:
            diabetes_type = "individual with unknown diabetes status"
    hist_str = ", ".join(str(int(x)) for x in history)
    prompt = f"""
    You are looking at CGM data of a {diabetes_type}. Each reading has a 5-minute interval,
    and the first reading is recorded at {timestamp}. Glucose levels range from 40–400mg/dL. 
    Assume the patient continues with their current lifestyle. Predict the next 12 glucose readings.
    Return the forecast as a sequence of numbers, do not include any other information (e.g., comments) in the forecast.
    Historical CGM readings: {hist_str}
    Predicted CGM readings: """
    return prompt

def RAGpromptFactory(diabetes_type, history, timestamp, retrieved_data):
    label = {0: "healthy individual", 1: "type 1 diabetes patient", 2: "type 2 diabetes patient"}.get(diabetes_type, "individual")
    fmt = lambda v: ", ".join(str(int(x)) for x in v)
    examples = []
    for i, (h, f, _) in enumerate(retrieved_data, 1):
        examples.append(f"Example {i}:\nHistorical CGM readings: {fmt(h)}\nPredicted  CGM readings: {fmt(f)}")
    fewshot = "\n\n".join(examples)
    header = (
        f"You are looking at CGM data of a {label}. Each reading is taken every 5 minutes; "
        f"the first reading below corresponds to {timestamp}. Glucose values are in mg/dL "
        f"(typical range 40‑400). Assume the patient continues their current lifestyle. "
        f"Your task is to forecast the next 12 readings.\n\n"
        f"Return **only** the 12 numbers, comma‑separated, on the line that starts with "
        f"'Predicted  CGM readings:' — do not add commentary."
    )
    current = f"Current data:\nHistorical CGM readings: {fmt(history)}\nPredicted  CGM readings: "
    return "\n\n".join([header, fewshot, current])