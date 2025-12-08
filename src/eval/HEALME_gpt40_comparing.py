import json
from openai import OpenAI
from tqdm import tqdm

# OpenAI 클라이언트 설정
client = OpenAI(api_key="")

# 데이터셋 JSONL 파일 경로
input1_file_path = "./data/"
input2_file_path = "./data/"
output_file_path = "./data/eval_output/"

def create_chat_completion(system_input, user_input, model="gpt-4o", temperature=0.7, max_tokens=550):
    messages = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if response and response.choices and len(response.choices) > 0:
            return {
                "content": response.choices[0].message.content.strip(),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            }
        else:
            print("Invalid response structure or no choices returned.")
            return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

def extract_score(response_text):
    import re, json
    match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON found in response.")



def score_and_compare(data1, data2):
    results = []
    system_prompt = (
        "You are an expert in evaluating psychotherapy counseling responses. "
        "Please assess each response based on the following three criteria:\n\n"
        "1. Empathy: How well does the counselor understand and support the client’s emotional state? (Score 0–3)\n"
        "2. Logical Coherence: How logically consistent and well-reasoned is the response? (Score 0–3)\n"
        "3. Guidance: How practical and actionable is the advice provided to the client? (Score 0–3)\n\n"
        "Detailed scoring criteria:\n\n"
        "Empathy:\n"
        "0: Completely ignores the client’s emotions and statements\n"
        "1: Summarizes the client’s words but does not reflect their emotions\n"
        "2: Understands and responds to both the content and the emotions\n"
        "3: Accurately reads the client’s emotions and goes beyond simple repetition or summary to offer emotional support\n\n"
        "Logical Coherence:\n"
        "0: The response lacks logic and coherence; fails to focus on the client’s issues and contains logical fallacies, contradictory perspectives, or excessive subjectivity\n"
        "1: The response shows some logic but lacks overall coherence; fails to identify reasoning based on the client’s statements or uses vague expressions\n"
        "2: The response is mostly clear and logically consistent, based on sufficient reasoning and reasonable assumptions, though minor logical issues may be present\n"
        "3: The response includes sufficient reasoning and clear assumptions, demonstrates thorough and consistent logical development, contains no logical errors or contradictions, and presents a persuasive conclusion\n\n"
        "Guidance:\n"
        "0: Lacks both specificity and practicality; no goals, action plans, or consideration of real-life situations\n"
        "1: The suggestions are somewhat specific and practical but lack clarity\n"
        "2: The suggestions are very specific and practical, including actionable plans and recommendations tailored to the client’s issues and needs\n"
        "3: The suggestions are highly specific, practical, and realistic, taking various factors and real-life circumstances into account, showing feasibility and executability. Additionally, the response offers insight into the client’s future growth and improvement\n\n"
        "Return the evaluation in the following JSON format:\n"
        "{\"Empathy\": score, \"Logical Coherence\": score, \"Guidance\": score}"

    )
    for entry1, entry2 in tqdm(zip(data1, data2), desc="Scoring and comparing", total=len(data1)):
        index = entry1["questionID"]
        answer1 = entry1["model_answer"]  # QAB
        answer2 = entry2["model_answer"]  # Q


        user_prompt1 = (
            f"Below is a counseling response to a client’s question. Please evaluate the response based on the criteria:\n\n"
            f"{answer1}\n\n"
            "Evaluation:"
        )

        user_prompt2 = (
            f"Below is a counseling response to a client’s question. Please evaluate the response based on the criteria:\n\n"
            f"{answer2}\n\n"
            "Evaluation:"
        )


        # GPT 평가 결과 생성
        score1_response = create_chat_completion(system_prompt, user_prompt1)
        score2_response = create_chat_completion(system_prompt, user_prompt2)

        try:
            score1 = extract_score(score1_response["content"]) if score1_response else {"Empathy": 0, "Logical Coherence": 0, "Guidance": 0}
            score2 = extract_score(score2_response["content"]) if score2_response else {"Empathy": 0, "Logical Coherence": 0, "Guidance": 0}
        except ValueError:
            score1 = {"Empathy": 0, "Logical Coherence": 0, "Guidance": 0}
            score2 = {"Empathy": 0, "Logical Coherence": 0, "Guidance": 0}

        # 총점 비교
        sum1 = sum(map(int, score1.values()))
        sum2 = sum(map(int, score2.values()))
        winner = "QAB" if sum1 > sum2 else "Q" if sum2 > sum1 else "Tie"


        # 결과 추가
        results.append({
            "questionID": index,
            "QAB_score": score1,
            "Q_score": score2,
            "winner": winner,
            "QAB_eval_gen": score1_response,
            "Q_eval_gen": score2_response,
            "QAB_model_answer": answer1,
            "Q_model_answer": answer2,

        })
        
    return results

if __name__ == "__main__":
    data1 = load_jsonl(input1_file_path)
    data2 = load_jsonl(input2_file_path)

    if len(data1) != len(data2):
        print("Error: The input files have different lengths.")
    else:
        scored_data = score_and_compare(data1, data2)
        save_jsonl(scored_data, output_file_path)
        print(f"Scored and compared data saved to {output_file_path}")
