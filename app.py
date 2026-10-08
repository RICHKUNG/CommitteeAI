"""CommitteeAI (萬能唬爛王主席系統) — local script version of the Colab notebook.

A chair LLM reads the user's question, invents a committee of role-played
agents (with their own system prompts and tasks), lets them discuss for N
rounds, then writes a final summary. Gradio UI with live streaming and a stop
button.

    export GEMINI_API_KEY=...   # Google AI Studio key
    python app.py
"""
import os
import re
import threading

import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------- LLM setup
gemini_flash = "gemini-2.5-flash"
gemini_pro = "gemini-2.5-pro"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set GEMINI_API_KEY to a Google AI Studio API key.")

chairman = OpenAI(api_key=api_key, base_url=base_url)
member_model = OpenAI(api_key=api_key, base_url=base_url)

committee_log = []
strategies = []
committee = []
stop_event = threading.Event()
# stop_flag = lambda: False

chairman_sys_prompt = """
你是萬能委員會的主席、 AI 任務規劃者。
你將會設計幾個AI Agent，他們會依你指定的順序發言

請根據使用者問題，完成以下任務：

1. 發想任意多種不同的解決策略（不同角度思考），或直接模擬情境
2. 為每種策略指定一種LLM的人格、角色或職業
3. 為每個LLM人格設定編寫適合的 system prompt
4. 分解使用者問題，為每個LLM人格指定要執行的任務(不需要設定主持人)
5. 輸出清單格式如下：

--- 策略列表 ---

1. ...
2. ...

--- 委員角色與 prompt ---

- 姓名：xxx
  system prompt: xxx
  負責任務: xxx
　情緒: xxx

--- 結束 ---
注意事項:
1. 請使用英文或台灣繁體中文輸出。
2. 可以使用純文字加上emoji，千萬不要使用Markdown
3. 千萬不要使用Markdown
4. 委員們看不到使用者輸入
5. 撰寫負責任務時請在一行內寫完
"""

def chairman_plan(llm_model, user_question, temperature=0.8):
    # 初始化
    global strategies, committee, committee_log
    committee_log = []
    strategies = []
    committee = []

    chairman_response = chairman.chat.completions.create(
        model=llm_model,
        reasoning_effort="high",
        temperature=temperature,
        messages=[
        {"role": "system", "content": chairman_sys_prompt},
        {"role": "user", "content": user_question}
        ]
    )
    chairman_text = chairman_response.choices[0].message.content
    #print("\n📜 規劃結果:\n", chairman_text)

    # 提取策略列表
    strategy_list_pattern = r"--- 策略列表 ---(.*?)--- 委員角色與 prompt ---"
    strategy_list_match = re.search(strategy_list_pattern, chairman_text, re.DOTALL)

    if strategy_list_match:
        strategy_list_text = strategy_list_match.group(1).strip()
        for line in strategy_list_text.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line:
                strategies.append(cleaned_line)
        strategies = [line.strip() for line in strategy_list_text.split('\n') if line.strip()]

    # 提取委員資訊
    pattern = r"姓名：(.+?)\s+system prompt:\s+(.+?)\s+負責任務:\s+(.+?)\s+情緒"
    matches = re.findall(pattern, chairman_text, re.DOTALL)

    for name, prompt, strategy in matches:
        committee.append({
            "name": name.strip(),
            "system_prompt": prompt.strip(),
            "strategy": strategy.strip()
        })
    markdown_strategies = "\n".join(strategies)

    markdown_committee = "| 姓名 | 設定 | 負責任務 |\n"
    markdown_committee += "|---|---|---|\n"
    for member in committee:
        markdown_committee += f"| {member['name']} | {member['system_prompt']} | {member['strategy']} |\n"
    return markdown_strategies, markdown_committee

def member_speak(member, round_num, model=gemini_flash, temperature=0.7, reasoning_effort="low", stop_flag=False):
    global committee_log
    if stop_flag:
        print("🚨 使用者要求中止！")
        return
    messages = [{"role": "system", "content": member["system_prompt"]}]

    #if round_num == 1:
    messages.append({"role": "user", "content": f"你是{member['name']}，你的任務是：{member['strategy']}"})
    #else:
    messages.append({
            "role": "user",
            "content": f"第 {round_num} 輪討論：請根據前面的討論內容，進一步闡述你的觀點或回應其他成員的發言，，要再重複一次或者換觀點也行，請務必發言。"
        })

    for prev_name, prev_content, prev_round in committee_log:
        if prev_name == member["name"]: #自己說的
            messages.append({
            "role": "assistant",
            "content": prev_content
            })
        else:
            messages.append({
                "role": "user",
                "content": f"第 {prev_round} 輪 - 成員 {prev_name} 發言：{prev_content}"
            })
        messages.append({
            "role": "assistant",
            "content": f"我聽到了 {prev_name} 第 {prev_round} 輪的發言。"
        })

    messages.append({
            "role": "user",
            "content": f"現在第 {round_num} 輪輪到你發言，請根據前面的討論內容提出你的觀點，要再重複一次或者換觀點也行，請務必發言。"
    })

    reply = member_model.chat.completions.create(
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        messages=messages
    )
    # print(f"\n🎭 第 {round_num} 輪 - 正在請 {member['name']} 回應...")

    if reply.choices is not None:
        content = reply.choices[0].message.content
        committee_log.append((member["name"], content, round_num))
        # print(f"🗣️ 第 {round_num} 輪 - {member['name']} 說：\n{content}\n")
    else:
        committee_log.append((member["name"], "(什麼都沒說，沉默不語)", round_num))
        # print(f"❌ 第 {round_num} 輪 - {member['name']} 無話可說。")
    yield committee_log[:]

def sum_up_committee_log(llm_model, user_question, temperature=0.7):
    summary_prompt = """
    剛才你聽取了以下委員們的意見：
    """
    global committee_log
    for name, content, round_num in committee_log:
        summary_prompt += f"第{round_num}輪【{name}】：\n{content}\n\n"

    summary_prompt += f"請你整合以上建議，並用適當的語氣對使用者給出一份結論建議，幫助使用者處理：「{user_question}」這個問題。請注意不要問使用者更多問題或索取更多資訊"

    summary = chairman.chat.completions.create(
        model=llm_model,
        temperature=temperature,
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": user_question}
        ]
    )

    final_output = summary.choices[0].message.content
    #print("\n🌟 主席總結建議：\n", final_output)
    return final_output

def generate_committee_discussion(user_question, temperature, reasoning_effort, chair_model_choice, member_model_choice, rounds, show_log):
    global committee_log, strategies, committee
    strategies = []
    committee = []
    committee_log = []

    stop_event.clear()

    member_llm_model = gemini_flash if member_model_choice == "gemini 2.5 flash" else gemini_pro
    chair_llm_model = gemini_flash if chair_model_choice == "gemini 2.5 flash" else gemini_pro

    # 主席策略規劃
    mark_strat, mark_comm = chairman_plan(chair_llm_model, user_question, temperature=temperature)
    yield mark_strat, mark_comm, "","", 0

    for round_num in range(1, rounds + 1):
        for i, member in enumerate(committee):
            # 檢查是否需要停止
            if stop_event.is_set():
                print("🚨 使用者要求中止討論！")
                final_log = format_committee_log_markdown(committee_log) if show_log else ""
                yield mark_strat, mark_comm, "討論已被使用者中止", gr.update(visible=show_log, value=final_log), round_num
                return

            # 呼叫委員發言函數
            member_generator = member_speak(
                member,
                round_num,
                model=member_llm_model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                stop_flag=stop_event.is_set()
            )

            for log in member_generator:
                if stop_event.is_set():
                    print("🚨 委員發言中途中止！")
                    final_log = format_committee_log_markdown(committee_log) if show_log else ""
                    yield mark_strat, mark_comm, "討論已被使用者中止", gr.update(visible=show_log, value=final_log), round_num
                    return
                log_md = format_committee_log_markdown(log) if show_log else ""
                log_md += "\n\n> 💬 下一位委員正在思考中..."
                yield mark_strat, mark_comm, "", gr.update(visible=show_log, value=log_md), round_num
        if stop_event.is_set():
            print("🚨 輪次結束時中止！")
            final_log = format_committee_log_markdown(committee_log) if show_log else ""
            yield mark_strat, mark_comm, "討論已被使用者中止", gr.update(visible=show_log, value=final_log), round_num
            return

    # 結果彙整
    summary = sum_up_committee_log(chair_llm_model, user_question, temperature=temperature)
    final_log = format_committee_log_markdown(committee_log) if show_log else ""

    yield mark_strat, mark_comm, summary, gr.update(visible=show_log, value=final_log), rounds

def format_committee_log_markdown(log):
    markdown_output = ""
    for name, content, round_num in log:
        markdown_output += f"### 第 {round_num} 輪 - {name}\n\n"
        markdown_output += f"{content}\n\n---\n\n"
    return markdown_output

starter = '''
### 📝 使用說明

歡迎來到 **唬爛王主席系統**！這是一個讓多位語言模型扮演委員進行「角色扮演式討論」的系統，無論你有什麼問題、點子、煩惱，還是需要腦力激盪，都可以丟進來讓多位委員來幫你集思廣益 💡

---

### 🚀 使用步驟

1. **輸入你的問題**

   * 👉 可以是任何主題，例如：「午餐要吃什麼」、「讓委員們玩誰是臥底」或「模擬一群人爭辯地球是圓的還是平的」
   * Hint: 就像使用其他LLM一樣，給更多資訊會幫助主席回答得更好!

2. **左側有進階設定**（可選）：

   * `🎛️ Temperature`: 控制模型創意程度（越高越跳躍、越不可預測）
   * `🧠 主席/委員模型`: 可選擇 **gemini 2.5 pro**（較強但較慢）或 **flash**（速度較快）
   * `🌀 思考強度`: 選擇委員們的思考深度（低、中、高）
   * `🔁 發言輪數`: 模擬委員們來回討論幾輪
   * `📜 顯示發言紀錄`: 是否要看每位委員的發言過程

3. **按下「開始召喚討論！」**

   * 🎤 主席會先分析問題並規劃討論策略，接著每位委員就會輪流發言並討論～

   * 最後主席會根據整體討論幫你總結出有幫助的建議！

---

### ☢️ 注意事項

1. 若你覺得委員們太吵，可以按下 **🔴 終止討論** 讓他們閉嘴(你無法終止主席)

2. 本系統無記憶(連續對話)功能，請放心使用

3. debug指南: 重新執行應該可以解決大部分問題(可能是Server過載)

---

本系統由 **清華大學 電機系 孔祥有** 製作

'''

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 萬能唬爛王主席系統 · CommitteeAI")
    gr.Markdown(starter)
    with gr.Row():
        # 左側：進階選項
        with gr.Column(scale=1):
            with gr.Accordion("🔧 進階選項", open=False):
                temperature_slider = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.05, label="Temperature")
                chair_model_radio = gr.Radio(choices=["gemini 2.5 flash", "gemini 2.5 pro"], value="gemini 2.5 pro", label="主席模型")
                member_model_radio = gr.Radio(choices=["gemini 2.5 flash", "gemini 2.5 pro"], value="gemini 2.5 flash", label="委員模型")
                reasoning_dropdown = gr.Dropdown(choices=["low", "medium", "high"], value="low", label="委員思考強度")
                rounds_slider = gr.Slider(minimum=1, maximum=15, value=2, step=1, label="發言輪數")
                show_log_checkbox = gr.Checkbox(label="顯示委員發言紀錄", value=True)

        # 中間：使用者輸入與主席區塊
        with gr.Column(scale=3):
            user_input = gr.Textbox(label="請輸入您的問題", placeholder="例如：請給我萬聖節裝扮的點子")
            with gr.Row():
                start_button = gr.Button("🚀 開始召喚討論！", variant="primary")
                stop_button = gr.Button("🔴 終止討論", visible=False, variant="stop")
            progress_bar = gr.Slider(minimum=0, maximum=rounds_slider.value, step=1, interactive=False, label="進度")
            gr.Markdown("## 🎯 主席分析策略")
            strategies_output = gr.Markdown()

            gr.Markdown("## 👥 委員會組成")
            committee_output = gr.Markdown()


        # 右側：主席總結
        with gr.Column(scale=3):
            gr.Markdown("## 🎓 主席總結建議")
            summary_output = gr.Markdown()
        running_state = gr.State(value=True)  # 是否正在討論中？
        global stop_flag
        stop_flag = lambda: not running_state
    with gr.Row():
        with gr.Group(visible=True) as log_group:
                        gr.Markdown("## 🗂️ 委員討論日誌")
                        log_output = gr.Markdown()

    start_button.click(
        fn=lambda: stop_event.clear(),  # reset 停止狀態
        outputs=[]
    ).then(
        fn=lambda rounds: gr.update(value=0, maximum=rounds),
        inputs=rounds_slider,
        outputs=progress_bar
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[start_button, stop_button]
    ).then(
        fn=generate_committee_discussion,
        inputs=[
            user_input,
            temperature_slider,
            reasoning_dropdown,
            chair_model_radio,
            member_model_radio,
            rounds_slider,
            show_log_checkbox,
        ],
        outputs=[
            strategies_output,
            committee_output,
            summary_output,
            log_output,
            progress_bar,
        ],
    ).then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[start_button, stop_button]
    )

    stop_button.click(
        fn=lambda: stop_event.set(),  # 真正的中止動作
        outputs=[]
    )

if __name__ == "__main__":
    demo.queue().launch()
