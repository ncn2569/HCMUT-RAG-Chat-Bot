import streamlit as st
import sys
import io
import contextlib
from rag.agent import run_agents
from rag.pipeline import reset_history
from rag.chat.semantic_cache import semantic_cache

st.set_page_config(page_title="HCMUT Agent Chatbot", page_icon="🤖", layout="centered")

# Hàm hỗ trợ để "chụp" lại các lệnh print() bên trong agent.py và đưa lên UI
@contextlib.contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        yield captured_output
    finally:
        sys.stdout = old_stdout

with st.sidebar:
    st.title("⚙️ Config zone (Agent Mode)")
    st.markdown("Giao diện thử nghiệm luồng **Plan-and-Execute Agent**.")

    if st.button("🔄 Bắt đầu cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn! Mình là AI Agent của HCMUT. Mình sẽ tự suy luận để tìm đáp án tốt nhất. Bạn hỏi đi nào!", "avatar": "🤖"}
        ]
        reset_history() 
        st.rerun() 
        
    if st.button("🗑️ Dọn dẹp Semantic Cache", use_container_width=True):
        semantic_cache.flush()
        st.toast("Đã xóa sạch bộ nhớ đệm!", icon="✅")
        
    st.divider() 
    st.caption("Phiên bản chạy bằng LLM Routing, tự quyết định dùng DB hay gọi API Web.")

st.title("🤖 HCMUT Agent Chatbot")
st.markdown("---")  

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn! Mình là AI Agent của HCMUT. Mình sẽ tự suy luận để tìm đáp án tốt nhất. Bạn hỏi đi nào!", "avatar": "🤖"}
    ]

# Render lại tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])
        # Nếu tin nhắn của bot có kèm log suy luận thì in ra
        if "thought_logs" in message and message["thought_logs"]:
            with st.expander("🧠 Xem quá trình suy luận (Plan & Tool)"):
                st.code(message["thought_logs"], language="json")

if prompt := st.chat_input("Hỏi thử 1 câu đánh đố xem Agent xử lý sao nhé..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        # Tạo 1 status box giống "Thinking Process" của Claude
        with st.status("Đang suy nghĩ...", expanded=True) as status:
            
            # Chạy agent và "bắt" toàn bộ lệnh print() lại
            with capture_stdout() as output:
                bot_response = run_agents(query=prompt,history=[])
            
            # Lấy chuỗi log thu được
            logs = output.getvalue()
            
            # Cố gắng bóc tách riêng phần "thought" từ chuỗi log
            thought_text = ""
            if "KẾ HOẠCH CỦA LLM" in logs:
                try:
                    import json
                    json_str = logs.split("KẾ HOẠCH CỦA LLM")[1].split("************************************************")[0].strip()
                    # Cắt bớt các dấu * thừa nếu có
                    json_str = json_str.strip("*").strip()
                    plan_dict = json.loads(json_str)
                    thought_text = plan_dict.get("thought", "")
                except Exception as e:
                    pass
            
            # Hiện dòng thought y hệt Claude
            if thought_text:
                st.markdown(f"_{thought_text}_")
            elif logs.strip():
                st.code(logs, language="text")
            
            # Đóng hộp lại và đổi tên nhãn sau khi nghĩ xong (giống hệt Claude)
            status.update(label="Suy nghĩ xong", state="complete", expanded=False)

        # In câu trả lời chính thức ra màn hình
        st.markdown(bot_response)
    
    # Lưu tin nhắn và kèm theo cả thought_logs để render lại sau này
    st.session_state.messages.append({
        "role": "assistant", 
        "content": bot_response, 
        "thought_logs": logs,
        "avatar": "🤖"
    })
