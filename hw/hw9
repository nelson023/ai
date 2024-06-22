#一個問答系統。這個系統會接收一個問題，然後呼叫openai回答它。
import openai

def setup_openai_api():
    
    openai.api_key = 'YOUR_API_KEY_HERE'

def get_response(question):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",  # 或者選擇其他的模型
            prompt=question,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

def main():
    setup_openai_api()
    while True:
        question = input("請輸入您的問題：")
        if question.lower() == "exit":
            print("結束問答系統。")
            break
        answer = get_response(question)
        print("答案：", answer)

if __name__ == "__main__":
    main()
