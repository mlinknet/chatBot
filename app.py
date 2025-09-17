from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, re, pickle, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

# --- OpenAI APIキーは環境変数で設定 ---
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 定数 =====
SIMILARITY_THRESHOLD = 0.5
UNKNOWN_LOG_FILE = "unknown_questions.log"

# ===== FAQ データ =====
faq_data = {
    "JR三ノ宮駅からの行き方を教えてください。": "徒歩でお越しの場合は...",
    "三ノ宮駅からどうやって行けばいいですか？": "徒歩でお越しの場合は...",
    "車で行く場合のルートを教えてください。": "（大阪方面よりお越しの方）...",
    "車でのアクセス方法は？": "（大阪方面よりお越しの方）...",
    "時間貸し駐車場はありますか？料金はいくらですか？": "当ビル地下1階にございます...",
    "駐車場割引はありますか？": "割引価格で「駐車回数券」を...",
    "駐車場の営業時間は何時から何時までですか？": "営業時間は7:00から23:00まで...",
    "飲食店はありますか？": "地下1階が飲食店フロアで...",
}

faq_questions = list(faq_data.keys())

# --- Embeddings生成・保存・読み込み ---
if not os.path.exists("faq_embeddings.pkl"):
    print("✅ Embeddings を新規生成中...")
    faq_embeddings = []
    for q in faq_questions:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=q
        ).data[0].embedding
        faq_embeddings.append(emb)
    with open("faq_embeddings.pkl", "wb") as f:
        pickle.dump(faq_embeddings, f)
    print("✅ Embeddings 保存完了")
else:
    with open("faq_embeddings.pkl", "rb") as f:
        faq_embeddings = pickle.load(f)
    print("✅ Embeddings ファイル読み込み完了")

# --- FastAPI 初期化 ---
app = FastAPI()

# --- CORS設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- index.html GET ---
@app.get("/")
def serve_html():
    return FileResponse("index.html")

# --- Pydanticモデル ---
class Question(BaseModel):
    question: str

# --- 正規化関数 ---
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[？?。．]+$", "", text)
    return text

# --- FAQ検索関数 ---
def get_faq_answer(user_question):
    normalized_question = normalize_text(user_question)
    normalized_faq = {normalize_text(k): v for k, v in faq_data.items()}

    if normalized_question in normalized_faq:
        return {"answer": normalized_faq[normalized_question], "candidates": []}

    # 類似度検索
    user_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_question
    ).data[0].embedding

    similarities = cosine_similarity([user_emb], faq_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:3]
    candidates = [
        {"question": faq_questions[i], "similarity": float(similarities[i])}
        for i in top_indices
    ]
    best_index = int(np.argmax(similarities))
    best_score = similarities[best_index]

    if best_score < SIMILARITY_THRESHOLD:
        with open(UNKNOWN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(user_question + "\n")
        return {
            "answer": (
                "申し訳ございません。該当する回答が見つかりませんでした。\n"
                "お手数ですが、こちらから直接お問い合わせください。\n"
                "👉 078-251-3141\n"
                "👉 https://www.kobe-citc.com/contact/"
            ),
            "candidates": candidates
        }

    return {"answer": faq_data[faq_questions[best_index]], "candidates": candidates}

# --- POSTエンドポイント ---
@app.post("/get_answer")
async def get_answer(q: Question):
    return get_faq_answer(q.question)
