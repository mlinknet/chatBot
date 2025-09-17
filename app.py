from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle, os, re
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI APIキーは環境変数で設定
openai.api_key = os.getenv("OPENAI_API_KEY")

# 定数
SIMILARITY_THRESHOLD = 0.5
UNKNOWN_LOG_FILE = "unknown_questions.log"

# ===== FAQ データ =====
faq_data = {
    # 【アクセス】JR三ノ宮駅からの行き方
    "JR三ノ宮駅からの行き方を教えてください。": "徒歩でお越しの場合は、阪急百貨店西沿いの「フラワーロード」を南下すると一番分かりやすいです。約10分南下（道路を挟んで西側には市役所や東遊園地が見えます）し、コンビニの角を左折（東方向）してください。道なりに行くと、茶色い貿易センタービルが見えてきます。駅前からは徒歩約14分です。https://maps.app.goo.gl/NqcPitiAm6mvg1y76",
    "三ノ宮駅からどうやって行けばいいですか？": "徒歩でお越しの場合は、阪急百貨店西沿いの「フラワーロード」を南下すると一番分かりやすいです。約10分南下（道路を挟んで西側には市役所や東遊園地が見えます）し、コンビニの角を左折（東方向）してください。道なりに行くと、茶色い貿易センタービルが見えてきます。駅前からは徒歩約14分です。https://maps.app.goo.gl/NqcPitiAm6mvg1y76",
    "JR三ノ宮駅から歩いて何分かかりますか？": "徒歩でお越しの場合は、阪急百貨店西沿いの「フラワーロード」を南下すると一番分かりやすいです。約10分南下（道路を挟んで西側には市役所や東遊園地が見えます）し、コンビニの角を左折（東方向）してください。道なりに行くと、茶色い貿易センタービルが見えてきます。駅前からは徒歩約14分です。https://maps.app.goo.gl/NqcPitiAm6mvg1y76",

    # 【アクセス】車での行き方
    "車で行く場合のルートを教えてください。": "（大阪方面よりお越しの方）阪神高速3号神戸線「生田川出口」で下車し直進、「浜辺通四丁目」の交差点を右折→２つ目の交差点の信号を左折→１つ目の交差点を左折→１つ目の信号を直進後、約50ｍほどで地下駐車場入口に到着 https://maps.app.goo.gl/HBuqm9eeR4V87jTVA （岡山・姫路方面よりお越しの方）阪神高速3号神戸線「京橋出口」で下車し1つ目の信号を左折→2号線を横断し1つ目の信号を右折→さらに1つ目の信号を右折後、約50mほどで地下駐車場に到着 https://maps.app.goo.gl/Ufh2amPmQbzyNqWW9",
    "車でのアクセス方法は？": "（大阪方面よりお越しの方）阪神高速3号神戸線「生田川出口」で下車し直進、「浜辺通四丁目」の交差点を右折→２つ目の交差点の信号を左折→１つ目の交差点を左折→１つ目の信号を直進後、約50ｍほどで地下駐車場入口に到着 https://maps.app.goo.gl/HBuqm9eeR4V87jTVA （岡山・姫路方面よりお越しの方）阪神高速3号神戸線「京橋出口」で下車し1つ目の信号を左折→2号線を横断し1つ目の信号を右折→さらに1つ目の信号を右折後、約50mほどで地下駐車場に到着 https://maps.app.goo.gl/Ufh2amPmQbzyNqWW9",

    # 【駐車場】時間貸し駐車場の有無・料金
    "時間貸し駐車場はありますか？料金はいくらですか？": "当ビル地下1階にございます。料金は、20分200円、1日最大3,000円です。詳細はhttps://www.kobe-citc.com/access/carpark/を御覧ください。",
    "駐車場の料金について教えてください。": "当ビル地下1階にございます。料金は、20分200円、1日最大3,000円です。詳細はhttps://www.kobe-citc.com/access/carpark/を御覧ください。",
    "駐車場はありますか？": "当ビル地下1階にございます。料金は、20分200円、1日最大3,000円です。詳細はhttps://www.kobe-citc.com/access/carpark/を御覧ください。",
    "車を停める場所はありますか？": "当ビル地下1階にございます。料金は、20分200円、1日最大3,000円です。詳細はhttps://www.kobe-citc.com/access/carpark/を御覧ください。",

    # 【駐車場】割引券の有無
    "駐車場の割引券はありますか？": "割引価格で「駐車回数券」を、地下1階のミニコンビニ「センターショップ」（https://www.kobe-citc.com/restaurant/shop06/）で販売しております。1枚から購入可能です。営業時間は「8:30～15:30」です。",
    "駐車場割引はありますか？": "割引価格で「駐車回数券」を、地下1階のミニコンビニ「センターショップ」（https://www.kobe-citc.com/restaurant/shop06/）で販売しております。1枚から購入可能です。営業時間は「8:30～15:30」です。",
    "駐車券の販売場所を教えてください。": "割引価格で「駐車回数券」を、地下1階のミニコンビニ「センターショップ」（https://www.kobe-citc.com/restaurant/shop06/）で販売しております。1枚から購入可能です。営業時間は「8:30～15:30」です。",

    # 【駐車場】営業時間
    "駐車場の営業時間は何時から何時までですか？": "営業時間は7:00から23:00まで。入庫は22:00までとなります。23:00から翌朝7:00までは出庫が出来ませんので、ご注意ください。その際、宿泊料金として留め置き料金1,000円（23:00～翌朝8:00まで）が加算されます。",
    "駐車場は何時から何時まで営業していますか？": "営業時間は7:00から23:00まで。入庫は22:00までとなります。23:00から翌朝7:00までは出庫が出来ませんので、ご注意ください。その際、宿泊料金として留め置き料金1,000円（23:00～翌朝8:00まで）が加算されます。",

    # 【飲食店】飲食店はありますか？
    "飲食店はありますか？": "地下1階が飲食店フロアとなっており、また、12階にも食堂があります。詳しくは（https://www.kobe-citc.com/restaurant/#restaurant）を御覧ください。なお、地下1階の中華料理店以外は、平日昼間のみの営業となっています。",
    "館内で食事できる場所はありますか？": "地下1階が飲食店フロアとなっており、また、12階にも食堂があります。詳しくは（https://www.kobe-citc.com/restaurant/#restaurant）を御覧ください。なお、地下1階の中華料理店以外は、平日昼間のみの営業となっています。",
    "レストランはありますか？": "地下1階が飲食店フロアとなっており、また、12階にも食堂があります。詳しくは（https://www.kobe-citc.com/restaurant/#restaurant）を御覧ください。なお、地下1階の中華料理店以外は、平日昼間のみの営業となっています。",
}

faq_questions = list(faq_data.keys())

# Embeddings読み込みまたは生成
if not os.path.exists("faq_embeddings.pkl"):
    faq_embeddings = []
    for q in faq_questions:
        emb = openai.Embedding.create(input=q, model="text-embedding-3-small")["data"][0]["embedding"]
        faq_embeddings.append(emb)
    with open("faq_embeddings.pkl", "wb") as f:
        pickle.dump(faq_embeddings, f)
else:
    with open("faq_embeddings.pkl", "rb") as f:
        faq_embeddings = pickle.load(f)

# 正規化関数
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[？?。．]+$", "", text)
    return text

# FAQ検索関数
def get_faq_answer(user_question):
    normalized_question = normalize_text(user_question)
    normalized_faq = {normalize_text(k): v for k, v in faq_data.items()}

    if normalized_question in normalized_faq:
        return {"answer": normalized_faq[normalized_question], "candidates": []}

    # Embedding作成
    user_emb = openai.Embedding.create(input=user_question, model="text-embedding-3-small")["data"][0]["embedding"]
    similarities = cosine_similarity([user_emb], faq_embeddings)[0]

    # 類似度上位3件
    top_indices = similarities.argsort()[::-1][:3]
    candidates = [{"question": faq_questions[i], "similarity": float(similarities[i])} for i in top_indices]

    best_index = int(np.argmax(similarities))
    best_score = similarities[best_index]

    if best_score < SIMILARITY_THRESHOLD:
        with open(UNKNOWN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(user_question + "\n")
        return {
            "answer": "申し訳ございません。該当する回答が見つかりませんでした。\nお手数ですが、こちらから直接お問い合わせください。",
            "candidates": candidates
        }

    return {"answer": faq_data[faq_questions[best_index]], "candidates": candidates}

# FastAPI初期化
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# index.html提供
@app.get("/")
def serve_html():
    return FileResponse("index.html")

# Pydanticモデル
class Question(BaseModel):
    question: str

# POSTエンドポイント
@app.post("/get_answer")
async def get_answer(q: Question):
    return get_faq_answer(q.question)