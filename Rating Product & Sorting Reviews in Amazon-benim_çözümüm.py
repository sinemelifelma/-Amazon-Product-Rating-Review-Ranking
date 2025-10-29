
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı


###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################

import pandas as pd
import numpy as np
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 500)
pd.set_option('display.precision', 3)

df = pd.read_csv("amazon_review.csv")
df.head()
df.info()
df["overall"].mean()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df.sort_values("reviewTime", ascending = False).head(10)
df.sort_values("reviewTime", ascending = True).head(10)

df["overall"].mean()
df["day_diff"].describe()

df.groupby("overall").agg({"day_diff": ["mean"]})
df.loc[df["day_diff"] <= 250, "overall"].mean()
df.loc[(df["day_diff"] > 250) & (df["day_diff"] <= 500), "overall"].mean()
df.loc[(df["day_diff"] > 500) & (df["day_diff"] <= 750), "overall"].mean()
df.loc[(df["day_diff"] > 750) & (df["day_diff"] <= df["day_diff"].max()), "overall"].mean()

df.loc[df["day_diff"] <= 250, "overall"].mean() * 40/100 + \
df.loc[(df["day_diff"] > 250) & (df["day_diff"] <= 500), "overall"].mean() * 30/100 + \
df.loc[(df["day_diff"] > 500) & (df["day_diff"] <= 750), "overall"].mean() * 20/100 + \
df.loc[(df["day_diff"] > 750) & (df["day_diff"] <= df["day_diff"].max()), "overall"].mean() * 10/100

#1. Way
def time_based_weighted_average(dataframe, w1=40, w2=30,  w3=20, w4=10):
    return df.loc[df["day_diff"] <= 250, "overall"].mean() * w1/100 + \
            df.loc[(df["day_diff"] > 250) & (df["day_diff"] <= 500), "overall"].mean() * w2/100 + \
            df.loc[(df["day_diff"] > 500) & (df["day_diff"] <= 750), "overall"].mean() * w3/100 + \
            df.loc[(df["day_diff"] > 750) & (df["day_diff"] <= df["day_diff"].max()), "overall"].mean() * w4/100

time_based_weighted_average(df)

#2. way
def time_based_weighted_average(dataframe, w1=50, w2=25, w3=15, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################

###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_rows", None)

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(20)

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

#score_pos_neg_diff
df["score_pos_neg_diff"] = df["helpful_yes"] - df["helpful_no"]
df.head(20)

#score_average_rating
df["score_average_rating"] = [0 if (df.loc[i, "total_vote"]) == 0
                                else (df.loc[i, "helpful_yes"] / (df.loc[i, "total_vote"]))
                                for i in range(len(df))
]

#My method
df["my method"] = [0 if (df.loc[i, "total_vote"]) == 0
                                else (df.loc[i, "score_pos_neg_diff"] / (df.loc[i, "total_vote"]))
                                for i in range(len(df))
]

df.sort_values("total_vote", ascending = False).head(20)

#wilson_lower_bound
import scipy.stats as st

def wilson_lower_bound(up, down, confidence=0.95):
    """
    up: yararlı (helpful_yes) oy sayısı
    down: yararsız (helpful_no) oy sayısı
    confidence: güven düzeyi (default: 0.95)
    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = up / n
    return (phat + z**2 / (2*n) - z * np.sqrt((phat*(1 - phat) + z**2 / (4*n)) / n)) / (1 + z**2 / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)
