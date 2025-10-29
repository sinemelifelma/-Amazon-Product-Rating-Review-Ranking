
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most important problems in e-commerce is the accurate calculation of post-purchase product ratings.
# Solving this problem means greater customer satisfaction for the e-commerce platform, better product visibility
# for sellers, and a smooth shopping experience for buyers.
#
# Another issue is the proper ranking of product reviews. Since misleading reviews being highlighted can directly
# affect product sales, this can lead to both financial losses and customer attrition.
#
# By solving these two fundamental problems, e-commerce platforms and sellers can increase their sales,
# while customers can enjoy a seamless purchasing journey.

###################################################
# Dataset Story
###################################################

# This dataset contains Amazon product data, including various metadata and product categories. It focuses on the
# Electronics category and includes user ratings and reviews for the most-reviewed product in that category.

# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: Username
# helpful: Helpfulness rating of the review
# reviewText: Full review text
# overall: Product rating
# summary: Review summary
# unixReviewTime: Review time (Unix timestamp)
# reviewTime: Raw review date
# day_diff: Number of days since the review was posted
# helpful_yes: Number of users who found the review helpful
# total_vote: Total number of votes received for the review


###################################################
# TASK 1: Calculate the Average Rating Based on Recent Reviews and Compare It with the Existing Average Rating
###################################################

# In the shared dataset, users have rated and reviewed a product. The goal of this task is to weight the ratings
# according to the review dates and evaluate how time affects the overall score. Finally, we will compare the original
# average rating with the time-weighted average rating.

###################################################
# Step 1: Read the dataset and calculate the product’s average rating.
###################################################

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 500)
pd.set_option('display.precision', 3)

df = pd.read_csv("amazon_review.csv")
df.head()
df.info()
df["overall"].mean()

###################################################
# Step 2: Calculate the Time-Based Weighted Average Rating
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
# TASK 2: Select 20 reviews to display on the product detail page.
###################################################

###################################################
# Step 1: Create the helpful_no variable.
###################################################

# Note:
# total_vote is a number of total up-down.
# up means helpful.
# There is no helpful_no variable in the dataset, so we should produce it.

from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_rows", None)

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(20)

###################################################
# Step 2: Calculate the score_pos_neg_diff, score_average_rating, and wilson_lower_bound scores and add them to the dataset.
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
    up: number of helpful (helpful_yes) votes
    down: number of unhelpful (helpful_no) votes
    confidence: confidence level (default: 0.95)
    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = up / n
    return (phat + z**2 / (2*n) - z * np.sqrt((phat*(1 - phat) + z**2 / (4*n)) / n)) / (1 + z**2 / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

##################################################
# Step 3: Select 20 reviews and interpret the results.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)
