# ⭐ Amazon Product Rating & Review Ranking

## 📍 Project Overview
This project addresses two key e-commerce challenges:
1. Calculating accurate product ratings by considering recency.
2. Sorting reviews to highlight the most reliable feedback.

By applying **time-based weighted averages** and **Wilson Lower Bound scoring**, we ensure that both ratings and reviews fairly represent recent and trustworthy customer opinions.

---

## 🧩 Dataset
The dataset includes Amazon product reviews for the *Electronics* category.

**Main columns:**
- `reviewerID`: Reviewer ID  
- `asin`: Product ID  
- `reviewText`: Review text  
- `overall`: Rating  
- `helpful_yes`: Helpful votes  
- `total_vote`: Total votes  
- `day_diff`: Days since the review  

---

## ⚙️ Methodology

### 1️⃣ Time-Based Weighted Rating
Gives higher importance to recent reviews:
```python
def time_based_weighted_average(dataframe, w1=50, w2=25, w3=15, w4=10):
    ...

### 2️⃣ Review Scoring

Calculates review reliability using three key metrics:

- **`score_pos_neg_diff`** → Difference between helpful and unhelpful votes  
- **`score_average_rating`** → Ratio of helpful votes to total votes  
- **`wilson_lower_bound`** → Statistical confidence score ensuring reliability  

---

## 📊 Results

- **Weighted Rating:** Reflects recent customer sentiment more accurately than a simple average.  
- **Top 20 Reviews:** Determined using the Wilson Lower Bound method to highlight the most reliable reviews.  

---

## 🧠 Learnings

This project demonstrates how combining **statistical confidence** with **recency weighting** can significantly improve how online reviews are ranked and displayed.  
It enhances transparency and helps customers make better purchasing decisions.

---

## 👩‍💻 Author

**Sinem Elif Elma**  
🎓 Data Analyst | Mathematics Educator | Researcher  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/sinem-elif-elma-bab7579b/)  

📘 **Medium Article:** [link here]  
📊 **Kaggle Notebook:** [link here]


## 👩‍💻 Author

**Sinem Elif Elma**  
🎓 Data Analyst | Mathematics Educator | Researcher  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/sinem-elif-elma-bab7579b/)  

📘 **Medium Article:** [(https://medium.com/@sinemelifelma/rating-products-and-sorting-reviews-on-amazon-using-python-58f98075c902)]  
📊 **Kaggle Notebook:** [(https://www.kaggle.com/code/sinemelifelma/rating-product-sorting-reviews-in-amazon)]
