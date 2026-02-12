# 🎬 Movie-Recommender-System 🍿🔮

Build Status 🟢 | Python 🐍 | PEP8 ✅ | Gitmoji 😄

The Project is Live At:- **Movie Recommender System**


## 📖 Introduction

The **Movie Recommender System** is a powerful machine learning web application that suggests movies based on user preferences 🎥✨. Using content-based filtering and similarity algorithms, this project helps users discover movies similar to their favorites.

Built with **Python** and deployed using **Streamlit**, this application demonstrates how data science and machine learning can enhance user experience in the entertainment domain.

Whether you're a movie enthusiast, data science learner, or ML practitioner, this project offers a practical implementation of recommendation systems 🍿📊.


## ✨ Tech Stack

* **Scripting Language:** Python
* **Web Framework / Hosting:** Streamlit
* **Machine Learning Library:** Scikit-learn
* **Data Processing:** Pandas, NumPy


## 📑 Project Summary

The project consists of the following steps:

### 🧹 Data Cleaning & Preprocessing

The dataset containing movie metadata was cleaned and processed.

* Removed null values
* Combined features like genres, keywords, cast
* Created a transformed feature column for similarity computation

This step ensures accurate and meaningful recommendations.


### 🔬 Model Creation

A **Content-Based Filtering** model was created using **Cosine Similarity**.

* Movie features were vectorized using CountVectorizer
* Cosine similarity matrix was computed
* Based on user input, the system finds top similar movies

This approach helps recommend movies with similar characteristics.


### 💻 User Interface

A simple and interactive UI was developed using **Streamlit**.

* User selects or enters a movie name
* System displays recommended movies
* Clean layout with fast response time

The interface makes the ML model easy to use even for non-technical users.


### 🚀 Deployment with Streamlit

The project is deployed using **Streamlit**, allowing remote access to users.

* Easy cloud deployment
* Accessible via browser
* Lightweight and fast

This makes the system convenient for movie lovers and ML learners alike.


## 🔧 Running This Project

To run the Movie Recommender System locally:

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Abhishek-Rai04/Movie-Recommender-System.git
cd Movie-Recommender-System

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt

### 3️⃣ Run the Project

```bash
streamlit run app.py


## 📊 How It Works

1. User selects a movie 🎬
2. System searches similarity matrix 🔎
3. Top 5 similar movies are displayed 🍿
4. Recommendations are generated instantly ⚡



🌟 If you found this project helpful, don’t forget to give it a star! 🌟
