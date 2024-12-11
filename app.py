import streamlit as st
import numpy as np
import random
import requests
import html

# Constants
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
CATEGORIES = {
    "General Knowledge": 9,
    "Science & Nature": 17,
    "Sports": 21,
    "History": 23,
    "Geography": 22,
    "Entertainment: Film": 11,
}
NUM_EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
MAX_QUESTIONS = 20

# Initialize Q-Table
q_table = np.zeros((len(DIFFICULTY_LEVELS), len(DIFFICULTY_LEVELS)))


# Fetch Question from API
def fetch_question(difficulty, category_id):
    url = f"https://opentdb.com/api.php?amount=1&difficulty={difficulty}&category={category_id}&type=multiple"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get("results", [])
        if data:
            data = data[0]
            question = html.unescape(data["question"])
            correct_answer = html.unescape(data["correct_answer"])
            options = [html.unescape(opt) for opt in data["incorrect_answers"] + [correct_answer]]
            random.shuffle(options)
            return {"question": question, "correct_answer": correct_answer, "options": options}
    st.error("Failed to fetch question. Please check your internet connection.")
    return None


# Q-Learning Algorithm
def train_agent():
    for episode in range(NUM_EPISODES):
        state = random.choice(range(len(DIFFICULTY_LEVELS)))
        steps = 0
        while steps < 100:  # Max steps per episode
            if random.uniform(0, 1) < EPSILON:
                action = random.choice(range(len(DIFFICULTY_LEVELS)))
            else:
                action = np.argmax(q_table[state])

            reward = random.choices([1, 0], weights=[0.7, 0.3])[0]
            next_state = action
            q_table[state, action] = q_table[state, action] + ALPHA * (
                reward + GAMMA * np.max(q_table[next_state]) - q_table[state, action]
            )
            state = next_state
            steps += 1


train_agent()

# Streamlit UI
st.title("Personalized Learning Tutor")
st.write("Answer the questions and let the system adapt the difficulty dynamically!")

# Genre selection
selected_genre = st.sidebar.selectbox("Select a Genre:", list(CATEGORIES.keys()))
category_id = CATEGORIES[selected_genre]

# Session State for User Tracking
if "state" not in st.session_state:
    st.session_state["state"] = 0
    st.session_state["score"] = 0
    st.session_state["questions"] = 0
    st.session_state["current_question"] = None
    st.session_state["category_id"] = category_id
    st.session_state["user_response"] = None

if st.session_state["category_id"] != category_id:
    st.session_state["category_id"] = category_id
    st.session_state["current_question"] = None

# Quiz Ended
if st.session_state["questions"] >= MAX_QUESTIONS:
    st.subheader("Quiz Completed!")
    st.write(f"**Your Final Score:** {st.session_state['score']} / {MAX_QUESTIONS}")
    if st.button("Restart Quiz"):
        st.session_state["state"] = 0
        st.session_state["score"] = 0
        st.session_state["questions"] = 0
        st.session_state["current_question"] = None
else:
    current_difficulty = DIFFICULTY_LEVELS[st.session_state["state"]]
    if st.session_state["current_question"] is None:
        st.session_state["current_question"] = fetch_question(current_difficulty, st.session_state["category_id"])

    question_data = st.session_state["current_question"]

    if question_data:
        st.write(f"**Genre:** {selected_genre}")
        st.write(f"**Difficulty:** {current_difficulty.capitalize()}")
        st.write(f"**Question:** {question_data['question']}")

        user_response = st.radio(
            "Choose your answer:",
            ["Select an option"] + question_data["options"],
            index=0,
            key=f"q_{st.session_state['questions']}",
        )

        if st.button("Submit Answer"):
            if user_response == "Select an option":
                st.warning("Please select an option!")
            else:
                if user_response == question_data["correct_answer"]:
                    st.success("Correct!")
                    reward = 1
                else:
                    st.error(f"Incorrect! The correct answer is: {question_data['correct_answer']}")
                    reward = 0

                st.session_state["score"] += reward
                st.session_state["questions"] += 1

                # Update state for next question
                current_state = st.session_state["state"]
                action = np.argmax(q_table[current_state])
                st.session_state["state"] = action

                # Fetch next question
                st.session_state["current_question"] = fetch_question(
                    DIFFICULTY_LEVELS[action], st.session_state["category_id"]
                )

st.write(f"**Your Score:** {st.session_state['score']} / {st.session_state['questions']}")
