from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_processing import preprocess_text

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    response = request.form['response']
    expected_answer = request.form['expected_answer']
    result = grade_response(response, expected_answer)
    return jsonify(result)


def grade_response(response, expected_answer):
    response = preprocess_text(response)
    expected_answer = preprocess_text(expected_answer)

    responses = [expected_answer, response]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(responses)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    score = similarity[0][0] * 100  # Score out of 100
    score = round(score, 2)  # Round the score to 2 decimal places
    feedback = "Your answer is {:.2f}% similar to the expected answer.".format(score)

    return {"score": score, "feedback": feedback}




if __name__ == '__main__':
    app.run(debug=True)
