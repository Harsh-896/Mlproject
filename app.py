import pickle
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # DEBUG
            print("Form data:", request.form)
            form = request.form

            # accept multiple possible field names from the HTML form
            def get_field(*names):
                for n in names:
                    v = form.get(n)
                    if v is not None and v != "":
                        return v
                return None

            gender = get_field('gender', 'Gender')
            race_ethnicity = get_field('race_ethnicity', 'ethnicity', 'race/ethnicity')
            parental_level_of_education = get_field('parental_level_of_education', 'parental level of education')
            lunch = get_field('lunch', 'Lunch')
            test_preparation_course = get_field('test_preparation_course', 'test preparation course', 'test_preparation')
            reading_score = get_field('reading_score', 'reading score', 'reading')
            writing_score = get_field('writing_score', 'writing score', 'writing')

            required = {
                'gender': gender,
                'race_ethnicity': race_ethnicity,
                'parental_level_of_education': parental_level_of_education,
                'lunch': lunch,
                'test_preparation_course': test_preparation_course,
                'reading_score': reading_score,
                'writing_score': writing_score
            }
            missing = [k for k, v in required.items() if v in (None, "")]

            if missing:
                msg = f"Error: Missing fields: {missing}"
                print(msg)
                return render_template('home.html', results=msg)

            # convert numeric fields
            try:
                reading_val = float(reading_score)
                writing_val = float(writing_score)
            except ValueError as e:
                return render_template('home.html', results=f"Error: Invalid numeric input - {e}")

            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_val,
                writing_score=writing_val
            )
            df = data.get_data_as_data_frame()
            print("Input DataFrame for prediction:\n", df)

            pipeline = PredictPipeline()
            preds = pipeline.predict(df)  # should return list
            print("Predictions:", preds)

            if not preds:
                return render_template('home.html', results="Error: No prediction returned")
            return render_template('home.html', results=f"Predicted math score: {float(preds[0]):.2f}")
        except Exception as e:
            print("Prediction error:", e)
            return render_template('home.html', results=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)