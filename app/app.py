from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

app = Flask(__name__)

model = None  
accuracy = 0

@app.route('/')
def index():
    return render_template('train.html', accuracy=accuracy)

@app.route('/train', methods=['POST'])
def train_model():
    global model, accuracy

    uploaded_file = request.files['csv_file']
    if not uploaded_file:
        return redirect(url_for('index'))

    test_size = float(request.form['test_size']) 

    df = pd.read_csv(uploaded_file)

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)  # Gunakan nilai slider

    imputer = SimpleImputer(strategy='mean')
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']

    train_data[features] = imputer.fit_transform(train_data[features])
    test_data[features] = imputer.transform(test_data[features])

    model = GaussianNB()
    model.fit(train_data[features], train_data['Survived'])

    predictions = model.predict(test_data[features])

    accuracy = accuracy_score(test_data['Survived'], predictions)
    accuracy = "{:.2f}".format(accuracy * 100)

    return redirect(url_for('index'))


@app.route('/classify', methods=['GET', 'POST'])
def classify_passenger():
    if request.method == 'POST':
        pclass = int(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        sex = request.form['sex']

        sex_numeric = 0 if sex == 'male' else 1

        user_data = pd.DataFrame({'Pclass': [pclass], 'Age': [age], 'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare], 'Sex': [sex_numeric]})

        if model:
            prediction = model.predict(user_data)
            result = "Selamat" if prediction[0] == 1 else "Tidak Selamat" 
            return render_template('classify.html', result=result)
        else:
            return "Model belum dilatih. Silakan latih model terlebih dahulu."

    return render_template('classify.html')


if __name__ == '__main__':
    app.run(debug=True)
