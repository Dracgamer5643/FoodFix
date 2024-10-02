from flask import Flask, render_template

app = Flask(__name__)

@app.route('/dashboard')
def dashboard():
    page="dashboard"
    return render_template('dashboard.html', page=page)

@app.route('/homepage')
def userpage():
    page="userpage"
    return render_template('userpage.html', page=page)

if __name__ == '__main__':
    app.run(debug=True, port=8080)