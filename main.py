from flask import Flask , render_template , request 

app = Flask(__name__)

@app.route("/", methods = ["GET","POST"])
def result():
    if request.method == "POST":
        query = request.form["queryInput"]
        print(query)
    return render_template("index.html")


app.run(debug=True)