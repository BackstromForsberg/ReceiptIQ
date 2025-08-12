from flask import Flask

# Create a Flask app instance
app = Flask(__name__)

# Define a route
@app.route("/")
def hello_world():
    return "Hello, World!"

# Define a route for the favicon to remove 404 log
@app.route('/favicon.ico')
def favicon():
    return ('', 204)  # No Content

# Run the app if this file is executed directly
if __name__ == "__main__":
    app.run(debug=True)