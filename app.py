from planet_b612_dashboard import app

# This is important for Render to find the Flask server
server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)