import logging

logging.basicConfig(level=logging.DEBUG)

from flask import Flask, render_template, request
from flask_mail import Mail, Message

app = Flask(__name__)

@app.route('/')
def home():
    app.logger.debug("Rendering Home Page")
    return render_template('index.html')

@app.route('/about')
def about():
    app.logger.debug("Rendering About Page")
    return render_template('about.html')

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kcpalmisano@gmail.com'
app.config['MAIL_PASSWORD'] = 'bbox yvcf tgtp oauu'

mail = Mail(app)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Send an email
        msg = Message(f"New Message from {name}", 
                      sender=email, 
                      recipients=['your-email@gmail.com'])
        msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        mail.send(msg)

          # Return a confirmation page
        return f"""
        <html>
        <head>
            <title>Thank You</title>
            <link rel="stylesheet" href="/static/style.css">
        </head>
        <body>
            <h1>Thank You, {name}!</h1>
            <p>Your message has been sent successfully.</p>
            <a href="/" class="button">Return Home</a>
        </body>
        </html>
        """
    return render_template('contact.html')


@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
