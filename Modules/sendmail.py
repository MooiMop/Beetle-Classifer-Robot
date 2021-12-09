import smtplib, ssl
from getpass import getpass

smtp_server = "mail.physics.leidenuniv.nl"
port = 587  # For starttls
sender_email = input("Please type your physics email name and press enter:")
if '@physics' not in sender_email:
    sender_email += '@physics.leidenuniv.nl'
password = getpass(f"Type the password for {sender_email} and press enter: ")
receiver_email = "scheinowitz@physics.leidenuniv.nl"
message = """\
Subject: Hi there

This message is sent from Python."""

# Create a secure SSL context
context = ssl.create_default_context()

# Try to log in to server and send email
try:
    server = smtplib.SMTP(smtp_server,port)
    server.ehlo() # Can be omitted
    server.starttls(context=context) # Secure the connection
    server.ehlo() # Can be omitted
    server.login(sender_email, password)
    # TODO: Send email here
    server.sendmail(sender_email, receiver_email, message)
except Exception as e:
    # Print any error messages to stdout
    print(e)
finally:
    server.quit()
