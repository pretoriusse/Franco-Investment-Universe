import hashlib
import requests
from flask import Flask, render_template, redirect, url_for, flash, request, render_template_string, send_file, jsonify
from flask_migrate import Migrate
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from webapp.models import db, Subscribers, Subscriptions, HTMLWebView, SubscriberIDNumbers, PortfolioTracker, PortfolioTransactionHistory
from webapp.forms import RegistrationForm, LoginForm
from assets.config import Config
from datetime import datetime, timezone, timedelta
from sqlalchemy import asc, create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import logging
from assets.const import EMAIL_ADDRESS, SERVER_ADDRESS, SERVER_PORT, EMAIL_PASSWORD
from jinja2 import Environment, FileSystemLoader
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from flask_wtf.csrf import CSRFProtect
from flask_wtf.csrf import CSRFError

# Initializa logging:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI

# Initialize SQLAlchemy and set up the database
db.init_app(app)

# Initialize Flask-Migrate for handling database migrations
migrate = Migrate(app, db)

# Initialize the login manager for handling user sessions
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

csrf = CSRFProtect(app)

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return Subscribers.query.get(int(user_id))

# Set up the engine and session
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
Session = scoped_session(sessionmaker(bind=engine))

# Helper function to generate PayFast signature
def generate_signature(data, passphrase):
    # Sort data by key
    sorted_data = {k: v for k, v in sorted(data.items())}
    
    # Concatenate data as query string
    signature_string = '&'.join([f'{key}={value}' for key, value in sorted_data.items()])
    
    # Append passphrase if present
    if passphrase:
        signature_string += f'&passphrase={passphrase}'
    
    # Return MD5 hash of the signature string
    return hashlib.md5(signature_string.encode('utf-8')).hexdigest()

# Helper functions for subscription management
def get_headers():
    timestamp = datetime.now().isoformat()
    headers = {
        'merchant-id': Config.PAYFAST_MERCHANT_ID,
        'version': Config.PAYFAST_VERSION,
        'timestamp': timestamp,
    }
    # Generate signature with headers and passphrase
    headers['signature'] = generate_signature(headers, Config.PAYFAST_PASSPHRASE)
    return headers

def fetch_subscription_details(token):
    url = f"{Config.PAYFAST_API_URL}/subscriptions/{token}/fetch?testing=true"
    response = requests.get(url, headers=get_headers())

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch subscription details: {response.text}")
        return None

def manage_subscription_status(token, action):
    url = f"{Config.PAYFAST_API_URL}/subscriptions/{token}/{action}?testing=true"
    response = requests.put(url, headers=get_headers())

    return response.status_code == 200

def update_subscription(token, data):
    url = f"{Config.PAYFAST_API_URL}/subscriptions/{token}/update?testing=true"
    headers = get_headers()
    response = requests.patch(url, json=data, headers=headers)

    return response.status_code == 200

def update_card_details(token):
    # Redirect to PayFast’s card update URL
    url = f"https://www.payfast.co.za/eng/process?cmd=subscription-token&token={token}?testing=true"
    return redirect(url)

def send_disabled_email(subject, name, email_address):
    try:
        # Load the HTML template
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('illigal_email_template.html')

        # Prepare the email message
        message = MIMEMultipart()
        message['From'] = formataddr(("Stock Bot", EMAIL_ADDRESS))
        message['Subject'] = subject
        message['To'] =','.join([formataddr((name, email_address))])
        message['Cc'] = ','.join([
            formataddr(("Raine Pretorius", 'raine.pretorius1@gmail.com')),
            formataddr(("Franco Pretorius", 'francopret@gmail.com'))
        ])
        
    except Exception as ex:
        logger.error(f"Failed to send emails: {ex}")

def update_subscription_paid_status(user_id, end_date, status):
    """
    Function to update the subscription_paid status after the subscription end date.
    """
    # This function can be scheduled to run after the subscription end date.
    # It can also be integrated with the recurring payment checks or webhooks.
    with Session() as session:
        user = session.query(Subscribers).filter_by(id=user_id).first()
        if user and datetime.now(timezone.utc) >= end_date:
            user.subscription_paid = status
            session.commit()

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Subscriptions route
@app.route('/subscriptions')
def subscriptions():
    try:
        subs = Subscriptions.query.order_by(asc(Subscriptions.cost)).all()  # Query all subscriptions
        return render_template('subscriptions.html', subscriptions=subs)
    except Exception as e:
        flash(f"Error loading subscriptions: {str(e)}", 'danger')
        return redirect(url_for('home'))

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    subscription_paid = False

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')

        # Check if the ID number already exists in the SubscriberIDNumbers table
        existing_id_number = SubscriberIDNumbers.query.filter_by(id_number=form.id_number.data).first()
        
        if existing_id_number:
            flash("ID number already used for registration. Free trial is not allowed again.", 'danger')
            return render_template('register.html', form=form)

        # Use Session to get the Subscription instance
        with Session() as session:
            subscription = session.get(Subscriptions, form.subscription.data)

        # Check if the subscription instance exists
        if not subscription:
            flash("Selected subscription not found.", 'danger')
            return render_template('register.html', form=form)

        # Set the expiration date for free and paid users
        expiration_date = datetime.now(timezone.utc) + timedelta(days=30)
        if form.email.data in ['raine.pretorius1@gmail.com', 'francopret@gmail.com', 'rudieprettie@gmail.com', 'lorrein.pretorius@gmail.com', 'sileziap@gmail.com', 'minettevdh1@gmail.com', 'danellevdh1@gmail.com']:
            expiration_date = datetime.now(timezone.utc) + timedelta(days=365 * 100)
            subscription_paid = True

        # Create the new user
        new_user = Subscribers(
            email=form.email.data,
            name=form.name.data,
            subscription=subscription,  # Assign the subscription instance, not the ID
            subscription_date=datetime.now(timezone.utc),  # Use timezone-aware datetime
            subscription_expiration_date=expiration_date,
            password=hashed_password,
            subscription_paid=subscription_paid,
            id_number=form.id_number.data
        )

        # Create the new ID number entry
        new_id_number = SubscriberIDNumbers(
            subscriber=new_user,  # Link the user to the ID number
            id_number=form.id_number.data
        )

        try:
            db.session.add(new_user)
            db.session.add(new_id_number)  # Add the ID number to the SubscriberIDNumbers table
            db.session.commit()

            flash('Your account has been created!', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f"Error registering user: {str(e)}", 'danger')
    
    return render_template('register.html', form=form)

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Fetch the user from the database
        user = Subscribers.query.filter_by(email=form.email.data).first()
        # Check if user exists and the password matches
        if user and check_password_hash(user.password, form.password.data):
            # Check if the subscription is paid
            if not user.subscription_paid:
                login_user(user)
                flash('Subscription not paid', 'info')
                return redirect(url_for('pay_subscription', subscription_id=user.subscription_id))
            
            # Log in the user
            login_user(user)
            flash('Login successful!', 'success')
            if int(user.subscription_id) == 2:
                return redirect(url_for('reports'))

            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html', form=form)

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

# Pay subscription route
@app.route('/pay_subscription/<int:subscription_id>', methods=['GET', 'POST'])
@login_required
def pay_subscription(subscription_id):
    try:
        subscription = Subscriptions.query.get_or_404(subscription_id)
        user = current_user

        # Create data for the PayFast form
        payfast_data = {
            'merchant_id': app.config['PAYFAST_MERCHANT_ID'],
            'merchant_key': app.config['PAYFAST_MERCHANT_KEY'],
            'return_url': app.config['PAYFAST_RETURN_URL'],
            'cancel_url': app.config['PAYFAST_CANCEL_URL'],
            'notify_url': app.config['PAYFAST_NOTIFY_URL'],
            'name_first': user.name,
            'email_address': user.email,
            'm_payment_id': f'subscription_{subscription_id}_{user.id}',  # Unique payment ID
            'amount': f"{int(subscription.cost)}",  # Amount in cents
            'item_name': subscription.name,
            'item_description': subscription.detail,
            'subscription_type': '2',  # Fixed setting for subscriptions
            'billing_date': datetime.now().strftime('%Y-%m-%d'),  # Example billing date
            'recurring_amount': f"{int(subscription.cost)}",  # Recurring amount in cents
            'subscription_notify_email': 'true',  # Enable email notifications
            'subscription_notify_webhook': 'true',  # Enable webhook notifications
            'subscription_notify_buyer': 'true',  # Notify the buyer as well
        }

        # Include passphrase if set in the configuration
        passphrase = app.config.get('PAYFAST_PASSPHRASE', '')
        
        # Generate the signature
        signature = generate_signature(payfast_data, passphrase)
        payfast_data['signature'] = signature

        # Log the complete payfast_data for troubleshooting
        print(f"PayFast Data: {payfast_data}")

        # Render the payment form with PayFast data
        return render_template('pay_subscription.html', payfast_data=payfast_data, payfast_url=app.config['PAYFAST_URL'])
    except Exception as e:
        flash(f"Error processing payment: {str(e)}", 'danger')
        return redirect(url_for('subscriptions'))

# Payment success route
@app.route('/payment/success')
def payment_success():
    flash('Payment successful!', 'success')
    return redirect(url_for('home'))

# Payment cancel route
@app.route('/payment/cancel')
def payment_cancel():
    flash('Payment was cancelled.', 'warning')
    return redirect(url_for('subscriptions'))

# Payment notification route (ITN Handler)
@app.route('/payment/ipn', methods=['POST'])
def payment_notify():
    try:
        data = request.form.to_dict()
        print(f'Payment notification received: {data}')
        
        # Extract relevant data
        user_email = data.get('email_address')
        payment_status = data.get('payment_status')
        amount_gross = data.get('amount_gross')

        # Use the scoped session to fetch the user
        with Session() as session:
            user_id = data.get('m_payment_id').split('_')[-1]
            user = session.query(Subscribers).filter_by(id=user_id).first()
            
            if not user:
                print(f"User with email {user_email} not found.")
                return 'Error: User not found', 404
            
            # Update user's subscription status if the payment is complete
            if payment_status == 'COMPLETE':
                user.subscription_paid = True
                user.token = data.get('token')
                user.subscription_date = datetime.strptime(data.get('billing_date'), '%Y-%m-%d').date()
                session.commit()
                print(f"Updated subscription status for {user.email}.")
            else:
                print(f"Payment status is {payment_status}, not updating subscription.")

        return 'OK'
    except Exception as e:
        print(f"Error handling payment notification: {str(e)}")
        return 'Error', 400

# Manage subscription route
@app.route('/manage_subscription', methods=['GET', 'POST'])
@login_required
def manage_subscription():
    try:
        # Fetch subscription details using the stored token
        token = current_user.token  # Assuming the token is stored in the user model
        subscription_details = fetch_subscription_details(token)
        
        if not subscription_details:
            flash('Failed to load subscription details.', 'danger')
            return redirect(url_for('home'))

        if request.method == 'POST':
            action = request.form.get('action')

            if action == 'update_card':
                return update_card_details(token)
            elif action == 'pause':
                result = manage_subscription_status(token, 'pause')
                if result:
                    end_date = datetime.strptime(subscription_details['data']['response']['run_date'], '%Y-%m-%dT%H:%M:%S%z')
                    # Logic to set subscription_paid = False after end_date
                    update_subscription_paid_status(current_user.id, end_date, False)
                    flash('Subscription paused successfully. Your subscription will be inactive after the current period ends.', 'success')
                else:
                    flash('Failed to pause subscription.', 'danger')
            elif action == 'unpause':
                result = manage_subscription_status(token, 'unpause')
                if result:
                    flash('Subscription unpaused successfully.', 'success')
                else:
                    flash('Failed to unpause subscription.', 'danger')
            elif action == 'cancel':
                result = manage_subscription_status(token, 'cancel')
                if result:
                    flash('Subscription canceled successfully.', 'success')
                    current_user.subscription_paid = False
                    db.session.commit()
                else:
                    flash('Failed to cancel subscription.', 'danger')
            
            return redirect(url_for('manage_subscription'))

        return render_template('manage_subscription.html', subscription=subscription_details)
    except Exception as e:
        flash(f"Error managing subscription: {str(e)}", 'danger')
        return redirect(url_for('home'))

@app.route('/reports', methods=['GET', 'POST'])
@login_required
@csrf.exempt
def reports():
    try:
        # Start with all reports
        reports_query = HTMLWebView.query.order_by(HTMLWebView.display_date.desc())

        # If a POST request with filters is submitted
        if request.method == 'POST':
            # Filter by date if provided
            date_filter = request.form.get('date_filter')
            if date_filter:
                reports_query = reports_query.filter(HTMLWebView.display_date == date_filter)
            
            # Filter by report type if provided
            report_type_filter = request.form.get('report_type')
            if report_type_filter:
                reports_query = reports_query.filter(HTMLWebView.report_type.ilike(report_type_filter))

        # Execute the query to get the filtered reports
        reports = reports_query.all()

        return render_template('reports.html', reports=reports)
    except Exception as e:
        flash(f"Error loading reports: {str(e)}", 'danger')
        return redirect(url_for('home'))

@app.route('/show_report/<int:report_id>/<string:report_type>')
@login_required
def show_report(report_id, report_type):
    # Fetch the report from the database
    report = HTMLWebView.query.get_or_404(report_id)

    # Determine which HTML path to use
    if report_type == 'summary':
        html_path = report.html_summary_path
    else:
        html_path = report.html_detailed_path
    
    # Read the content of the HTML file
    try:
        with open(html_path, 'r') as file:
            content = file.read()
        return render_template_string(content)  # Dynamically render HTML content
    except FileNotFoundError:
        flash(f"Report file not found!", 'danger')
        return redirect(url_for('reports'))
    
@app.route('/download_report/<int:report_id>/<string:report_type>')
@login_required
def download_report(report_id, report_type):
    # Fetch the report from the database
    report = HTMLWebView.query.get_or_404(report_id)

    # Determine which PDF path to use
    if report_type == 'summary':
        file_path = report.pdf_summary_path
    else:
        file_path = report.pdf_detailed_path

    # Send the file for download
    try:
        return send_file(file_path, as_attachment=True)
    except FileNotFoundError:
        flash(f"Report file not found!", 'danger')
        return redirect(url_for('reports'))
    
@app.route('/track/<string:email_hash>', methods=['GET'])
def track_email(email_hash):
    try:
        # Lookup the email_hash in the database
        subscriber = Subscribers.query.filter_by(email_hash=email_hash).first()

        if subscriber:
            # Log that the email was opened
            subscriber.email_opened_count += 1
            db.session.commit()
            return jsonify({"message": "Email tracked successfully"}), 200
        else:
            return jsonify({"error": "User not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/track-web/<string:web_hash>', methods=['GET'])
def track_web(web_hash):
    try:
        # Lookup the email_hash in the database
        subscriber = Subscribers.query.filter_by(web_hash=web_hash).first()

        if subscriber:
            # Log that the email was opened
            subscriber.web_opened_count += 1
            db.session.commit()
            return jsonify({"message": "Web tracked successfully"}), 200
        else:
            return jsonify({"error": "User not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/disable-user/<int:user_id>', methods=['POST'])
@csrf.exempt
def disable_illegal_user(user_id):
    try:
        # Lookup the email_hash in the database
        subscriber:Subscribers = Subscribers.query.filter_by(id=user_id).first()

        if subscriber and not subscriber.email in ['raine.pretorius1@gmail.com', 'raine.pretorius9@gmail.com', 'rudieprettie@gmail.com', 'francopret@gmail.com']:
            # Log that the email was opened
            subscriber.black_listed = True
            db.session.commit()
            return jsonify({"message": "User has been blacklisted!"}), 200
        else:
            return jsonify({"error": "User not found"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
@app.route('/portfolio', methods=['GET'])
@login_required
def portfolio():
    # Query the portfolio tracker and transaction history
    portfolio = PortfolioTracker.query.all()
    transactions = PortfolioTransactionHistory.query.order_by(PortfolioTransactionHistory.date.desc()).all()

    # Render the HTML template with the queried data
    return render_template('portfolio.html', portfolio=portfolio, transactions=transactions)

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5003)
