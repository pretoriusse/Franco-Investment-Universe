import csv
import hashlib
import io
import secrets
from functools import wraps
import requests
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    flash,
    request,
    send_file,
    jsonify,
    abort,
    Response,
)
from flask_migrate import Migrate
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from webapp.models import (
    db,
    Base,
    Subscribers,
    Subscriptions,
    SubscriptionFunctions,
    HTMLWebView,
    SubscriberIDNumbers,
    PortfolioTracker,
    PortfolioTransactionHistory,
)
from webapp.forms import RegistrationForm, LoginForm
from assets.config import Config
from datetime import datetime, timezone, timedelta
from sqlalchemy import asc, create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import logging
from assets.const import (
    EMAIL_ADDRESS,
    SERVER_ADDRESS,
    SERVER_PORT,
    EMAIL_PASSWORD,
    DB_PARAMS_WEBAPP,
)
import assets.pg_ratelimit  # noqa: F401  registers the postgresql:// rate-limit storage scheme
from jinja2 import Environment, FileSystemLoader
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

# Initialize logging:
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)
app.config["SQLALCHEMY_DATABASE_URI"] = Config.SQLALCHEMY_DATABASE_URI

# Initialize SQLAlchemy and set up the database
db.init_app(app)

# Attach the legacy ``Model.query`` accessor to every model. The models use a
# custom typed DeclarativeBase, for which Flask-SQLAlchemy does not reliably
# add ``.query`` — without this, ``Subscriptions.query`` etc. raise
# AttributeError. Bind it to FSA's scoped session so all existing queries work.
Base.query = db.session.query_property()

# Initialize Flask-Migrate for handling database migrations
migrate = Migrate(app, db)

# Initialize the login manager for handling user sessions
login_manager = LoginManager(app)
login_manager.login_view = "login"

with app.app_context():
    db.create_all()

# Load the backlog of report files into htmlwebview now and daily at 07:00.
# Each gunicorn worker starts its own thread; the sync is idempotent so the
# duplication is harmless (losers of the insert race just roll back).
from webapp.report_sync import start_report_sync  # noqa: E402

start_report_sync()

csrf = CSRFProtect(app)

# Rate limiting. Generous global default to stop hammering without hurting
# normal browsing; tighter per-route limits are applied on sensitive routes.
# Counters are shared across all gunicorn workers via PostgreSQL (custom
# ``postgresql://`` backend in assets/pg_ratelimit.py, imported at top to
# register the scheme). The target database and table are created on startup if
# missing. Override with RATELIMIT_STORAGE_URI.
_default_ratelimit_uri = (
    f"postgresql://{DB_PARAMS_WEBAPP['user']}:{DB_PARAMS_WEBAPP['password']}"
    f"@{DB_PARAMS_WEBAPP['host']}:{DB_PARAMS_WEBAPP['port']}/"
    f"{os.getenv('RATELIMIT_DB', 'ratelimits')}"
)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["600 per hour", "120 per minute"],
    storage_uri=os.getenv("RATELIMIT_STORAGE_URI", _default_ratelimit_uri),
)


@limiter.request_filter
def _exempt_static():
    # Don't count static asset requests toward rate limits — an asset-heavy
    # page load would otherwise trip the limit for legitimate users.
    return request.endpoint == "static"


@app.after_request
def set_security_headers(response):
    """Baseline hardening headers on every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"  # clickjacking
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # Conservative CSP. 'unsafe-inline' is required by the inline <style>/<script>
    # in base.html; self + the CDNs actually used are allowed, everything else blocked.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "frame-ancestors 'none'"
    )
    return response


@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template(
        "error.html",
        message="Too many requests — please slow down and try again shortly.",
    ), 429


@app.errorhandler(403)
def forbidden_handler(e):
    return render_template(
        "error.html", message="You don't have permission to access that page."
    ), 403


@app.errorhandler(404)
def not_found_handler(e):
    return render_template("error.html", message="Page not found."), 404


# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Subscribers, int(user_id))


# Set up the engine and session
engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
Session = scoped_session(sessionmaker(bind=engine))


# Helper function to generate PayFast signature
def generate_signature(data, passphrase):
    # Sort data by key
    sorted_data = {k: v for k, v in sorted(data.items())}

    # Concatenate data as query string
    signature_string = "&".join(
        [f"{key}={value}" for key, value in sorted_data.items()]
    )

    # Append passphrase if present
    if passphrase:
        signature_string += f"&passphrase={passphrase}"

    # Return MD5 hash of the signature string
    return hashlib.md5(signature_string.encode("utf-8")).hexdigest()


# Helper functions for subscription management
def get_headers():
    timestamp = datetime.now().isoformat()
    headers = {
        "merchant-id": Config.PAYFAST_MERCHANT_ID,
        "version": Config.PAYFAST_VERSION,
        "timestamp": timestamp,
    }
    # Generate signature with headers and passphrase
    headers["signature"] = generate_signature(headers, Config.PAYFAST_PASSPHRASE)
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
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("illegal_email_template.html")
        html_body = template.render(name=name)

        message = MIMEMultipart("alternative")
        message["From"] = formataddr(("Stock Bot", EMAIL_ADDRESS))
        message["Subject"] = subject
        message["To"] = formataddr((name, email_address))
        cc_addresses = [
            formataddr(("Raine Pretorius", "raine.pretorius1@gmail.com")),
            formataddr(("Franco Pretorius", "francopret@gmail.com")),
        ]
        message["Cc"] = ", ".join(cc_addresses)
        message.attach(MIMEText(html_body, "html"))

        all_recipients = [
            email_address,
            "raine.pretorius1@gmail.com",
            "francopret@gmail.com",
        ]
        with smtplib.SMTP(SERVER_ADDRESS, SERVER_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, all_recipients, message.as_string())
        logger.info(f"Disabled-user notification sent to {email_address}")
    except Exception as ex:
        logger.error(f"Failed to send disabled-user email: {ex}")


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


def requires_feature(feature):
    """Gate a route on a subscription-tier feature flag (SubscriptionFunctions).

    Admins always pass. A user passes if any of their tier's function rows has
    the named flag set True. Otherwise they're bounced to the subscriptions page.
    """

    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if not current_user.is_admin:
                subscription = current_user.subscription
                functions = subscription.functions if subscription else []
                if not any(getattr(f, feature, False) for f in functions):
                    flash(
                        "Your subscription tier does not include this feature.",
                        "warning",
                    )
                    return redirect(url_for("subscriptions"))
            return view(*args, **kwargs)

        return wrapped

    return decorator


def admin_required(view):
    """Restrict a route to admins (401/redirect otherwise)."""

    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)

    return wrapped


# Home route
@app.route("/")
def home():
    return render_template("home.html")


# Subscriptions route
@app.route("/subscriptions")
def subscriptions():
    try:
        subs = Subscriptions.query.order_by(
            asc(Subscriptions.cost)
        ).all()  # Query all subscriptions
        return render_template("subscriptions.html", subscriptions=subs)
    except Exception as e:
        flash(f"Error loading subscriptions: {str(e)}", "danger")
        return redirect(url_for("home"))


# Register route
@app.route("/register", methods=["GET", "POST"])
@limiter.limit("5 per minute; 20 per hour", methods=["POST"])
def register():
    form = RegistrationForm()
    subscription_paid = False

    if form.validate_on_submit():
        hashed_password = generate_password_hash(
            form.password.data, method="pbkdf2:sha256"
        )

        # Check if the ID number already exists in the SubscriberIDNumbers table
        existing_id_number = SubscriberIDNumbers.query.filter_by(
            id_number=form.id_number.data
        ).first()

        if existing_id_number:
            flash(
                "ID number already used for registration. Free trial is not allowed again.",
                "danger",
            )
            return render_template("register.html", form=form)

        # Use Session to get the Subscription instance
        with Session() as session:
            subscription = session.get(Subscriptions, form.subscription.data)

        # Check if the subscription instance exists
        if not subscription:
            flash("Selected subscription not found.", "danger")
            return render_template("register.html", form=form)

        # Set the expiration date for free and paid users
        expiration_date = datetime.now(timezone.utc) + timedelta(days=30)
        if form.email.data in [
            "raine.pretorius1@gmail.com",
            "francopret@gmail.com",
            "rudieprettie@gmail.com",
            "lorrein.pretorius@gmail.com",
            "sileziap@gmail.com",
            "minettevdh1@gmail.com",
            "danellevdh1@gmail.com",
        ]:
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
            id_number=form.id_number.data,
        )

        # Create the new ID number entry
        new_id_number = SubscriberIDNumbers(
            subscriber=new_user,  # Link the user to the ID number
            id_number=form.id_number.data,
        )

        try:
            db.session.add(new_user)
            db.session.add(
                new_id_number
            )  # Add the ID number to the SubscriberIDNumbers table
            db.session.commit()

            flash("Your account has been created!", "success")
            return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            flash(f"Error registering user: {str(e)}", "danger")

    return render_template("register.html", form=form)


# Login route
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute; 50 per hour", methods=["POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Fetch the user from the database
        user = Subscribers.query.filter_by(email=form.email.data).first()
        # Check if user exists and the password matches
        if user and check_password_hash(user.password, form.password.data):
            # Blocked accounts cannot log in, regardless of a correct password.
            if user.black_listed:
                flash("This account has been disabled. Contact support.", "danger")
                return render_template("login.html", form=form)
            # Check if the subscription is paid
            if not user.subscription_paid:
                login_user(user)
                flash("Subscription not paid", "info")
                return redirect(
                    url_for("pay_subscription", subscription_id=user.subscription_id)
                )

            # Log in the user
            login_user(user)
            flash("Login successful!", "success")
            if user.subscription_id and int(user.subscription_id) == 2:
                return redirect(url_for("reports"))

            return redirect(url_for("home"))
        else:
            flash("Login failed. Check your email and password.", "danger")
    return render_template("login.html", form=form)


# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# Pay subscription route
@app.route("/pay_subscription/<int:subscription_id>", methods=["GET", "POST"])
@login_required
def pay_subscription(subscription_id):
    try:
        subscription = Subscriptions.query.get_or_404(subscription_id)
        user = current_user

        # Create data for the PayFast form
        payfast_data = {
            "merchant_id": app.config["PAYFAST_MERCHANT_ID"],
            "merchant_key": app.config["PAYFAST_MERCHANT_KEY"],
            "return_url": app.config["PAYFAST_RETURN_URL"],
            "cancel_url": app.config["PAYFAST_CANCEL_URL"],
            "notify_url": app.config["PAYFAST_NOTIFY_URL"],
            "name_first": user.name,
            "email_address": user.email,
            "m_payment_id": f"subscription_{subscription_id}_{user.id}",  # Unique payment ID
            "amount": f"{int(subscription.cost)}",  # Amount in cents
            "item_name": subscription.name,
            "item_description": subscription.detail,
            "subscription_type": "2",  # Fixed setting for subscriptions
            "billing_date": datetime.now().strftime("%Y-%m-%d"),  # Example billing date
            "recurring_amount": f"{int(subscription.cost)}",  # Recurring amount in cents
            "subscription_notify_email": "true",  # Enable email notifications
            "subscription_notify_webhook": "true",  # Enable webhook notifications
            "subscription_notify_buyer": "true",  # Notify the buyer as well
        }

        # Include passphrase if set in the configuration
        passphrase = app.config.get("PAYFAST_PASSPHRASE", "")

        # Generate the signature
        signature = generate_signature(payfast_data, passphrase)
        payfast_data["signature"] = signature

        # Log the complete payfast_data for troubleshooting
        print(f"PayFast Data: {payfast_data}")

        # Render the payment form with PayFast data
        return render_template(
            "pay_subscription.html",
            payfast_data=payfast_data,
            payfast_url=app.config["PAYFAST_URL"],
        )
    except Exception as e:
        flash(f"Error processing payment: {str(e)}", "danger")
        return redirect(url_for("subscriptions"))


# Payment success route
@app.route("/payment/success")
def payment_success():
    flash("Payment successful!", "success")
    return redirect(url_for("home"))


# Payment cancel route
@app.route("/payment/cancel")
def payment_cancel():
    flash("Payment was cancelled.", "warning")
    return redirect(url_for("subscriptions"))


# Payment notification route (ITN Handler)
# Server-to-server POST from PayFast: exempt from CSRF (no browser token) and
# from rate limiting; forgery is prevented by the signature check below.
@app.route("/payment/ipn", methods=["POST"])
@csrf.exempt
@limiter.exempt
def payment_notify():
    try:
        data = request.form.to_dict()
        logger.info("Payment notification received")

        # Verify PayFast IPN signature to prevent forged notifications
        received_signature = data.pop("signature", None)
        expected_signature = generate_signature(data, Config.PAYFAST_PASSPHRASE)
        if received_signature != expected_signature:
            logger.warning("PayFast IPN signature mismatch — request rejected")
            return "Invalid signature", 400

        payment_status = data.get("payment_status")
        user_email = data.get("email_address")

        m_payment_id = data.get("m_payment_id")
        if not m_payment_id:
            logger.error("IPN missing m_payment_id field")
            return "Bad request", 400
        user_id = m_payment_id.split("_")[-1]

        with Session() as session:
            user = session.query(Subscribers).filter_by(id=user_id).first()

            if not user:
                logger.error(f"IPN: user id={user_id} (email={user_email}) not found")
                return "Error: User not found", 404

            if payment_status == "COMPLETE":
                user.subscription_paid = True
                user.token = data.get("token")
                billing_date_str = data.get("billing_date")
                if billing_date_str:
                    user.subscription_date = datetime.strptime(
                        billing_date_str, "%Y-%m-%d"
                    ).date()
                session.commit()
                logger.info(f"Subscription activated for {user.email}")
            else:
                logger.info(f"IPN payment_status={payment_status!r}, no action taken")

        return "OK"
    except Exception as e:
        logger.error(f"Error handling payment notification: {e}")
        return "Error", 400


# Manage subscription route
@app.route("/manage_subscription", methods=["GET", "POST"])
@login_required
def manage_subscription():
    try:
        # Fetch subscription details using the stored token
        token = current_user.token  # Assuming the token is stored in the user model
        subscription_details = fetch_subscription_details(token)

        if not subscription_details:
            flash("Failed to load subscription details.", "danger")
            return redirect(url_for("home"))

        if request.method == "POST":
            action = request.form.get("action")

            if action == "update_card":
                return update_card_details(token)
            elif action == "pause":
                result = manage_subscription_status(token, "pause")
                if result:
                    end_date = datetime.strptime(
                        subscription_details["data"]["response"]["run_date"],
                        "%Y-%m-%dT%H:%M:%S%z",
                    )
                    # Logic to set subscription_paid = False after end_date
                    update_subscription_paid_status(current_user.id, end_date, False)
                    flash(
                        "Subscription paused successfully. Your subscription will be inactive after the current period ends.",
                        "success",
                    )
                else:
                    flash("Failed to pause subscription.", "danger")
            elif action == "unpause":
                result = manage_subscription_status(token, "unpause")
                if result:
                    flash("Subscription unpaused successfully.", "success")
                else:
                    flash("Failed to unpause subscription.", "danger")
            elif action == "cancel":
                result = manage_subscription_status(token, "cancel")
                if result:
                    flash("Subscription canceled successfully.", "success")
                    current_user.subscription_paid = False
                    db.session.commit()
                else:
                    flash("Failed to cancel subscription.", "danger")

            return redirect(url_for("manage_subscription"))

        return render_template(
            "manage_subscription.html", subscription=subscription_details
        )
    except Exception as e:
        flash(f"Error managing subscription: {str(e)}", "danger")
        return redirect(url_for("home"))


@app.route("/reports", methods=["GET", "POST"])
@login_required
@csrf.exempt
def reports():
    try:
        # Start with all reports
        reports_query = HTMLWebView.query.order_by(HTMLWebView.display_date.desc())

        selected_year = ""
        selected_month = ""

        # If a POST request with filters is submitted
        if request.method == "POST":
            # Filter by date if provided
            date_filter = request.form.get("date_filter")
            if date_filter:
                reports_query = reports_query.filter(
                    HTMLWebView.display_date == date_filter
                )

            # Filter by year / year+month (display_date is 'YYYY-MM-DD')
            selected_year = request.form.get("year_filter", "")
            selected_month = request.form.get("month_filter", "")
            if selected_year and selected_month:
                reports_query = reports_query.filter(
                    HTMLWebView.display_date.like(f"{selected_year}-{selected_month}-%")
                )
            elif selected_year:
                reports_query = reports_query.filter(
                    HTMLWebView.display_date.like(f"{selected_year}-%")
                )
            elif selected_month:
                reports_query = reports_query.filter(
                    HTMLWebView.display_date.like(f"%-{selected_month}-%")
                )

            # Filter by report type if provided
            report_type_filter = request.form.get("report_type")
            if report_type_filter:
                reports_query = reports_query.filter(
                    HTMLWebView.report_type.ilike(report_type_filter)
                )

        # Execute the query to get the filtered reports
        reports = reports_query.all()

        # Years that actually have reports, for the filter dropdown
        years = sorted(
            {d[0][:4] for d in db.session.query(HTMLWebView.display_date).distinct()},
            reverse=True,
        )

        return render_template(
            "reports.html",
            reports=reports,
            years=years,
            selected_year=selected_year,
            selected_month=selected_month,
        )
    except Exception as e:
        flash(f"Error loading reports: {str(e)}", "danger")
        return redirect(url_for("home"))


@app.route("/show_report/<int:report_id>/<string:report_type>")
@login_required
def show_report(report_id, report_type):
    # Fetch the report from the database
    report = HTMLWebView.query.get_or_404(report_id)

    # Determine which HTML path to use
    if report_type == "summary":
        html_path = report.html_summary_path
    else:
        html_path = report.html_detailed_path

    try:
        return send_file(html_path, mimetype="text/html")
    except FileNotFoundError:
        flash("Report file not found!", "danger")
        return redirect(url_for("reports"))


@app.route("/download_report/<int:report_id>/<string:report_type>")
@login_required
def download_report(report_id, report_type):
    # Fetch the report from the database
    report = HTMLWebView.query.get_or_404(report_id)

    # Determine which PDF path to use
    if report_type == "summary":
        file_path = report.pdf_summary_path
    else:
        file_path = report.pdf_detailed_path

    # Send the file for download
    try:
        return send_file(file_path, as_attachment=True)
    except FileNotFoundError:
        flash(f"Report file not found!", "danger")  # noqa: F541
        return redirect(url_for("reports"))


@app.route("/track/<string:email_hash>", methods=["GET"])
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


@app.route("/track-web/<string:web_hash>", methods=["GET"])
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


@app.route("/disable-user/<int:user_id>", methods=["POST"])
@login_required
def disable_illegal_user(user_id):
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden"}), 403
    try:
        subscriber: Subscribers = Subscribers.query.filter_by(id=user_id).first()
        if not subscriber:
            return jsonify({"error": "User not found"}), 404
        if subscriber.is_admin:
            return jsonify({"error": "Cannot blacklist an admin"}), 403
        subscriber.black_listed = True
        db.session.commit()
        logger.info(f"Admin {current_user.email} blacklisted user {subscriber.email}")
        return jsonify({"message": "User has been blacklisted!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/portfolio", methods=["GET"])
@requires_feature("portfolio")
def portfolio():
    # This user's holdings joined with today's buy/sell verdict
    from assets.database_queries import get_portfolio_with_signals

    holdings = get_portfolio_with_signals(current_user.id)
    transactions = (
        PortfolioTransactionHistory.query.filter_by(subscriber_id=current_user.id)
        .order_by(PortfolioTransactionHistory.date.desc())
        .all()
    )
    return render_template(
        "portfolio.html", portfolio=holdings, transactions=transactions
    )


@app.route("/portfolio/add", methods=["POST"])
@requires_feature("portfolio")
def portfolio_add():
    ticker = (request.form.get("ticker") or "").strip().upper()
    if not ticker:
        flash("Ticker is required.", "danger")
        return redirect(url_for("portfolio"))
    holding = PortfolioTracker(
        subscriber_id=current_user.id,
        ticker=ticker,
        weight=(request.form.get("weight") or "").strip(),
        comment=(request.form.get("comment") or "").strip(),
    )
    try:
        db.session.add(holding)
        db.session.commit()
        flash(f"{ticker} added to your portfolio.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Could not add holding: {e}", "danger")
    return redirect(url_for("portfolio"))


@app.route("/portfolio/delete/<int:holding_id>", methods=["POST"])
@requires_feature("portfolio")
def portfolio_delete(holding_id):
    holding = PortfolioTracker.query.get_or_404(holding_id)
    if holding.subscriber_id != current_user.id and not current_user.is_admin:
        abort(403)
    try:
        db.session.delete(holding)
        db.session.commit()
        flash("Holding removed.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Could not remove holding: {e}", "danger")
    return redirect(url_for("portfolio"))


# ---------------------------------------------------------------------------
# Admin panel — tier + user management. All routes require is_admin.
# ---------------------------------------------------------------------------


def _tier_functions(subscription):
    """Return the tier's SubscriptionFunctions row, creating one if absent."""
    functions = subscription.functions[0] if subscription.functions else None
    if functions is None:
        functions = SubscriptionFunctions(subscription_id=subscription.id)
        db.session.add(functions)
        db.session.commit()
    return functions


@app.route("/admin")
@admin_required
def admin_dashboard():
    stats = {
        "users": Subscribers.query.count(),
        "paid": Subscribers.query.filter_by(subscription_paid=True).count(),
        "admins": Subscribers.query.filter_by(is_admin=True).count(),
        "tiers": Subscriptions.query.count(),
    }
    return render_template("admin/dashboard.html", stats=stats)


@app.route("/admin/tiers", methods=["GET"])
@admin_required
def admin_tiers():
    tiers = Subscriptions.query.order_by(asc(Subscriptions.cost)).all()
    # Ensure every tier has a functions row so the template can render flags.
    rows = [{"tier": t, "fn": _tier_functions(t)} for t in tiers]
    return render_template("admin/tiers.html", rows=rows)


@app.route("/admin/tiers/create", methods=["POST"])
@admin_required
def admin_tier_create():
    try:
        tier = Subscriptions(
            name=(request.form.get("name") or "").strip(),
            cost=float(request.form.get("cost") or 0),
            detail=(request.form.get("detail") or "").strip(),
        )
        if not tier.name:
            flash("Tier name is required.", "danger")
            return redirect(url_for("admin_tiers"))
        db.session.add(tier)
        db.session.commit()
        _tier_functions(tier)  # create its flags row
        flash(f'Tier "{tier.name}" created.', "success")
    except (ValueError, TypeError):
        db.session.rollback()
        flash("Cost must be a number.", "danger")
    return redirect(url_for("admin_tiers"))


@app.route("/admin/tiers/<int:tier_id>/update", methods=["POST"])
@admin_required
def admin_tier_update(tier_id):
    tier = Subscriptions.query.get_or_404(tier_id)
    try:
        tier.name = (request.form.get("name") or tier.name).strip()
        tier.cost = float(request.form.get("cost") or tier.cost)
        tier.detail = (request.form.get("detail") or "").strip()
        fn = _tier_functions(tier)
        # Checkboxes: present in form => True
        fn.company_research = bool(request.form.get("company_research"))
        fn.portfolio = bool(request.form.get("portfolio"))
        fn.api_access = bool(request.form.get("api_access"))
        db.session.commit()
        flash(f'Tier "{tier.name}" updated.', "success")
    except (ValueError, TypeError):
        db.session.rollback()
        flash("Cost must be a number.", "danger")
    return redirect(url_for("admin_tiers"))


@app.route("/admin/tiers/<int:tier_id>/delete", methods=["POST"])
@admin_required
def admin_tier_delete(tier_id):
    tier = Subscriptions.query.get_or_404(tier_id)
    if tier.subscribers:
        flash(
            f'Cannot delete "{tier.name}" — {len(tier.subscribers)} subscriber(s) still on it.',
            "danger",
        )
        return redirect(url_for("admin_tiers"))
    db.session.delete(tier)
    db.session.commit()
    flash("Tier deleted.", "success")
    return redirect(url_for("admin_tiers"))


@app.route("/admin/users", methods=["GET"])
@admin_required
def admin_users():
    users = Subscribers.query.order_by(Subscribers.id).all()
    tiers = Subscriptions.query.order_by(asc(Subscriptions.cost)).all()
    return render_template("admin/users.html", users=users, tiers=tiers)


@app.route("/admin/users/<int:user_id>/update", methods=["POST"])
@admin_required
def admin_user_update(user_id):
    user = Subscribers.query.get_or_404(user_id)
    action = request.form.get("action")

    if action == "toggle_paid":
        user.subscription_paid = not user.subscription_paid
    elif action == "toggle_admin":
        if user.id == current_user.id:
            flash("You can't change your own admin status.", "warning")
            return redirect(url_for("admin_users"))
        user.is_admin = not user.is_admin
    elif action == "toggle_blacklist":
        if user.is_admin:
            flash("Cannot blacklist an admin.", "danger")
            return redirect(url_for("admin_users"))
        user.black_listed = not bool(user.black_listed)
    elif action == "set_tier":
        tier_id = request.form.get("tier_id")
        user.subscription_id = int(tier_id) if tier_id else None
    elif action == "extend":
        days = int(request.form.get("days") or 0)
        base = max(user.subscription_expiration_date, datetime.now(timezone.utc).date())
        user.subscription_expiration_date = base + timedelta(days=days)
    else:
        flash("Unknown action.", "warning")
        return redirect(url_for("admin_users"))

    db.session.commit()
    flash("User updated.", "success")
    return redirect(url_for("admin_users"))


# ---------------------------------------------------------------------------
# Data API (CSV) — gated on the `api_access` subscription-tier flag.
# Keys are shown once on generation; only their sha256 is stored.
# ---------------------------------------------------------------------------


def _hash_key(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _authed_api_subscriber():
    """Return the subscriber for a valid X-API-Key with API access, else None."""
    raw = request.headers.get("X-API-Key", "")
    if not raw:
        return None
    subscriber = Subscribers.query.filter_by(api_key_hash=_hash_key(raw)).first()
    if not subscriber or not subscriber.is_active or not subscriber.subscription_paid:
        return None
    subscription = subscriber.subscription
    functions = subscription.functions if subscription else []
    if not (
        subscriber.is_admin or any(getattr(f, "api_access", False) for f in functions)
    ):
        return None
    return subscriber


@app.route("/api", methods=["GET"])
@login_required
def api_portal():
    return render_template("api.html")


@app.route("/api/key", methods=["POST"])
@login_required
@limiter.limit("5 per hour")
def api_generate_key():
    raw = secrets.token_urlsafe(32)
    current_user.api_key_hash = _hash_key(raw)
    db.session.commit()
    flash(f"Your new API key (copy it now — it is not shown again): {raw}", "success")
    return redirect(url_for("api_portal"))


@app.route("/api/v1/signals.csv", methods=["GET"])
@csrf.exempt
@limiter.limit(
    "120 per hour",
    key_func=lambda: request.headers.get("X-API-Key", get_remote_address()),
)
def api_signals_csv():
    if _authed_api_subscriber() is None:
        return jsonify({"error": "Invalid or unauthorized API key"}), 401

    from assets.database_queries import get_all_latest_signals

    rows = get_all_latest_signals()
    columns = [
        "ticker",
        "share_name",
        "run_date",
        "current_price",
        "next_week_prediction",
        "next_month_prediction",
        "z_score",
        "ma24",
        "ma55",
        "verdict",
        "reason",
    ]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=signals.csv"},
    )


# Main entry point
if __name__ == "__main__":
    debug = app.config.get("DEBUG", False)
    app.run(host="0.0.0.0", debug=debug, port=5003)
