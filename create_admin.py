"""Create or promote an admin user for the web app.

Run on the server (where the webapp DB is reachable):

    python create_admin.py --email you@example.com --name "Your Name"

You'll be prompted for a password (hidden). If the email already exists the
user is promoted to admin and, if you enter a password, it is reset. There is
no "default" password — passwords are stored only as one-way hashes.
"""

import argparse
import getpass
from datetime import date, timedelta

from werkzeug.security import generate_password_hash

# Importing app configures Flask, the DB, and the Base.query accessor.
from app import app, db
from webapp.models import Subscribers


def main():
    parser = argparse.ArgumentParser(description="Create/promote a web-app admin.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="Administrator")
    parser.add_argument(
        "--id-number",
        default="0000000000000",
        help="ID number (required by schema; default is a placeholder).",
    )
    args = parser.parse_args()

    password = getpass.getpass("New password: ")
    if password and password != getpass.getpass("Confirm password: "):
        raise SystemExit("Passwords do not match.")

    with app.app_context():
        user = Subscribers.query.filter_by(email=args.email).first()
        if user:
            user.is_admin = True
            user.subscription_paid = True
            user.black_listed = False
            if password:
                user.password = generate_password_hash(password, method="pbkdf2:sha256")
            action = "promoted to admin"
        else:
            if not password:
                raise SystemExit("A password is required to create a new user.")
            user = Subscribers(
                email=args.email,
                name=args.name,
                id_number=args.id_number,
                password=generate_password_hash(password, method="pbkdf2:sha256"),
                subscription_date=date.today(),
                subscription_expiration_date=date.today() + timedelta(days=365 * 100),
                subscription_paid=True,
                is_admin=True,
            )
            db.session.add(user)
            action = "created as admin"

        db.session.commit()
        print(f"User {args.email} {action} (id={user.id}).")


if __name__ == "__main__":
    main()
