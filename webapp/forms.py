from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, EmailField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from .models import Subscriptions


class RegistrationForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired(), Email()])
    name = StringField("Name", validators=[DataRequired()])
    id_number = StringField("ID Number", validators=[DataRequired()])
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=8, message="Password must be at least 8 characters."),
        ],
    )
    confirm_password = PasswordField(
        "Confirm Password", validators=[DataRequired(), EqualTo("password")]
    )
    subscription = SelectField(
        "Subscription", choices=[], coerce=int, validators=[DataRequired()]
    )
    submit = SubmitField("Register")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subscription.choices = [(s.id, s.name) for s in Subscriptions.query.all()]


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")
