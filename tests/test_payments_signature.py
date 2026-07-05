"""
Unit tests for assets.payments.generate_signature.

This signature guards real money movement (PayFast subscription charges), so
its ordering/encoding rules deserve a pinned regression test: any accidental
change to field order, URL-encoding, or passphrase handling would silently
break payment processing without these assertions.
"""

from assets.payments import generate_signature


def test_signature_is_order_independent_in_input_dict():
    ordered = {
        "merchant_id": "10000100",
        "merchant_key": "key",
        "amount": "100.00",
        "item_name": "Test Item",
    }
    shuffled = {
        "item_name": "Test Item",
        "amount": "100.00",
        "merchant_id": "10000100",
        "merchant_key": "key",
    }
    assert generate_signature(ordered) == generate_signature(shuffled)


def test_signature_known_value_without_passphrase():
    data = {
        "merchant_id": "10000100",
        "merchant_key": "key",
        "amount": "100.00",
        "item_name": "Test Item",
    }
    assert generate_signature(data) == "479bac5ea36001062eef34c092b6cb7b"


def test_signature_changes_with_passphrase():
    data = {
        "merchant_id": "10000100",
        "merchant_key": "key",
        "amount": "100.00",
        "item_name": "Test Item",
    }
    without = generate_signature(data)
    with_pass = generate_signature(data, passphrase="mypassphrase")
    assert with_pass == "ebfc7fe1b2cd24d3fcf8d7e7dc60062f"
    assert with_pass != without


def test_signature_ignores_keys_outside_the_payfast_field_order():
    data = {"merchant_id": "10000100", "item_name": "Test Item"}
    with_extra = {**data, "some_unrelated_field": "ignored"}
    assert generate_signature(data) == generate_signature(with_extra)


def test_signature_skips_empty_string_values():
    data = {"merchant_id": "10000100", "merchant_key": "", "item_name": "Test Item"}
    without_empty = {"merchant_id": "10000100", "item_name": "Test Item"}
    assert generate_signature(data) == generate_signature(without_empty)


def test_signature_url_encodes_spaces_as_plus():
    data = {"merchant_id": "10000100", "item_name": "Coffee & Cake"}
    assert generate_signature(data) == "7e3d7ce7d8666416a0a7e9e52f389654"
