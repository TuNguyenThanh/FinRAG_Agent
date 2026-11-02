from app.security.pii import mask_pii


def test_masks_vietnamese_identifiers():
    r = mask_pii("CCCD 012345678901, sđt 0912345678, email a.b@x.vn")
    assert r.text == "CCCD [CCCD], sđt [PHONE], email [EMAIL]"
    assert r.found == {"cccd": 1, "phone": 1, "email": 1}


def test_masks_valid_card_only():
    assert mask_pii("thẻ 4111 1111 1111 1111").text == "thẻ [CARD]"
    # fails Luhn -> not a card number, keep as-is
    assert "[CARD]" not in mask_pii("mã 1234 5678 9012 3456").text


def test_account_needs_cue_word_so_amounts_survive():
    assert mask_pii("STK: 19034567890 của tôi").text == "STK: [ACCOUNT] của tôi"
    assert mask_pii("số tài khoản 0071000123456").text == "số tài khoản [ACCOUNT]"
    r = mask_pii("chuyển 50000000 VND thì phí bao nhiêu")
    assert r.text == "chuyển 50000000 VND thì phí bao nhiêu"
    assert not r.has_pii


def test_hotline_not_masked():
    assert mask_pii("gọi 1900 1234").text == "gọi 1900 1234"
