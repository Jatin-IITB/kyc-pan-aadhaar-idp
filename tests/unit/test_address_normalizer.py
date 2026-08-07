from services.cross_doc.address_normalizer import IndianAddressNormalizer


def test_extract_pincode():
    n = IndianAddressNormalizer()
    result = n.normalize("42 MG Road, Bengaluru, Karnataka 560001")
    assert result["pincode"] == "560001"


def test_extract_state():
    n = IndianAddressNormalizer()
    result = n.normalize("42 MG Road, Bengaluru, Karnataka 560001")
    assert result["state"].lower() == "karnataka"


def test_abbreviation_expansion():
    n = IndianAddressNormalizer()
    result = n.normalize("42 MG Rd, Blr, KA 560001")
    assert "road" in result["normalized"].lower()


def test_compare_same_address_different_format():
    n = IndianAddressNormalizer()
    a1 = n.normalize("42 MG Road, Bangalore, Karnataka 560001")
    a2 = n.normalize("42 MG Rd, Bengaluru, KA 560001")
    score = n.compare(a1, a2)
    assert score > 0.5


def test_compare_different_addresses():
    n = IndianAddressNormalizer()
    a1 = n.normalize("42 MG Road, Bangalore, Karnataka 560001")
    a2 = n.normalize("15 Park Street, Kolkata, West Bengal 700016")
    score = n.compare(a1, a2)
    assert score < 0.5


def test_empty_address():
    n = IndianAddressNormalizer()
    result = n.normalize("")
    assert result == {}
