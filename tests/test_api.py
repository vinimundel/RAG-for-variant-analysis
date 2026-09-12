from healthrag.api import health, profiles


def test_health_endpoint():
    assert health() == {"status": "ok", "version": "0.2.0", "mode": "local-research"}


def test_profiles_make_braf_an_optional_customization():
    assert [profile["name"] for profile in profiles()] == ["general", "braf_oncology"]
