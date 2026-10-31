from urllib.parse import urlsplit
import ipaddress
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


BRAND_WORDS = (
    "paypal",
    "microsoft",
    "google",
    "apple",
    "amazon",
    "facebook",
    "youtube",
    "netflix",
)

LOGIN_WORDS = ("login", "signin", "verify", "secure")


def parse_url(value):
    value = str(value).strip()

    try:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        scheme = parsed.scheme.lower()

        if scheme not in {"http", "https"} or not hostname:
            return None

        try:
            ipaddress.ip_address(hostname)
            is_ip = 1
        except ValueError:
            is_ip = 0

        return {
            "url": value,
            "hostname": hostname,
            "scheme": scheme,
            "path": parsed.path or "",
            "query": parsed.query or "",
            "fragment": parsed.fragment or "",
            "is_ip": is_ip,
        }

    except (ValueError, TypeError):
        return None


def hostname_features(url):
    parsed = parse_url(url)

    if parsed is None:
        return {
            "invalid_url": 1,
            "hostname_length": 0,
            "hostname_dot_count": 0,
            "subdomain_count": 0,
            "is_ip": 0,
            "uses_https": 0,
            "uses_http": 0,
            "path_length": 0,
            "query_length": 0,
            "url_length": 0,
            "hostname_hyphen_count": 0,
            "hostname_digit_count": 0,
            "hostname_underscore_count": 0,
            "hostname_has_punycode": 0,
            "hostname_has_brand_word": 0,
            "hostname_has_brand_as_subdomain": 0,
            "hostname_has_suspicious_brand_domain_pattern": 0,
            "host_has_login_word": 0,
            "url_has_at": 0,
            "url_has_encoded_chars": 0,
            "url_has_double_slash_path": 0,
            "url_has_password_word": 0,
            "url_has_redirect_word": 0,
            "url_has_ip_text": 0,
            "tld_length": 0,
        }

    host = parsed["hostname"]
    labels = host.split(".")
    path = parsed["path"]
    query = parsed["query"]
    url_lower = parsed["url"].lower()

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    # This is a deliberately simple heuristic, not a public-suffix-list parser.
    # It handles common two-part domains such as example.com and youtube.com.
    domain_labels = labels[-2:] if len(labels) >= 2 else labels
    subdomain_labels = labels[:-2] if len(labels) >= 3 else []

    brand_in_hostname = any(
        brand in host for brand in BRAND_WORDS
    )

    brand_as_subdomain = any(
        any(brand in label for brand in BRAND_WORDS)
        for label in subdomain_labels
    )

    # Flag hostnames containing a brand token outside the final two labels.
    # This is a review signal only; it is not a definitive phishing verdict.
    suspicious_brand_domain_pattern = int(
        brand_in_hostname and brand_as_subdomain
    )

    return {
        "invalid_url": 0,
        "hostname_length": len(host),
        "hostname_dot_count": host.count("."),
        "subdomain_count": max(0, len(labels) - 2),
        "is_ip": is_ip,
        "uses_https": int(parsed["scheme"] == "https"),
        "uses_http": int(parsed["scheme"] == "http"),
        "path_length": len(path),
        "query_length": len(query),
        "url_length": len(parsed["url"]),
        "hostname_hyphen_count": host.count("-"),
        "hostname_digit_count": sum(c.isdigit() for c in host),
        "hostname_underscore_count": host.count("_"),
        "hostname_has_punycode": int("xn--" in host),
        "hostname_has_brand_word": int(brand_in_hostname),
        "hostname_has_brand_as_subdomain": int(brand_as_subdomain),
        "hostname_has_suspicious_brand_domain_pattern": (
            suspicious_brand_domain_pattern
        ),
        "host_has_login_word": int(
            any(word in host for word in LOGIN_WORDS)
        ),
        "url_has_at": int("@" in parsed["url"]),
        "url_has_encoded_chars": int("%" in parsed["url"]),
        "url_has_double_slash_path": int("//" in path),
        "url_has_password_word": int(
            any(word in url_lower for word in ("password", "passwd"))
        ),
        "url_has_redirect_word": int(
            any(
                word in url_lower
                for word in ("redirect", "return=", "continue=")
            )
        ),
        "url_has_ip_text": int(
            bool(re.search(r"(?:\d{1,3}\.){3}\d{1,3}", host))
        ),
        "tld_length": len(labels[-1]) if labels else 0,
    }


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert URLs to numeric features for an experimental model."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if "URL" in X.columns:
                values = X["URL"].tolist()
            elif X.shape[1] == 1:
                values = X.iloc[:, 0].tolist()
            else:
                raise ValueError(
                    "URLFeatureExtractor expects a URL column "
                    "or a single-column DataFrame."
                )
        elif isinstance(X, pd.Series):
            values = X.tolist()
        else:
            values = np.asarray(X, dtype=object).reshape(-1).tolist()

        records = [hostname_features(value) for value in values]
        return pd.DataFrame(records).fillna(0).to_numpy(dtype=np.float64)
