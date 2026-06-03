"""Verify the Anthropic API key in .env is loaded and working.

Loads <root>/.env (via tigerform.config), confirms the key is present, then makes
one tiny Claude call and reports success or a clear error.

    python scripts/check_api.py
"""
import _bootstrap  # noqa: F401
import os

from tigerform import config  # importing this loads .env


def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("✗ ANTHROPIC_API_KEY not found.")
        print("  Make sure tigerform/.env contains: ANTHROPIC_API_KEY=sk-ant-...")
        return 1
    print(f"✓ Key loaded (…{key[-4:]}), length {len(key)}.")
    print(f"  Feedback model: {config.FEEDBACK_MODEL}")

    try:
        import anthropic
    except ImportError:
        print("✗ The 'anthropic' package isn't installed (pip install anthropic).")
        return 1

    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=config.FEEDBACK_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        print(f"✓ API call succeeded. Model replied: {text!r}")
        print("  Claude-phrased feedback will work.")
        return 0
    except Exception as e:
        name = type(e).__name__
        print(f"✗ API call failed: {name}: {e}")
        if "authentication" in name.lower() or "401" in str(e):
            print("  → The key is invalid or revoked. Generate a new one in the Anthropic console.")
        elif "not_found" in str(e).lower() or "404" in str(e):
            print(f"  → Your account may not have access to '{config.FEEDBACK_MODEL}'. "
                  "Set TIGERFORM_FEEDBACK_MODEL in .env to a model you can use.")
        elif "credit" in str(e).lower() or "billing" in str(e).lower():
            print("  → Check your account billing/credits.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
