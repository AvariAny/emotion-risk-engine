"""
Simple API entry point.
"""

from app.engine.analyzer import analyze_text


def main():
    text = input("Enter a message: ")

    score = analyze_text(text)

    print(f"\nRisk score: {score}")


if __name__ == "__main__":
    main()