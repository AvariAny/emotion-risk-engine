# Prompt Generation Guide

Version: 1.0

---

# 1. Purpose

This document defines the standards for generating new examples for the Emotion Risk Engine dataset using Large Language Models (LLMs).

Its objective is to ensure consistency, diversity, and high-quality annotations regardless of the AI model used.

Supported models include, but are not limited to:

- ChatGPT
- Claude
- Gemini
- DeepSeek
- Mistral
- Llama
- Other instruction-following language models

Every generated example must comply with the Dataset Annotation Guide.

If any generated example contradicts the annotation guide, the guide always takes precedence.

---

# 2. General Generation Principles

Every generated example must satisfy the following principles.

## Natural Language

Messages must sound like real people.

Never generate:

- essays
- poems
- motivational phrases
- philosophical reflections
- news articles

Generate realistic chat messages.

---

## Realistic Context

Messages should resemble conversations written in:

- WhatsApp
- Messenger
- Telegram
- Discord
- SMS
- Social media

---

## Language

Use natural Latin American Spanish.

Avoid extremely formal language.

Avoid repetitive vocabulary.

---

## Diversity

Every batch should contain different:

- ages
- occupations
- personalities
- family situations
- economic situations
- educational levels
- hobbies
- writing styles

---

## Sentence Length

Include a mixture of:

- very short messages

- medium messages

- long messages

---

## Writing Style

Occasionally include:

- emojis

- abbreviations

- missing punctuation

- lowercase writing

- minor spelling mistakes

Do not abuse these variations.

The dataset must remain readable.

---

# 3. Diversity Rules

Never generate the same sentence twice.

Avoid repeating:

- sentence openings

- emotional expressions

- contexts

- vocabulary

- professions

- locations

- hobbies

The goal is to maximize linguistic diversity.

---

# 4. Forbidden Patterns

Never generate:

Repeated templates.

Examples:

"I'm tired..."

"I'm tired..."

"I'm tired..."

Never generate obvious AI wording.

Examples:

"I have reached a point in my emotional journey..."

Avoid unnatural language.

---

# 5. Label Consistency

Every generated example must belong to exactly one label.

The generated sentence should not reasonably fit multiple labels.

If ambiguity exists, rewrite the sentence.

---

# 6. Generation Workflow

Step 1

Read Dataset_Annotation_Guide.md

↓

Step 2

Select one label.

↓

Step 3

Generate examples.

↓

Step 4

Review every sentence.

↓

Step 5

Remove duplicates.

↓

Step 6

Verify annotation consistency.

↓

Step 7

Export as CSV.

---

# 7. Prompt Template

Every generation prompt should contain:

Objective

Label definition

Inclusion criteria

Exclusion criteria

Writing style

Output format

Quality requirements

CSV format

Number of examples

Nothing else.

---

# 8. Multi-LLM Strategy

Whenever possible, use multiple language models.

Recommended workflow:

30% ChatGPT

30% Claude

20% Gemini

20% DeepSeek

The final dataset should combine examples from multiple models to maximize linguistic diversity.

---

# 9. Human Review

Every generated batch must be manually reviewed.

Review checklist:

✓ Correct label

✓ Natural language

✓ No duplicates

✓ No contradictory examples

✓ Correct grammar (except intentional mistakes)

✓ Diverse vocabulary

✓ Fits only one emotional level

---

# 10. Final Rule

Human judgment always overrides AI-generated content.

Language models assist dataset creation.

They do not define the annotation rules.