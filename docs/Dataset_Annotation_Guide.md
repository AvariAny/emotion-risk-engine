# Emotion Risk Dataset Annotation Guide

Version: 1.0

---

# 1. Purpose

The purpose of this document is to define the annotation rules used to build the Emotion Risk Engine dataset.

This guide establishes clear and consistent criteria for assigning emotional risk labels to Spanish-language text messages.

Its objective is to minimize ambiguity during annotation and ensure that identical messages receive the same label regardless of who performs the annotation.

The resulting dataset is intended for training Transformer-based Natural Language Processing models capable of estimating emotional risk from text.

---

# 2. Scope

This guide applies to every text sample included in the Emotion Risk Engine dataset.

The annotation rules are designed for short Spanish-language messages that resemble real conversations, including:

- Chat messages
- SMS messages
- Social media posts
- Private messages
- Forum posts
- Comments
- Personal notes

The dataset is not intended to diagnose mental disorders.

Instead, it estimates the level of emotional risk expressed in a written message.

Only the textual content should be considered during annotation.

No assumptions should be made about the author's age, gender, personality, medical history, or real-life circumstances.

Annotations must rely exclusively on the information explicitly expressed in the text.

---

# 3. Annotation Principles

The following principles must always be respected during annotation.

## Principle 1

Label only what is explicitly expressed.

Never infer information that is not present in the message.

---

## Principle 2

Do not assume context.

Each message must be evaluated independently.

Previous conversations are unknown.

Future messages are unknown.

---

## Principle 3

The highest emotional expression determines the label.

If multiple emotions appear in the same message, assign the label corresponding to the highest emotional risk level.

---

## Principle 4

Emotional intensity is more important than the topic.

Talking about death does not automatically imply emotional risk.

Talking about stress does not necessarily imply emotional suffering.

The annotation must focus on the emotional meaning of the sentence.

---

## Principle 5

When two labels seem possible, always assign the lower label unless clear evidence supports the higher one.

This rule minimizes false positives and improves annotation consistency.

---

# 4. Risk Scale Overview

The dataset defines four levels of emotional risk.

| Label | Description |
|--------|-------------|
| 0 | No emotional risk |
| 1 | Mild emotional distress |
| 2 | High emotional distress |
| 3 | Critical emotional risk |

Each level represents the emotional state expressed in the message rather than a clinical diagnosis.

The labels form an ordinal scale, meaning that each level represents a progressively higher degree of emotional risk.

---

# 5. Risk Level Definitions

The labels used in this dataset do not represent clinical diagnoses.

Instead, they estimate the level of emotional risk expressed in a single text message.

The assigned label must be based exclusively on the language contained in the message.

No external information about the author should be assumed.

The annotation reflects the emotional content of the text, not the mental health status of the individual.

---

# Label 0 — No Emotional Risk

## Definition

A message belongs to Label 0 when it does not express significant emotional distress.

The author appears emotionally stable or communicates everyday situations without evidence of psychological suffering.

The message may express positive emotions, neutral information, or minor daily frustrations that do not affect emotional well-being.

Label 0 represents normal day-to-day communication.

---

## Inclusion Criteria

A message should be assigned Label 0 if it includes one or more of the following:

- Daily activities.
- Work.
- School.
- University.
- Technology.
- Food.
- Shopping.
- Sports.
- Music.
- Movies.
- Travel.
- Pets.
- Family.
- Friends.
- Games.
- Hobbies.
- Positive emotions.
- Neutral conversations.
- Small inconveniences.
- Mild complaints without emotional suffering.

---

## Exclusion Criteria

A message must NOT receive Label 0 if it expresses:

- Persistent sadness.
- Emotional exhaustion.
- Anxiety.
- Hopelessness.
- Loneliness.
- Emotional emptiness.
- Guilt.
- Feeling trapped.
- Feeling like a burden.
- Desire to disappear.
- Suicidal thoughts.

---

## Decision Rule

If the message contains any meaningful emotional suffering, Label 0 should not be assigned.

---

# Label 1 — Mild Emotional Distress

## Definition

A message belongs to Label 1 when the author expresses mild emotional discomfort that is temporary, understandable, and compatible with normal daily functioning.

The individual experiences emotional distress but still maintains hope, motivation, or the expectation that the situation can improve.

The emotional discomfort is present but does not dominate the person's entire perspective.

---

## Typical Emotional States

- Stress.
- Frustration.
- Temporary sadness.
- Mild anxiety.
- Worry.
- Mental fatigue.
- Discouragement.
- Feeling overwhelmed.
- Temporary lack of motivation.

---

## Inclusion Criteria

Examples include:

- Work stress.
- School stress.
- Financial concerns.
- Relationship problems.
- Feeling tired.
- Having a bad day.
- Feeling nervous.
- Feeling emotionally exhausted after a difficult week.

---

## Exclusion Criteria

A message must NOT receive Label 1 if it expresses:

- Loss of purpose.
- Persistent hopelessness.
- Emotional emptiness.
- Belief that nothing will improve.
- Feeling like life has no meaning.
- Desire to disappear.
- Death wishes.
- Self-harm.
- Suicide.

---

## Decision Rule

The person is emotionally affected but still psychologically engaged with everyday life.

---

# Label 2 — High Emotional Distress

## Definition

A message belongs to Label 2 when the author expresses severe and persistent emotional suffering that significantly affects their emotional well-being.

The person may describe hopelessness, emotional emptiness, loss of purpose, deep sadness, or feeling like a burden.

However, the message does not contain explicit or implicit expressions of suicidal intent or a desire to die.

---

## Typical Emotional States

- Hopelessness.
- Emotional emptiness.
- Persistent sadness.
- Loss of meaning.
- Emotional exhaustion.
- Feeling disconnected.
- Feeling like a burden.
- Severe loneliness.
- Lack of motivation.
- Loss of interest in life.

---

## Inclusion Criteria

Examples include:

- "Nada mejora."
- "Ya no disfruto nada."
- "Me siento vacío."
- "No encuentro sentido a mi vida."
- "Siento que soy una carga."
- "Todo parece inútil."

---

## Exclusion Criteria

A message must NOT receive Label 2 if it expresses:

- Wanting to die.
- Wanting to disappear permanently.
- Suicide.
- Self-harm.
- Farewell messages.
- Suicide planning.
- Statements implying that death is the preferred solution.

---

## Decision Rule

The emotional suffering is severe, but the person does not express an intention to end their life.

---

# Label 3 — Critical Emotional Risk

## Definition

A message belongs to Label 3 when it expresses explicit or implicit suicidal ideation, desire to die, self-harm, or an immediate risk of self-directed violence.

This label represents the highest emotional risk category in the dataset.

---

## Typical Indicators

- Desire to die.
- Desire to stop living.
- Suicide.
- Self-harm.
- Saying goodbye.
- Feeling that others would be better off without them.
- Believing death is the only solution.
- Planning suicide.
- Giving away possessions before dying.
- Expressing that life should end immediately.

---

## Inclusion Criteria

The message expresses one or more of the following:

- Suicidal ideation.
- Wish to die.
- Intent to self-harm.
- Suicide planning.
- Indirect statements clearly implying death as a desired outcome.

---

## Exclusion Criteria

Messages expressing sadness, hopelessness, emotional pain, or depression WITHOUT suicidal content must be classified as Label 2.

---

## Decision Rule

The defining characteristic of Label 3 is the presence of suicidal intent or desire for death.

Without suicidal content, Label 3 must never be assigned.


# Dataset Annotation Guide
Emotion Risk Engine

---

# Phase 3 — Annotation Rules

This section defines the decision process used to assign labels,
handle ambiguous situations,
review dataset quality,
and generate future examples consistently.

---

# 1. Decision Tree

When annotating a sentence, answer these questions in order.

START

│

├── Does the message express suicidal intent,
│   desire to die,
│   self-harm,
│   or saying goodbye?
│
│        YES → Label 3
│        NO
│
├── Does the message express deep emotional suffering,
│   hopelessness,
│   feeling like a burden,
│   emotional emptiness,
│   or severe depression,
│   WITHOUT suicidal intent?
│
│        YES → Label 2
│        NO
│
├── Does the message express temporary stress,
│   sadness,
│   frustration,
│   anxiety,
│   or emotional discomfort,
│   while remaining functional?
│
│        YES → Label 1
│        NO
│
└── Label 0

---

# 2. Ambiguous Cases

Some messages may fit multiple labels.

Always choose the highest level that is clearly supported.

Examples

"I'm exhausted."

Label 1

----------------------------

"I feel empty every day."

Label 2

----------------------------

"I don't want to wake up tomorrow."

Label 3

----------------------------

"I'm stressed because of work."

Label 1

----------------------------

"I'm tired after work."

Label 0

Physical tiredness alone is not emotional distress.

----------------------------

"I miss my grandfather."

Label 1

Grief alone is not high emotional risk.

----------------------------

"I can't enjoy anything anymore."

Label 2

Persistent loss of interest.

----------------------------

"I wish everything would end."

Label 3

Indirect suicidal ideation.

---

# 3. Borderline Examples

These examples are intentionally difficult.

Example

"I'm having a terrible week."

Label 1

Reason

Temporary emotional discomfort.

----------------------------

"I've lost interest in everything."

Label 2

Reason

Persistent emotional deterioration.

----------------------------

"I feel like everyone would be happier without me."

Label 3

Reason

Strong suicidal indicator.

----------------------------

"I don't know what to do anymore."

Label 1

Reason

Stress.

Without hopelessness.

----------------------------

"I don't see a future for myself."

Label 2

Reason

Hopelessness.

No explicit suicide.

----------------------------

"I don't want to exist anymore."

Label 3

Reason

Indirect desire to die.

---

# 4. Common Annotation Errors

Never confuse:

Stress
with
Depression

Temporary sadness
with
Hopelessness

Being tired
with
Emotional exhaustion

Loneliness
with
Suicidal ideation

Anger
with
Emotional risk

Frustration
with
Desire to disappear

Missing someone
with
Wanting to die

---

# 5. Quality Control Checklist

Every generated example should satisfy all checks.

✓ Sounds like a real chat message

✓ Natural Spanish

✓ No robotic wording

✓ No copied phrases

✓ Different sentence length

✓ Different contexts

✓ Different vocabulary

✓ Correct label

✓ No contradictions

✓ Fits only one label

---

# 6. Rules for Future Dataset Generation

When expanding the dataset:

Never regenerate previous examples.

Use different:

- professions

- ages

- family situations

- school contexts

- jobs

- hobbies

- locations

- vocabulary

Include:

short messages

medium messages

long messages

questions

statements

messages with emojis (occasionally)

messages without punctuation

messages with spelling mistakes (occasionally)

Avoid repetitive sentence openings.

Bad

"Hoy..."

"Hoy..."

"Hoy..."

Good

"Últimamente..."

"No sé..."

"A veces..."

"Me cuesta..."

"Desde hace días..."

"Creo que..."

"Dormí mal otra vez..."

---

# 7. Annotation Priority

If a sentence matches multiple labels,
always assign the highest emotional risk supported by the text.

Priority:

Label 3

↓

Label 2

↓

Label 1

↓

Label 0

---

# 8. Final Principle

Annotate what the message expresses.

Never infer information that is not explicitly or strongly implied.

The model predicts emotional risk,
not clinical diagnoses.
