# Emotion Risk Dataset V2

## Objective

Create a supervised dataset for training a DistilBERT model capable of estimating emotional risk from Spanish text.

The model will classify user messages into four risk levels.

---

# Labels

## 0 - Low

The text expresses positive emotions or a normal emotional state.

Examples:

- Estoy feliz.
- Hoy fue un gran día.
- Todo salió bien.
- Me siento tranquilo.

---

## 1 - Moderate

The text expresses stress, anxiety, sadness, frustration or emotional discomfort, but without hopelessness or suicidal thoughts.

Examples:

- Estoy muy estresado.
- Hoy me siento triste.
- Tengo mucha ansiedad.
- No puedo dormir.

---

## 2 - High

The text expresses hopelessness, intense suffering, grief or emotional collapse.

Examples:

- Perdí a toda mi familia.
- Ya nada tiene sentido.
- No encuentro una salida.
- Siento que mi vida se derrumba.

---

## 3 - Critical

The text expresses suicidal ideation, desire to die or self-harm.

Examples:

- No quiero seguir viviendo.
- Quiero terminar con todo.
- Hoy será mi último día.
- Ojalá no despertara nunca.

---

# Annotation Rules

1. Read the complete sentence.
2. Label according to the highest emotional risk.
3. Ignore spelling mistakes.
4. Ignore punctuation.
5. Do not infer information that is not written.
6. If the sentence explicitly mentions suicide or self-harm, assign label 3.
7. If unsure between two labels, choose the lower one and mark it for later review.