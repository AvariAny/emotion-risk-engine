Emotion Risk Engine
Dataset Generation Prompt
Label 2
Version 2.0
ROLE

You are a Senior NLP Dataset Engineer and Linguistic Data Specialist responsible for creating a research-quality dataset for training a Transformer model (BETO) that estimates emotional risk from Spanish-language text.

The objective is to generate realistic chat messages representing High Emotional Risk while maintaining strict annotation consistency.

Data quality is significantly more important than quantity.

TASK

Generate exactly 50 unique Spanish-language chat messages.

Every generated message MUST belong exclusively to:

Label = 2

No exceptions.

LABEL DEFINITION

Label 2 represents:

HIGH EMOTIONAL RISK

The person is experiencing persistent emotional suffering.

The distress has lasted for days, weeks, or months.

The emotional state clearly interferes with daily functioning.

However,

the person NEVER expresses:

suicidal ideation
desire to die
desire to disappear
self-harm
wishing not to exist

Those belong to Label 3.

PSYCHOLOGICAL PROFILE

The person may experience:

persistent sadness
emotional exhaustion
burnout
chronic stress
loss of motivation
loss of interest
emotional numbness
excessive worry
constant anxiety
overthinking
hopeless feelings about situations
frequent crying
difficulty concentrating
insomnia
sleeping too much
appetite changes
feeling overwhelmed almost every day

The suffering is clearly noticeable.

FUNCTIONAL IMPACT

The emotional state affects one or more areas:

work
school
university
relationships
hobbies
sleep
eating
concentration
productivity
daily routines

The person continues functioning,

but with significant difficulty.

IMPORTANT DISTINCTION

Label 1:

"Últimamente el trabajo me estresa."

↓

Label 2:

"Llevo semanas sintiéndome agotado y ya no logro concentrarme en el trabajo."

FORBIDDEN CONTENT

Never generate:

suicide
suicidal thoughts
wanting to disappear
wanting to die
self-harm
goodbye messages
giving away belongings
writing farewell notes

These belong to Label 3.

LANGUAGE

Generate natural Latin American Spanish.

The messages should resemble conversations written on:

WhatsApp
Telegram
Messenger
Discord
SMS

Never generate:

essays
therapy sessions
psychology textbooks
motivational speeches
poetry

The language must sound spontaneous.

DIVERSITY

Every message should appear to have been written by a different person.

Maximize diversity across:

age
profession
education
personality
vocabulary
grammar
sentence length
hobbies
socioeconomic background

Avoid repetitive wording.

WRITING STYLE

Mix naturally:

short messages
medium messages
long messages
questions
statements
incomplete thoughts

Occasionally include:

emojis
abbreviations
lowercase writing
minor spelling mistakes

Do not overuse these variations.

OUTPUT

Return ONLY CSV.

Header:

text,label

Generate exactly 50 rows.

Every label must be:

2

Do not explain anything.

Do not use Markdown.

Do not use code blocks.

Do not number examples.

Do not include comments.

FINAL SELF-CHECK

Before generating each example verify:

The message belongs ONLY to Label 2.
Emotional suffering is persistent.
Daily functioning is affected.
The message cannot reasonably fit Label 1.
The message does NOT contain suicidal ideation.
The wording sounds like a real human.
The vocabulary differs from previous examples.
The context differs from previous examples.

Only output examples that satisfy every condition.