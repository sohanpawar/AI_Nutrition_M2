# Problem Statement

## Dietary Guidance RAG Chatbot

### Brief

Build the prototype of a chatbot that answers questions about food, nutrition, and food safety. Put a retrieval layer under the chatbot so it answers only from official public dietary guidance documents. Every claim carries a citation. When the guidance does not cover a question, the assistant says so.

In the final project this becomes the service that answers questions such as "is this a reasonable way to eat" and "how long can I keep this in the fridge".

The real guidance already exists. Health authorities publish long, careful documents on exactly this. There is no API, only written prose, and almost nobody reads it. That gap is what retrieval-augmented generation is for.

Nutrient numbers for individual foods are different data and do not belong here. Those come from a structured database in Milestone 3.

---

## Context

People ask everyday questions about how to eat and how to store food:

- Is this a reasonable way to eat?
- How should I balance meals across food groups?
- How long can I keep this in the fridge?
- What temperature should a refrigerator be?
- How should raw and cooked food be handled so it stays safe?

Authoritative answers already exist. National nutrition institutes, food safety regulators, and international health bodies publish public guidance as written prose: fact sheets, guides, and safety pages. That material is careful, specific, and often long. It is also hard to use in the moment. There is no clean API behind it, and almost nobody reads the original documents when they have a question at the kitchen counter.

A general-purpose language model can sound confident on these topics and still invent a recommendation, blend two authorities into one claim, or give medical advice it was never asked to give. For diet and food safety, an unsourced answer is not good enough. The person asking needs to see which document the claim came from, who published it, and when.

This milestone closes that gap with a retrieval layer. The assistant may only answer from a small, fixed corpus of official public guidance. Every claim is cited. Questions the corpus does not cover are refused in plain language. Questions that are out of scope by design — medical advice, calorie targets, weight targets — are declined in code and redirected to a qualified professional.

This is a prototype of the guidance service, not the nutrition-facts service. Per-food nutrient numbers (calories, macros, micronutrients for a named food) are structured data and belong in Milestone 3. They must not be scraped, embedded, or answered from this corpus.

---

## Problem

Build a prototype chatbot that:

1. Answers questions about food, nutrition, and food safety.
2. Grounds every answer in retrieved passages from official public dietary guidance.
3. Cites the document, publisher, year, and a link for every claim.
4. Keeps sources separate when more than one document speaks to the same question.
5. Refuses clearly when the corpus does not contain the answer, and refuses separately when the question is out of scope by design.

---

## What this milestone is for

The finished product needs a service that can say, from published guidance:

- whether a described way of eating lines up with official dietary advice, and
- how long a food can be kept, and under what handling or temperature conditions.

This milestone builds that service as a retrieval-augmented chatbot over written guidance only.

| In this milestone | Not in this milestone |
| --- | --- |
| Official written guidance on diet and food safety | Nutrient numbers for individual foods |
| Answers grounded in retrieved chunks | Free-form model knowledge as the source of a claim |
| Citations on every claim | A single blended "what the guidelines say" |
| Refusal when the corpus is silent | Medical advice, calorie targets, or weight targets |
| Metadata: publisher, year, source URL, retrieval date | Any source that already has a clean API |

---

## Corpus

The corpus is one prepared file of official guidance, already split into blocks. Do not crawl the live authority pages, and do not add any other source. Written prose only: nutrient databases, recipe APIs, and anything with a clean API behind it do not belong here.

- **Dataset:** https://github.com/saksham20189575/health-data/blob/main/blocks.jsonl
- **File to ingest:** https://raw.githubusercontent.com/saksham20189575/health-data/main/blocks.jsonl

Each line is one block of JSON:

| Field | Meaning |
| --- | --- |
| `document_id` | Which guidance document the block belongs to. This is the unit of citation and of filtered retrieval. |
| `ordinal` | Order of the block inside that document. |
| `type` | `heading`, `paragraph`, `list_item`, or `table`. |
| `text` | The prose, list item, or markdown table to retrieve. |
| `heading_path` | Section headings from the document root down to this block. |
| `page` | Page number, when the source is a PDF. |
| `level` | Heading level, on heading blocks. |
| `header`, `rows` | Present on table blocks, alongside the markdown in `text`. |

The file contains seven documents. Store publisher, year, source URL, and retrieval date with every document. Publisher and year are not columns in the file; attach them from the document identity below. The source URL for this prototype is the dataset URL above. The retrieval date is the day the file was ingested. Those fields travel with the document and with every chunk derived from it.

### 1. `who-healthy-diet`

- **Document:** WHO Healthy diet, Fact sheet N°394
- **Publisher:** World Health Organization (WHO)
- **Role:** Population-level dietary guidance. Use for questions about what a healthy diet includes: fruit and vegetables, fats, free sugars, salt, and advice for adults and for infants and young children.

### 2. `fao-who-healthy-diets`

- **Document:** What are healthy diets?
- **Publisher:** Food and Agriculture Organization of the United Nations (FAO) and the World Health Organization (WHO)
- **Role:** Joint statement of the principles of a healthy diet: adequate, balanced, moderate, and diverse, plus the annexed WHO recommendations on fat, carbohydrate, sodium, potassium, and non-sugar sweeteners.

### 3. `eatwell-guide`

- **Document:** The Eatwell Guide
- **Publisher:** UK Government (Public Health England)
- **Role:** National food-group guidance. Use for questions about who the guide is for, how a day's eating is divided across food groups, oils and spreads, and foods high in fat, salt, and sugars.

### 4. `cold-food-storage`

- **Document:** Cold Food Storage Chart
- **Publisher:** U.S. Department of Agriculture, Food Safety and Inspection Service (USDA FSIS)
- **Role:** Refrigerator and freezer holding times. Use for questions about how long a named food can be kept at 40°F (4°C) or below, or at 0°F (−18°C) or below. The chart is one table. Do not invent a hold time the row does not state.

### 5. `fsa-chill`

- **Document:** How to chill, freeze and defrost food safely
- **Publisher:** UK Food Standards Agency (FSA)
- **Role:** Chilling, freezing, defrosting, and how to stock a fridge.

### 6. `who-five-keys`

- **Document:** Five Keys to Safer Food
- **Publisher:** World Health Organization (WHO)
- **Role:** Food-safety guidance on keeping food clean, separating raw and cooked food, cooking thoroughly, keeping food at safe temperatures, and using safe water and raw materials.

### 7. `kitchen-companion`

- **Document:** Kitchen Companion
- **Publisher:** U.S. Department of Agriculture, Food Safety and Inspection Service (USDA FSIS)
- **Role:** Consumer guidance on foodborne illness, shopping, storage, the temperature danger zone, and safe handling in the kitchen.

These seven cover the two jobs of the final service:

- **How to eat:** `who-healthy-diet`, `fao-who-healthy-diets`, and `eatwell-guide`.
- **How to keep food safe:** `cold-food-storage`, `fsa-chill`, `who-five-keys`, and `kitchen-companion`.

Do not add an eighth document. Do not add nutrient databases, recipe APIs, or any source that already has a clean API.

---

## What to build

### 1. Corpus

Ingest `blocks.jsonl` as the only corpus. Group blocks by `document_id`. Persist, for every document:

- document name
- `document_id`
- publisher
- year
- source URL (the dataset URL)
- retrieval date

### 2. Chunking

Split each document into chunks that stay meaningful as citations.

Every chunk carries:

- document name
- publisher
- year
- section heading

The file is already split into blocks, and it is full of tables and numbered recommendations. Fixed-size chunking will cut those in half and produce citations that no longer say what the source said. Choose a chunking strategy that respects `heading_path`, `list_item` blocks, and `table` blocks, and say in the README what you chose and what it cost you (for example, larger chunks, harder retrieval, or a table that still had to be flattened). The cold-storage chart is one table: a citation must not split a row away from its food, its refrigerator time, and its freezer time.

### 3. Retrieval

Build a vector index over the chunks. It must support both of these:

- retrieval across all documents
- retrieval filtered to one named document

### 4. Answer layer

The assistant answers only from retrieved chunks. It does not fill gaps from parametric knowledge.

Every claim carries a citation showing:

- document name
- publisher
- year
- a link to the source

### 5. Cross-document questions

Some questions have two documents with something to say. Cooking oil is the example: a nutrition institute and a food safety regulator can both weigh in, and they are not saying the same kind of thing.

When that happens:

- answer per document
- give each document its own citation
- never blend two sources into one claim about what "the guidelines say"

If the Eatwell Guide and the WHO fact sheet both speak to fats, report each position separately. If the USDA cold-storage chart and the FSA chilling guidance both speak to refrigeration, report each position separately. Agreement is allowed to be visible. It is not allowed to be collapsed into a single unattributed rule.

### 6. Two kinds of refusal

Both refusals are required. They are different, and they must stay different in the product.

**Not in the corpus.** When the retrieved chunks do not hold the answer, the assistant says the guidance does not cover it and names what it searched. Silence from retrieval is not permission to guess.

**Out of scope by design.** No medical advice. No calorie targets. No weight targets. Nothing about what anyone should weigh. The assistant declines and points the person to a qualified professional. Enforce this in code, the same way scope was enforced last week, so a prompt cannot talk the model into answering.

---

## Example questions the prototype should handle

| Question | Expected behaviour |
| --- | --- |
| What does a healthy diet include? | Answer from `who-healthy-diet` and from `fao-who-healthy-diets` where that statement also speaks to the point. Include `eatwell-guide` only as its own section, if its chunks support a claim. Separate citations. |
| How does the Eatwell Guide divide food groups? | Answer from `eatwell-guide` only, cited. |
| How should I handle raw meat when cooking? | Answer from `who-five-keys`, and from `kitchen-companion` if that document also covers it, cited separately. |
| What temperature should my fridge be, and how long can leftovers stay there? | Answer from `cold-food-storage` and from `fsa-chill` where each document actually states the fact. Do not invent a hold time the chart row or the FSA section does not state. |
| Which oil should I use, and is it safe to reuse? | Split the answer. Nutrition guidance on fats and oils comes from `who-healthy-diet`, `fao-who-healthy-diets`, and/or `eatwell-guide`. Safety guidance comes from `who-five-keys`, `kitchen-companion`, and/or `fsa-chill`. Never merge those into one claim. |
| How many calories should I eat to lose 5 kg? | Out of scope. Decline in code and point to a qualified professional. |
| What is the vitamin C content of an orange? | Not this milestone. Nutrient numbers belong to the Milestone 3 database. If these documents do not state it, say the corpus does not cover it and name what was searched. |
| Is this rash from something I ate? | Out of scope. No medical advice. Decline and point to a qualified professional. |

---

## Constraints

- Answers come from retrieved chunks of the corpus, not from the model's memory of nutrition.
- Written prose only. Do not ingest a nutrient API, a food-composition database, or any source that already exposes a clean API.
- Citations are mandatory on every claim: document name, publisher, year, and link.
- Cross-document answers stay per document. Do not synthesise a consensus the sources did not write.
- The two refusals are implemented differently. "Not in the corpus" is a retrieval outcome the assistant reports. "Out of scope" is a coded policy the assistant cannot be prompted out of.
- Chunks keep document name, publisher, year, and section heading so a citation can be traced to a place in the source, not only to a file.
- The README records the chunking decision and its cost.

---

## Done when

The prototype is done when a person can ask a food, nutrition, or food-safety question and get one of three outcomes:

1. **A grounded answer.** Claims are supported by retrieved chunks, each cited with document name, publisher, year, and link. If two documents apply, the answer is split by document.
2. **A corpus refusal.** The assistant says this guidance does not cover the question and names what it searched.
3. **A scope refusal.** The assistant declines medical advice, calorie targets, and weight targets, and points the person to a qualified professional, with that rule enforced in code.

Retrieval works across the full corpus and can be limited to one named document. Every stored document carries publisher, year, source URL, and retrieval date. The README explains how the documents were chunked and what that choice cost.
