# Chatbot for Question Answering on provided document

## Part 1 
Training of intent Classification Model
(DIET) with Entity Extraction

The first part will be about intent classification with Rasa Open Source

- Rasa NLU vs Rasa Core

Steps:
- install conda
- create a new conda environment (teachat)
- conda activate <teachat>
- install rasa (pip install rasa)
- start a new rasa project (rasa init)
- 3 files: nlu.yml, domani.yml, stories.yml
- only train nlu: rasa train nlu
- test it on shell: rasa shell nlu
- I'd like to have some \[pasta](food): entities
- we must add `- name: RegexEntityExtractor` in the pipeline in the `config.yml`
- to add a look up table (menu) we can use>

```yml
- lookup: food
  examples: |
    - pizza
    - pasta
    - salad
```




## Part 2
Question Answering with Hugging Face.
Let's interrogate a question answering model.
There are 2 kinds of question answering: extractive and abstractive (generated)
These models are only useful when the answer is part of the context.
We will start with extractive.
This mean there is a context, question and the answer is found within the context.
The evaluation can be either exact match or f1 (average of precision and recall)
SQuAD is a common training dataset.

For our restaurant case, we have a context about the restaurant and we want customers receive the relevant answers.

The context is 
```
The kitchen opens at 11.30am
The kitchen closes at 9pm
The washroom is upstairs.
The restaurant opens at 8am and closes at 10pm
The address is Younge Street 12, Toronto
The phone number is +1 123 456789
```

We can either run it locally (with the transformers library) or using the inference API (we need an API key from Hugging Face)

[Colab](https://drive.google.com/file/d/15qweHuE_mVeDe1JluCUSua45NisdSP3o/view?usp=sharing)

On Hugging Face there are different question answering models.
We will try an extractive question answering that takes tabular data as an input.
It uses relative position embeddings (depending on the position index of a cell).

In our case we will provide a menu from the restaurant ask ask questions related to it.

| Type |Name  | Allergies | Price |
| -- |-- | -- | -- |
| Starter |Omelette  | Lactose | $20 |
| Starter |Mushroom Soup  | None | $20 |
| Starter |Ceasar Salad  | None | $15 |
| Main |Tenderloin  | None | $30 |
| Main |Norwegian Salmon Steak  | None | $30 |
| Main |Beef Hamburger  | Gluten | $30 |
| Dessert |Chocolate Lava  | Gluten | $10 |
| Dessert |Cheese Cake  | Glueten, Lactose | $10 |
| Dessert |Tiramisu  | Lactose, Gluten | $10 |

Questions:
- How much does the Tiramisu cost?
- What's the price of the Tenderloin?
- Which is the cheapest starter?
- Can I take eat the omelette if I'm lactose intollerant?
- Which is the most expensive dish in the menu ?
- Does the mushroom soup have any allergy restriction?

[Colab](https://drive.google.com/file/d/15qweHuE_mVeDe1JluCUSua45NisdSP3o/view?usp=sharing)

## Part 3: Generative Question Answering

We will use LangChain to perform generative question answering.
We have a dataset of food recipes for our restaurant.
The reference code is in ```part3.ipynb``` and the dataset is ```recipes_nlg.csv```

They are taken from the [NLG Dataset Lite](https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite)

The original NLG Dataset contains more than 1M recipes, while in our case we are only considering part of the NLG Lite dataset, with a total of 6118 rows.

Each row contains for each recipe the following entries:
uid 
name 
description - description of the dish
link - link to recipe blog post
ner - food named entity extracted
ingredients - ingredients and quantities
steps - cooking instructions

We will keep using the teachat conda environment from Part 1.

We will use LangChain library together with FAISS vector store.
Vector stores are one way to perform the RAG task (Retrieval Augmented Generator).
RAG is made of 2 components, the retriever and the generator.

The retriever picks elements that are most similar to the query.

The individual elements could be chuncks of the larger text, split in equal sizes.
In our case thought the reference text is already split, because it is just a list of recipes, so we will use the individual recipes as elements for the retriever, more specifically we will only keep the description column.

We will ask for initially for recipes of a particular type, eg containing tomatoes.
The query will also instruct the generator to format the response in a markdown table, so that we can easily visualed it.

We will also try modifying the top_k, which is the number of retrieved documents and we will see that the number of recipes displayed will change accordingly.







