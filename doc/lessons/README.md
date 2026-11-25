# FOAM-AILab Lessons

In this series you build FOAM-AILab step by step: a FOAM application that
turns source code into a training dataset, fine-tunes a language model on it,
measures whether it got better, and shares the result.

> [!NOTE]
> **Lessons are being added gradually, about one per week.** The series is in
> progress, so not every lesson is available yet. Check the
> [roadmap](#roadmap) to see which lessons are out, and
> [follow along](../../README.md#follow-along) to be notified when the next one is released.

Along the way you learn the FOAM features the [FOAM tutorial][foam-tutorial]
did not cover, such as credentials and vaults, jobs, rules and permissions,
each one at the point where the application needs it.

## What you'll build

By the end of the series, your application:

1. Generates an instruction dataset from FOAM source code using an LLM
2. Lets people review and clean that dataset
3. Measures a base model, fine-tunes it, and measures it again
4. Serves your fine-tuned model to tools such as Claude Code
5. Publishes the dataset and the fine-tuned model to Hugging Face

## Before you start

- **The FOAM tutorial.** Complete the [FOAM tutorial][foam-tutorial] first.
  These lessons build on its models, DAOs, journals, views and services, and
  don't explain them again.
- **The theory.** The [foundations guides](../foundations/) explain how models
  learn. Each lesson links to the sections it relies on, so you can read them
  as you go.
- **A Hugging Face account** (free). You'll need it from lesson 1.

### Hardware

Training uses [MLX](https://github.com/ml-explore/mlx), which runs on Apple
Silicon Macs.

| Your machine | What you can do |
|---|---|
| Apple Silicon Mac, 16 GB or more | The whole series with the default model (Qwen2.5-Coder-1.5B) |
| Apple Silicon Mac, 8 GB | The whole series with a smaller model (Qwen2.5-Coder-0.5B) |
| Apple Silicon Mac, 32 GB or more | Optionally, the larger Qwen2.5-Coder-7B |
| Windows or Linux | Lessons 0–5 and 11; a training path for other platforms is planned |

### Generating samples: free or paid

The dataset is written by an LLM. You can choose:

- **Ollama**: free, runs on your own machine. Good for learning the pipeline.
- **A paid API** (such as Claude or OpenAI): better samples, faster. Recommended
  for the final dataset. Each lesson that uses it states the expected cost.

Every lesson works with Ollama.

## How the lessons work

1. **Fork this repository** and work in your fork.
2. **Start from a lesson tag.** The end state of every lesson is tagged
   `lesson-00`, `lesson-01`, and so on. To start lesson 3 from a clean state,
   check out `lesson-02`.
3. **Compare when stuck.** `git diff lesson-02 lesson-03` shows everything
   lesson 3 changes.

Each lesson has the same parts: the goal, what you need before you start,
links to the concepts it uses, the steps, a checkpoint to confirm everything
works, troubleshooting, and a link to the next lesson. If a tag is ever moved
to fix a lesson, it's recorded in the changelog.

## Roadmap

The lessons follow the pipeline, from source code to a published model.
New lessons are released roughly once a week, most with a short video. To get
notified, watch this repository (**Watch → Custom → Releases**): each lesson is
published as a release. See [Follow along](../../README.md#follow-along) for
other ways to keep up.

### Part 1: Build the dataset

| # | Lesson | What you build | FOAM features | Status |
|---|---|---|---|---|
| 0 | Setup and the plan | Turn on FOAM's AI Model training features; see where the series is going and why | Build flags, deployments | Coming soon |
| 1 | Credentials and the vault | Store your Hugging Face token (and API key, if you use one) safely | Credentials, vaults | Coming soon |
| 2 | Crawl the sources | Find every file the dataset will be built from | Crawlers, Java classes with a client companion | Coming soon |
| 3 | Make it a job | Run the crawl in the background, with pause, resume and checkpoints | Jobs, scripts | Coming soon |
| 4 | Generate samples | Turn source code into training samples with an LLM | DAOs and journals, put to work | Coming soon |
| 5 | Review and clean | Review samples, remove duplicates, filter out bad ones | Rules, groups and permissions | Coming soon |

### Part 2: Train and measure

| # | Lesson | What you build | FOAM features | Status |
|---|---|---|---|---|
| 6 | Measure first | Test the base model before training it | Running Python from FOAM | Coming soon |
| 7 | Fine-tune | Train the model on your dataset and watch the loss | Long-running jobs | Coming soon |
| 8 | Evaluate and iterate | Compare results, choose the best checkpoint, fix what got worse | Queries and reports over DAOs | Coming soon |

### Part 3: Use and share

| # | Lesson | What you build | FOAM features | Status |
|---|---|---|---|---|
| 9 | Use your model | Call your model from Claude Code | Authentication for external clients | Coming soon |
| 10 | Share it on Hugging Face | Publish your dataset and fine-tuned model, with cards | Publishing as a job | Coming soon |
| 11 | Better samples with retrieval | Improve samples by adding relevant documentation to the LLM's input | Vector search | Coming soon |
| 12 | Optional: automate | Regenerate and re-evaluate on a schedule | Scheduled jobs | Coming soon |

## Learn more

- [Foundations](../foundations/): the theory behind training
- [FOAM tutorial][foam-tutorial]: FOAM basics
- [FOAM guides](https://github.com/foam-foundation/foam3/tree/development/doc/guides):
  reference for every FOAM feature used in the lessons

[foam-tutorial]: https://github.com/foam-foundation/foam3/blob/development/doc/tutorials/foam-tutorial.md
