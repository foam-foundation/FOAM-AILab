# FOAM-AILab

Learn how AI models are trained by building one end to end. FOAM-AILab is a
[FOAM](https://github.com/foam-foundation/foam3) application that you build
step by step: it turns source code into a training dataset, fine-tunes a
language model on it, measures whether the model got better, and publishes
the result to Hugging Face.

> [!NOTE]
> **Lessons are being added gradually, about one per week.** The series is in
> progress, so not every lesson is available yet. Check the
> [roadmap](doc/lessons/README.md#roadmap) to see which lessons are out, and
> [follow along](#follow-along) to be notified when the next one is released.

New here? Start with the video
[Neural Nets Demystified](https://www.youtube.com/watch?v=5WQdctBUO6M).

## What you'll learn

- **How training works**: gradient descent, backpropagation, neural networks
  and language models, explained from basic algebra up
- **How to build a dataset**: crawl source code, generate instruction samples
  with an LLM, then review and clean them
- **How to fine-tune and evaluate**: measure a base model, train it with LoRA,
  and compare the results honestly
- **How to use and share a model**: call it from tools such as Claude Code and
  publish it, with its dataset, on Hugging Face
- **FOAM beyond the basics**: credentials and vaults, jobs, rules, permissions
  and authentication, each introduced when the application needs it

## How it's organized

The material has two parts, both in [`/doc`](doc/):

- **[Foundations](doc/README.md#foundations)**: the theory behind training,
  with a worked example for each idea. No prior calculus or programming needed.
- **[Lessons](doc/lessons/README.md)**: hands-on lessons that build this
  application. Each one ends with a running app and links to the theory it uses.

## Getting started

1. Complete the [FOAM tutorial](https://github.com/foam-foundation/foam3/blob/development/doc/tutorials/foam-tutorial.md)
   if you haven't already; the lessons build on it.
2. Fork this repository.
3. Follow the [lessons roadmap](doc/lessons/README.md), starting with lesson 0.
   It lists what you need, including hardware.

## Follow along

New lessons are released roughly once a week, most with a short video.

- **Get notified:** watch this repository (**Watch → Custom → Releases**).
  Each lesson is published as a release.
- **Watch the videos:** subscribe to the
  [FOAM Foundation YouTube channel](https://www.youtube.com/@FOAM-Foundation).
- **Follow updates:** on [LinkedIn](https://www.linkedin.com/in/vesna-urosev-greer-5a92b7302/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This repository uses two licenses:

- **The application and all code** are licensed under the [Apache License 2.0](LICENSE-CODE).
  You may use it freely in your own projects, including commercial ones.
- **The documentation** in [`/doc`](doc/) (text, images, diagrams, media) is licensed under
  [CC BY-NC-SA 4.0](LICENSE-CONTENT).
  Code snippets in the documentation are also available under the Apache License 2.0.
  [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Copyright © 2026 FOAM Foundation and contributors.