<flow name="AILab:about" label="About FOAM-AILab" category="AILab" spid="aILab" accessLevel="PUBLIC_RO" description="What FOAM-AILab is and where the lessons are going." keywords="ailab,about,welcome"/>

# About FOAM-AILab

Learn how AI models are trained by building one end to end. FOAM-AILab is a
[FOAM](https://github.com/foam-foundation/foam3) application that you build
step by step, one lesson at a time. By the end of the series it:

1. Generates an instruction dataset from FOAM source code using an LLM
2. Lets people review and clean that dataset
3. Measures a base model, fine-tunes it, and measures it again
4. Serves the fine-tuned model to tools such as Claude Code
5. Publishes the dataset and the model to Hugging Face

Along the way you learn the FOAM features the
[FOAM tutorial](https://github.com/foam-foundation/foam3/blob/development/doc/tutorials/foam-tutorial.md)
did not cover, such as credentials and vaults, jobs, rules and permissions,
each one at the point where the application needs it.

## Where you are

This is the app at the end of **lesson 0**: FOAM's AI model training features
are turned on, so admins see the **ML Lab** menu. Its screens are empty for
now; the next lessons fill them in.

The series has three parts:

- **Build the dataset** (lessons 0–5): credentials, crawling sources, jobs, generating samples, review and cleanup
- **Train and measure** (lessons 6–8): test the base model, fine-tune it, compare the results
- **Use and share** (lessons 9–12): call your model from Claude Code, publish it on Hugging Face, improve samples with retrieval

See the [lessons roadmap](https://github.com/foam-foundation/FOAM-AILab/blob/main/doc/lessons/README.md#roadmap)
for which lessons are out.

## About this page

This page is a **Markdown flow**: the file `src/com/foamdev/ailab/About.md`,
turned into a flow at build time because it starts with a `<flow>` tag. The
**About** menu opens it in presentation mode. In development mode, edit the
file and reload the browser to see the change, with no rebuild.

## Demo logins

This is a learning app, so its logins are public: `admin` / `ailab` and
`demo` / `demopassword`. A real deployment uses real passwords.
