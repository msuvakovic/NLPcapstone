BASE_INSTRUCTION = "Classify the sentiment of the following text as Positive or Negative."


def build_classification_prompt(instruction: str, text: str) -> str:
    header = instruction.strip()
    body = f"Text: {text}\nAnswer with exactly one word: Positive or Negative."
    return f"{header}\n\n{body}" if header else body


REASONING_BASE_INSTRUCTION = "Solve the math word problem. Show your work, then end with a line in the exact form: Answer: <number>"


REASONING_BARE_INSTRUCTION = "Solve the math word problem."


def build_reasoning_prompt(instruction: str, question: str) -> str:
    header = instruction.strip()
    body = f"Question: {question}\nEnd your response with a final line in the exact form: Answer: <number>"
    return f"{header}\n\n{body}" if header else body


META_TEMPLATE = """You are improving an instruction for a {task_name} task.
Here are previous instructions and their accuracy on a held-out dev set:
{history_block}
Write ONE new instruction that might score higher than all of the above.
Respond with only the new candidate instruction, nothing else."""


CROSSOVER_TEMPLATE = """You are evolving instructions for a {task_name} task \
using a genetic algorithm (crossover + mutation).
Parent A: {parent_a}
Parent A dev score: {score_a:.2f}
Parent B: {parent_b}
Parent B dev score: {score_b:.2f}

Combine the useful ideas from both parents into ONE new child instruction. You may also \
introduce a small new idea (mutation).
Respond with only the new candidate instruction, nothing else."""


MUTATION_TEMPLATE = """Slightly modify the following instruction for a {task_name} \
task to try to improve it:
Instruction: {parent}
Respond with only the new candidate instruction, nothing else."""


# ---- GEPA-lite, TextGrad-lite and Random-search (ideas from the team's branches) ----

REFLECTION_TEMPLATE = """You are improving an instruction for a {task_name} task.
Current instruction:
{instruction}

On the example below the model gave a wrong answer.
{failure}

In two or three sentences, say what the instruction failed to tell the model and what to add or change.
Respond with only that explanation."""


REFLECTIVE_MUTATION_TEMPLATE = """Rewrite the instruction below for a {task_name} task, using the feedback.
Current instruction: {parent}
Feedback: {reflection}
Keep whatever already works. Respond with only the new instruction, nothing else."""


TEXTGRAD_CRITIQUE_TEMPLATE = """You are critiquing an instruction for a {task_name} task.
Instruction:
{instruction}

The model answered these inputs incorrectly:
{failures_block}

Write a short critique: what do these mistakes have in common, and how should the instruction change?
Respond with only the critique."""


TEXTGRAD_UPDATE_TEMPLATE = """Apply the critique to improve an instruction for a {task_name} task.
Instruction: {instruction}
Critique: {gradient}
Respond with only the improved instruction, nothing else."""


# Independent rewrites of the starting instruction, with no score or failure
# feedback. The variant number keeps a temperature-0 model from returning the
# same rewrite every time.
RANDOM_SEARCH_TEMPLATES = [
    """Rewrite the following instruction for a {task_name} task in your own words. Keep the task the same.
Instruction: {instruction}
(Variant #{variant})
Respond with only the rewritten instruction, nothing else.""",
    """Write a more concise version of the following instruction for a {task_name} task. Keep the task the same.
Instruction: {instruction}
(Variant #{variant})
Respond with only the new instruction, nothing else.""",
    """Write a more detailed version of the following instruction for a {task_name} task, adding guidance that would help the model get the answer right. Keep the task the same.
Instruction: {instruction}
(Variant #{variant})
Respond with only the new instruction, nothing else.""",
]


# ---- more of the team's optimizers (PBT, Ensemble, SAPO): see optimizers/ ----

PARAPHRASE_TEMPLATE = """Rewrite the following instruction for a {task_name} task in different words, \
keeping exactly the same meaning and keeping any required answer format:
Instruction: {instruction}
Respond with only the rewritten instruction, nothing else."""
