from .tasks import SENTIMENT, TaskSpec

BASE_INSTRUCTION = "Classify the sentiment of the following text as Positive or Negative."


def build_classification_prompt(instruction: str, text: str, task: TaskSpec = SENTIMENT) -> str:
    header = instruction.strip()
    body = f"Text: {text}\n{task.answer_format}"
    return f"{header}\n\n{body}" if header else body


META_TEMPLATE = """You are improving an instruction for a classification task.
Here are previous instructions and their accuracy on a held-out dev set:
{history_block}
Write ONE new instruction that might score higher than all of the above.
Respond with only the new candidate instruction, nothing else."""


CROSSOVER_TEMPLATE = """You are evolving instructions for a classification task \
using a genetic algorithm (crossover + mutation).
Parent A: {parent_a}
Parent A dev score: {score_a:.2f}
Parent B: {parent_b}
Parent B dev score: {score_b:.2f}

Combine the useful ideas from both parents into ONE new child instruction. You may also \
introduce a small new idea (mutation).
Respond with only the new candidate instruction, nothing else."""


MUTATION_TEMPLATE = """Slightly modify the following instruction for a classification \
task to try to improve it:
Instruction: {parent}
Respond with only the new candidate instruction, nothing else."""

REFLECTION_TEMPLATE = """You are an expert prompt engineer. You wrote an instruction for a classification task, but it failed on some examples.
Instruction: {instruction}

Failed example:
Text: {text}
True Label: {label}
Predicted Label: {prediction}

Why did the instruction fail? Keep your reflection brief (1-2 sentences)."""


REFLECTIVE_MUTATION_TEMPLATE = """You are evolving instructions for a classification task.
Parent Instruction: {parent}

Reflection on failures:
{reflection}

Using the reflection, write ONE new child instruction that improves upon the parent.
Respond with only the new candidate instruction, nothing else."""

PARAPHRASE_TEMPLATE = """Paraphrase the following instruction while keeping its core meaning intact. Do not change the underlying task.
Original Instruction: {instruction}
Respond with only the paraphrased instruction, nothing else."""


TEXTGRAD_CRITIQUE_TEMPLATE = """You are an expert prompt engineer. The following instruction failed on a batch of examples.
Instruction: {instruction}

Failures:
{failures_block}

Analyze the batch of failures to compute a "textual gradient": a concise summary of what is fundamentally wrong with the instruction and how it should shift its focus. Keep it under 3 sentences."""


TEXTGRAD_UPDATE_TEMPLATE = """You are an expert prompt engineer. You are updating an instruction using a textual gradient.
Current Instruction: {instruction}
Textual Gradient (Critique): {gradient}

Apply the gradient to fix the instruction. Respond with only the new instruction, nothing else."""

OPRO_META_TEMPLATES = [
    """You are improving an instruction for a classification task.
Here are previous instructions and their scores:
{history_block}
Write ONE new instruction that might score higher than all of the above.
Respond with only the new candidate instruction, nothing else.""",
    """You are improving an instruction for a classification task.
Here are previous instructions and their scores:
{history_block}
Write ONE new instruction that focuses on being concise and direct while maintaining accuracy.
Respond with only the new candidate instruction, nothing else.""",
    """You are improving an instruction for a classification task.
Here are previous instructions and their scores:
{history_block}
Write ONE new instruction that adds a specific reasoning step or focuses on nuances.
Respond with only the new candidate instruction, nothing else.""",
]
