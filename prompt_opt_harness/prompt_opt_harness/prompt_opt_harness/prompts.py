BASE_INSTRUCTION = "Classify the sentiment of the following text as Positive or Negative."


def build_classification_prompt(instruction: str, text: str) -> str:
    header = instruction.strip()
    body = f"Text: {text}\nAnswer with exactly one word: Positive or Negative."
    return f"{header}\n\n{body}" if header else body


REASONING_BASE_INSTRUCTION = "Solve the math word problem. Show your work, then end with a line in the exact form: Answer: <number>"


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
