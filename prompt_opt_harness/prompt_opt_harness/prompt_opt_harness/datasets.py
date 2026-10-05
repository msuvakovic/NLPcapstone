from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .prompts import BASE_INSTRUCTION
from .tasks import SENTIMENT, BINARY_NLI, TaskSpec

BASE_NLI_INSTRUCTION = "Determine if the Hypothesis entails or contradicts the Premise. Answer with exactly one word: Entailment or Contradiction."

@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str  # "Positive" or "Negative" (or "Entailment"/"Contradiction")
    domain: str


@dataclass(frozen=True)
class DomainSplit:
    dev: List[Example]
    test: List[Example]


@dataclass
class Dataset:
    source_domain: str
    source: DomainSplit
    ood: Dict[str, List[Example]]
    base_instruction: str = BASE_INSTRUCTION
    task: TaskSpec = SENTIMENT

    def lookup(self) -> Dict[str, Tuple[str, str]]:
        # text -> (label, domain), used by MockBackend to grade answers
        table: Dict[str, Tuple[str, str]] = {}
        for ex in self.source.dev + self.source.test:
            table[ex.text] = (ex.label, ex.domain)
        for examples in self.ood.values():
            for ex in examples:
                table[ex.text] = (ex.label, ex.domain)
        return table


def _mk(prefix: str, domain: str, rows: List[Tuple[str, str]]) -> List[Example]:
    return [Example(id=f"{prefix}{i}", text=text, label=label, domain=domain) for i, (text, label) in enumerate(rows)]


def load_demo_dataset() -> Dataset:
    """Small hand-written toy dataset: movie reviews (source) vs amazon/tweets (OOD)."""
    source_domain = "movie_reviews"

    movie_dev = _mk("mdev", source_domain, [
        ("The acting was excellent and the plot was engaging from start to finish.", "Positive"),
        ("This film was a wonderful experience with brilliant direction.", "Positive"),
        ("The lead actress gave an amazing performance in this movie.", "Positive"),
        ("The plot was boring and the pacing dragged through most of the film.", "Negative"),
        ("A terrible movie with awful acting and a dull script.", "Negative"),
        ("The cinematography was mediocre and the story felt disappointing.", "Negative"),
    ])
    movie_test = _mk("mtest", source_domain, [
        ("Great acting and a fantastic script made this film enjoyable.", "Positive"),
        ("The director did a superb job crafting an exciting story.", "Positive"),
        ("An excellent movie with a wonderful cast.", "Positive"),
        ("The film was dull, and the acting felt forced and poor.", "Negative"),
        ("A disappointing plot ruined what could have been a good movie.", "Negative"),
        ("Terrible pacing and a boring script made this a bad film.", "Negative"),
    ])

    amazon = _mk("amz", "amazon", [
        ("The battery life is amazing and the build quality is excellent.", "Positive"),
        ("Great product, works perfectly and shipping was fast.", "Positive"),
        ("I love this laptop, it's fast and reliable.", "Positive"),
        ("Wonderful headphones with excellent sound quality.", "Positive"),
        ("The battery died within a day, terrible product.", "Negative"),
        ("Poor build quality and the screen cracked immediately.", "Negative"),
        ("Awful customer service and a defective item.", "Negative"),
        ("Disappointing performance for the price, would not recommend.", "Negative"),
    ])

    tweets = _mk("tw", "tweets", [
        ("omg this new phone is amazing!! best purchase ever", "Positive"),
        ("so happy with my new shoes, super comfortable", "Positive"),
        ("loving the weather today, feels great outside", "Positive"),
        ("this app update is wonderful, so much faster now", "Positive"),
        ("ugh this service is terrible, never coming back", "Negative"),
        ("worst customer support ever, so frustrating", "Negative"),
        ("my flight got delayed again, this airline is awful", "Negative"),
        ("so disappointed with this restaurant, food was bad", "Negative"),
    ])

    return Dataset(
        source_domain=source_domain,
        source=DomainSplit(dev=movie_dev, test=movie_test),
        ood={"amazon": amazon, "tweets": tweets},
    )

def load_nli_demo_dataset() -> Dataset:
    """Small toy dataset for MultiNLI-style testing: fiction (source) vs telephone/slate (OOD)."""
    source_domain = "fiction"

    fiction_dev = _mk("fdev", source_domain, [
        ("Premise: The hero walked quickly into the dark cave.\nHypothesis: The hero entered a cave.", "Entailment"),
        ("Premise: She smiled as she read the letter from her friend.\nHypothesis: She was happy to hear from her friend.", "Entailment"),
        ("Premise: The king ruled with an iron fist, leaving his people in fear.\nHypothesis: The king was loved by everyone.", "Contradiction"),
        ("Premise: The spaceship launched into orbit successfully.\nHypothesis: The spaceship exploded on the launchpad.", "Contradiction"),
    ])
    fiction_test = _mk("ftest", source_domain, [
        ("Premise: The wizard cast a spell that illuminated the dark room.\nHypothesis: The room became bright.", "Entailment"),
        ("Premise: He found a hidden treasure chest filled with gold coins.\nHypothesis: He discovered something valuable.", "Entailment"),
        ("Premise: The knight lost his sword during the fierce battle.\nHypothesis: The knight defeated his enemies with his sword.", "Contradiction"),
        ("Premise: The dragon slept peacefully in its lair.\nHypothesis: The dragon was attacking the village.", "Contradiction"),
    ])

    telephone = _mk("tel", "telephone", [
        ("Premise: Yeah I totally agree with you on that point.\nHypothesis: We share the same opinion.", "Entailment"),
        ("Premise: I'll try to call you back tomorrow morning if I can.\nHypothesis: I might call you tomorrow.", "Entailment"),
        ("Premise: No, I definitely didn't say that to him.\nHypothesis: I told him exactly that.", "Contradiction"),
        ("Premise: We are planning to visit next weekend.\nHypothesis: We are never going to visit.", "Contradiction"),
    ])

    slate = _mk("slt", "slate", [
        ("Premise: The new policy will likely reduce taxes for the middle class.\nHypothesis: Middle class people will pay less in taxes.", "Entailment"),
        ("Premise: The governor's speech was widely criticized by both parties.\nHypothesis: Both political parties found flaws in the speech.", "Entailment"),
        ("Premise: The company announced record-breaking profits for Q3.\nHypothesis: The company lost money in Q3.", "Contradiction"),
        ("Premise: The treaty was signed yesterday by all participating nations.\nHypothesis: No one signed the treaty.", "Contradiction"),
    ])

    return Dataset(
        source_domain=source_domain,
        source=DomainSplit(dev=fiction_dev, test=fiction_test),
        ood={"telephone": telephone, "slate": slate},
        base_instruction=BASE_NLI_INSTRUCTION,
        task=BINARY_NLI,
    )
