from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .prompts import BASE_INSTRUCTION, REASONING_BASE_INSTRUCTION


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str  # "Positive" or "Negative"
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


def load_demo_reasoning_dataset() -> Dataset:
    """Toy math word problems: grade-school arithmetic (source) vs SVAMP-style
    distractor numbers / harder multi-step problems (OOD). Labels are the gold
    number as a string."""
    source_domain = "gsm8k_style"

    source_dev = _mk("rdev", source_domain, [
        ("Sam has 3 apples. He buys 5 more apples. How many apples does he have now?", "8"),
        ("A store had 20 shirts. It sold 8 shirts. How many shirts are left?", "12"),
        ("There are 4 boxes with 6 pencils each. How many pencils are there in total?", "24"),
        ("Maria has 15 stickers. She gives 6 stickers to her friend. How many stickers does she have left?", "9"),
        ("A farmer has 7 cows and buys 5 more. How many cows does he have now?", "12"),
        ("A classroom has 5 rows with 4 desks in each row. How many desks are there?", "20"),
    ])
    source_test = _mk("rtest", source_domain, [
        ("Ben had 10 marbles. He found 4 more. How many marbles does he have now?", "14"),
        ("A bakery made 30 muffins and sold 18. How many muffins are left?", "12"),
        ("There are 3 shelves with 7 books each. How many books in total?", "21"),
        ("Emma has 22 candies. She eats 9 of them. How many candies does she have left?", "13"),
        ("A parking lot has 6 rows with 8 cars each. How many cars are there?", "48"),
        ("Jake has 9 toy cars and buys 6 more. How many toy cars does he have now?", "15"),
    ])

    svamp_style = _mk("svamp", "svamp_style", [
        ("Rahul has 8 toy cars. His friend has 5 toy cars and 2 toy trucks. How many toy cars does Rahul have?", "8"),
        ("A baker made 24 cupcakes using 3 trays. He sold 9 cupcakes. How many cupcakes are left?", "15"),
        ("Liam bought 6 packs of stickers with 4 stickers in each pack, plus 2 extra loose stickers he found. How many stickers came from the packs?", "24"),
        ("A library has 5 shelves of fiction books with 10 books each and 3 shelves of nonfiction books. How many fiction books does the library have?", "50"),
        ("Priya had 40 dollars and 3 coupons. She spent 15 dollars on a book. How much money does she have left?", "25"),
        ("A garden has 6 rows of tomato plants with 5 plants each and 2 rows of pepper plants. How many tomato plants are in the garden?", "30"),
        ("Tom collected 18 seashells over 2 days. He gave away 5 seashells to his sister. How many seashells does he have left?", "13"),
        ("A parking garage has 4 floors with 25 spots each and 1 floor reserved for staff. How many total public spots are there across the 4 floors?", "100"),
    ])

    multistep = _mk("multi", "multistep", [
        ("A school bought 6 boxes of pencils with 12 pencils in each box. They gave 30 pencils to students. How many pencils are left?", "42"),
        ("A factory produces 150 toys a day. After 4 days, 200 toys were shipped out. How many toys remain?", "400"),
        ("Anna buys 3 packs of 8 pens each. She gives 2 pens to each of her 5 friends. How many pens does she have left?", "14"),
        ("A theater has 15 rows with 20 seats each. If 3 rows are reserved and cannot be sold, how many seats are available to sell?", "240"),
        ("A farmer harvested 8 baskets of apples with 25 apples each. He sold 3 baskets and gave away 40 apples from what remained. How many apples does he have left?", "85"),
        ("A company had 500 units in stock. It received 3 shipments of 120 units each, then sold 350 units. How many units are left?", "510"),
        ("A classroom has 6 tables with 4 chairs each. Two tables are removed along with their chairs, and 5 extra chairs are added. How many chairs are there now?", "21"),
        ("A train has 10 cars with 60 seats each. At the first stop, 150 passengers board. At the second stop, 3 full cars' worth of passengers (60 each) board as well. How many passengers are on the train after the second stop?", "330"),
    ])

    return Dataset(
        source_domain=source_domain,
        source=DomainSplit(dev=source_dev, test=source_test),
        ood={"svamp_style": svamp_style, "multistep": multistep},
        base_instruction=REASONING_BASE_INSTRUCTION,
    )
