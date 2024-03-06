from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
	words: List[str]
	ori_words: List[str]
	dep_heads: List[int]
	dep_labels: List[str]
	span_labels: List[set] = None
	labels: List[str] = None
	prediction: List[str]  = None

