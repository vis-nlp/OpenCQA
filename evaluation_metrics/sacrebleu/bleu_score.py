from sacrebleu.metrics import BLEU, CHRF, TER
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
import pandas as pd

# Read data
with open(f"./outputs/generated-p80.txt") as f:
    model_output_summary = f.read().strip().split("\n")

with open(f"./outputs/testOriginalSummary.txt") as f:
    test_summary = f.read().strip().split("\n")

# Detokenize
mpn = MosesPunctNormalizer()
mt = MosesTokenizer(lang="en")
md = MosesDetokenizer(lang="en")

def detokenize(sent):
    sent = mpn.normalize(sent)
    tokens = mt.tokenize(sent)
    return md.detokenize(tokens)

model_output_summary = list(map(detokenize, model_output_summary))
test_summary = list(map(detokenize, test_summary))

# Calculate BLEU score
bleu = BLEU()
print(bleu.corpus_score(model_output_summary, [test_summary]))
