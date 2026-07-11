"""Self-contained planning package for the lite_standalone agent.

A single-game (no batch axis), Kaggle-ready port of the speed-first
flow-diff producer. Everything the agent needs lives in this package; it has
no dependencies beyond ``torch`` and the standard library.

Attribution
-----------
This package is a PyTorch port of the planning design published by
slawekbiel in the Kaggle notebook "The Producer V2":
    https://www.kaggle.com/code/slawekbiel/the-producer-v2
The flow-diff scoring idea, candidate/shortlist structure, and the
reinforcement-risk term originate from that work. All credit for the
original design goes to the author; porting mistakes are ours.
"""
