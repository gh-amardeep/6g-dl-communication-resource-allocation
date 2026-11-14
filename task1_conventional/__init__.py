"""Task 1: conventional (bit-level) end-to-end text link.

Chain: text -> Huffman -> conv. encoder (K=7, 133/171) -> puncturing -> Gray QAM
       -> AWGN / Rayleigh block fading -> ZF equaliser -> max-log LLR demapper
       -> depuncturing -> soft Viterbi -> Huffman decoder -> text.

Import submodules explicitly, e.g. ``from task1_conventional.huffman import HuffmanCodec``.
"""
__version__ = "1.0.0"
