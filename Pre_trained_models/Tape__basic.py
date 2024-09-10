import os


def main():
    os.system(
        "tape-embed transformer ../Data/Miyata/90/90_neg.fasta ../Features/Pre-trained_features/90_neg_tape_bert bert-base "
        "--batch_size 16 --tokenizer iupac")

main()
