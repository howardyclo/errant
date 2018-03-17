import argparse
import os
import spacy
from nltk.stem.lancaster import LancasterStemmer
import scripts.align_text as align_text
import scripts.cat_rules as cat_rules
import scripts.toolbox as toolbox
from tqdm import tqdm

def main(args):
    # Get base working directory.
    basename = os.path.dirname(os.path.realpath(__file__))
    print("Loading resources...")
    # Load Tokenizer and other resources
    nlp = spacy.load("en_core_web_lg", disable=['ner'])
    # Lancaster Stemmer
    stemmer = LancasterStemmer()
    # GB English word list (inc -ise and -ize)
    gb_spell = toolbox.loadDictionary(basename+"/resources/en_GB-large.txt")
    # Part of speech map file
    tag_map = toolbox.loadTagMap(basename+"/resources/en-ptb_map")
    # Setup output m2 file
    out_m2 = open(args.out, "w")

    print("Processing files...")
    # Open the original and corrected text files.
    with open(args.orig) as orig, open(args.cor) as cor:
        # Process each pre-aligned sentence pair.
        for orig_sent, cor_sent in tqdm(zip(orig, cor)):
            # Write the original sentence to the output m2 file.
            out_m2.write("S "+orig_sent)
            out_m2.write("T "+cor_sent)
            # Identical sentences have no edits, so just write noop.
            if orig_sent.strip() == cor_sent.strip():
                out_m2.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
            # Otherwise, do extra processing.
            else:
                # Markup the parallel sentences with spacy (assume tokenized)
                proc_orig = toolbox.applySpacy(orig_sent.strip(), nlp)
                proc_cor = toolbox.applySpacy(cor_sent.strip(), nlp)
                # Auto align the parallel sentences and extract the edits.
                auto_edits = align_text.getAutoAlignedEdits(proc_orig, proc_cor, nlp, args)
                # Loop through the edits.
                for auto_edit in auto_edits:
                    # Give each edit an automatic error type.
                    cat = cat_rules.autoTypeEdit(auto_edit, proc_orig, proc_cor, gb_spell, tag_map, nlp, stemmer)
                    auto_edit[2] = cat
                    # Write the edit to the output m2 file.
                    out_m2.write(toolbox.formatEdit(auto_edit)+"\n")
            # Write a newline when there are no more edits.
            out_m2.write("\n")

if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser(description="Convert parallel original and corrected text files (1 sentence per line) into M2 format.\nThe default uses Damerau-Levenshtein and merging rules and assumes tokenized text.",
                                                            formatter_class=argparse.RawTextHelpFormatter,
                                                            usage="%(prog)s [-h] [options] -orig ORIG -cor COR -out OUT")
    parser.add_argument("-orig", help="The path to the original text file.", required=True)
    parser.add_argument("-cor", help="The path to the corrected text file.", required=True)
    parser.add_argument("-out",     help="The output filepath.", required=True)
    parser.add_argument("-lev",     help="Use standard Levenshtein to align sentences.", action="store_true")
    parser.add_argument("-merge", choices=["rules", "all-split", "all-merge", "all-equal"], default="rules",
                                            help="Choose a merging strategy for automatic alignment.\n"
                                                            "rules: Use a rule-based merging strategy (default)\n"
                                                            "all-split: Merge nothing; e.g. MSSDI -> M, S, S, D, I\n"
                                                            "all-merge: Merge adjacent non-matches; e.g. MSSDI -> M, SSDI\n"
                                                            "all-equal: Merge adjacent same-type non-matches; e.g. MSSDI -> M, SS, D, I")
    args = parser.parse_args()
    # Run the program.
    main(args)
