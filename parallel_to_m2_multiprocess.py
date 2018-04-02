import argparse
import os
import spacy
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.moses import MosesDetokenizer
import scripts.align_text as align_text
import scripts.cat_rules as cat_rules
import scripts.toolbox as toolbox
from tqdm import tqdm
import sys
from joblib import Parallel, delayed

# Get base working directory.
basename = os.path.dirname(os.path.realpath(__file__))
print("Loading SpaCy...")
# Load Tokenizer and other resources
print("Note: disable unecessary pipelines: ner, textcats")
nlp = spacy.load("en_core_web_lg", disable=['ner', 'textcat'])
# Lancaster Stemmer
stemmer = LancasterStemmer()
# Moses Detokenizer
detokenizer = MosesDetokenizer()
# GB English word list (inc -ise and -ize)
gb_spell = toolbox.loadDictionary(basename+"/resources/en_GB-large.txt")
# Part of speech map file
tag_map = toolbox.loadTagMap(basename+"/resources/en-ptb_map")

def _generate_m2(orig_sent, cor_sent):
    ignore_count= 0
    out_m2_str = ''
    # Process each pre-aligned sentence pair.
    try:
        # Detokenize sents if they're pre-tokenized. Otherwise the result will be wrong.
        if args.is_tokenized_orig:
            orig_sent = detokenizer.detokenize(orig_sent.strip().split(), return_str=True)
        if args.is_tokenized_cor:
            cor_sent = detokenizer.detokenize(cor_sent.strip().split(), return_str=True)
        # Markup the parallel sentences with spacy (assume tokenized)
        proc_orig = toolbox.applySpacy(orig_sent.strip(), nlp)
        proc_cor = toolbox.applySpacy(cor_sent.strip(), nlp)
        # Write the original sentence to the output m2 file.
        out_m2_str += "S " + toolbox.formatProcSent(proc_orig, feature_delimiter=args.feature_delimiter) + "\n"
        out_m2_str += "T " + toolbox.formatProcSent(proc_cor, feature_delimiter=args.feature_delimiter) + "\n"
        # out_m2.write("S " + toolbox.formatProcSent(proc_orig, feature_delimiter=args.feature_delimiter) + "\n")
        # out_m2.write("T " + toolbox.formatProcSent(proc_cor, feature_delimiter=args.feature_delimiter) + "\n")
        # Identical sentences have no edits, so just write noop.
        if orig_sent.strip() == cor_sent.strip():
            out_m2_str += "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n"
            # out_m2.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
        # Otherwise, do extra processing.
        else:
            # Auto align the parallel sentences and extract the edits.
            auto_edits = align_text.getAutoAlignedEdits(proc_orig, proc_cor, nlp, args)
            # Loop through the edits.
            for auto_edit in auto_edits:
                # Give each edit an automatic error type.
                cat = cat_rules.autoTypeEdit(auto_edit, proc_orig, proc_cor, gb_spell, tag_map, nlp, stemmer)
                auto_edit[2] = cat
                # Write the edit to the output m2 file.
                out_m2_str += toolbox.formatEdit(auto_edit)+"\n"
                # out_m2.write(toolbox.formatEdit(auto_edit)+"\n")
        # Write a newline when there are no more edits.
        out_m2_str += "\n"
        # out_m2.write("\n")
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        ignore_count += 1
        print('\nIgnore example:')
        print('- Source: ', orig_sent)
        print('- Target: ', cor_sent)
        print()

    return out_m2_str, ignore_count

def main(args):  
    print("Processing files...")
    # Open the original and corrected text files.
    with open(args.orig) as orig, open(args.cor) as cor, open(args.out, "w") as out_m2, \
         Parallel(n_jobs=args.n_jobs, verbose=5) as parallel:
        # Process each pre-aligned sentence pair.
        results = parallel(delayed(_generate_m2)(orig_sent, cor_sent)
                           for orig_sent, cor_sent in tqdm(zip(orig, cor)))

        out_m2_strs, ignore_counts = zip(*results)
        out_m2_str = ''.join(out_m2_strs)
        ignore_count = sum(ignore_counts)
        out_m2.write(out_m2_str)
        
        print('Total number of ignored examples: {}\n'.format(ignore_count))

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
    parser.add_argument("-feature_delimiter", type=str, default="￨",
                        help='The delimiter for word features concatenation.')
    parser.add_argument("-is_tokenized_orig", help="True if original sentences are tokenized by space. Otherwise we will detokenized them.", action="store_true")
    parser.add_argument("-is_tokenized_cor", help="True if corrected sentences are tokenized by space. Otherwise we will detokenized them.", action="store_true")
    parser.add_argument('-n_jobs', help="The maximum number of concurrently running jobs", type=int, default=8)
    args = parser.parse_args()
    # Run the program.
    main(args)
