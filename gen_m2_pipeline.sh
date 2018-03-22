orig=$1
cor=$2
out_filename_prefix=$3

python parallel_to_m2.py \
-orig $orig \
-cor $cor \
-out $out_filename_prefix.rules.damerau_lev.m2 \
-is_tokenized

python parallel_to_m2.py \
-orig $orig \
-cor $cor \
-out $out_filename_prefix.rules.standard_lev.m2 \
-is_tokenized \
-lev

python parallel_to_m2.py \
-orig $orig \
-cor $cor \
-out $out_filename_prefix.all_split.damerau_lev.m2 \
-is_tokenized \
-merge all-split

python parallel_to_m2.py \
-orig $orig \
-cor $cor \
-out $out_filename_prefix.all_split.standard_lev.m2 \
-is_tokenized \
-merge all-split \
-lev