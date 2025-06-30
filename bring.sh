# define arrays of branch names and target folder names
branches=(diff_dim mpl_seq static_mask static_after_mlp)
prefixes=(MISA+MMLATCH_seq_after_extr \
          MISA+MMLATCH_seq_after_mlp \
          MISA+MMLATCH_static_extr \
          MISA+MMLATCH_static_after_mlp)

# loop over them
for i in "${!branches[@]}"; do
  git archive misa/${branches[$i]} \
    --prefix=${prefixes[$i]}/ | tar -x
done

git add .
git commit -m "Import snapshots of diff_dim, mpl_seq, static_mask, static_after_mlp into folders"

