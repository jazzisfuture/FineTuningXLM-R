pair=zh-en
PARA_PATH=/home/yljiang/code/finetuningXlmr/data
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    echo "NLINES: $NLINES"
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    echo $NTRAIN $NVAL
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
}
split_data $PARA_PATH/$pair.spm.all $PARA_PATH/$pair.train $PARA_PATH/$pair.valid $PARA_PATH/$pair.test