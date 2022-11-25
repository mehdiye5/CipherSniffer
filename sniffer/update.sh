#echo "line1" >> test.txt
#echo "line2" >> test.txt
#echo "line3" >> test.txt
#sed '$d' test.txt
#cp test.txt test.txt.tmp
#sed '$d' test.txt.tmp > test.txt
#rm -f test.txt.tmp


ma=7
d=2
wk=("Mon" "Tue" "Wed" "Thu" "Fri" "Sat" "Sun")
mon=("Jan" "Feb" "Mar" "Apr" "May" "Jun" "July" "Aug" "Sep" "Oct" "Nov" "Dec")
day=("31" "28" "31" "30" "31" "30" "31" "31" "30" "31" "30" "31")
mn=("bert.py" "ciphers.py" "models.py" "tokenizers.py" "training.py" "token.py")
al=("ber.py" "cipher.py" "model.py" "tokenizer.py" "trainin.py" "toke.py")


comm=(0 4 7 12 18)

for k in `seq 11 11`
do
    #curr=$((day[k]))
    #echo "$curr"
    for j in `seq 1 $((day[k]))`
    do  
         
        ct=1
        idmax=$(( $RANDOM % 6 + 1 ))
        hr=$(( $RANDOM % 59 + 1 ))
        max=${comm[idmax]}

        for i in `seq 0 $max`
        do
            if (( $i % 2 == 0))
            then
                #echo "$i"
                #cp training.py training.py.tmp
                #sed '$d' training.py.tmp > training.py
                #rm -f training.py.tmp
                a=1
            else
                #echo "line1" >> ./training.py
                a=1
            fi

            idx=$(( $RANDOM % 6 + 1 ))
            cp ${mn[idx]} ${al[idx]}
            head -$((ct*5)) ${al[idx]}>>${mn[idx]}
            ct=$((ct+1))
            rm -f ${al[idx]}        

            git add "./${mn[idx]}"
            git commit --amend --no-edit --date="${wk[d]} ${mon[k]} $j 11:$hr:$i 2022 -0600"
            git push -f

            d=$((d+1))
            if [ $d -gt 7 ]
            then
                d=1
            fi

        done
    done
done