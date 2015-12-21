for i in `seq 0 1 9`; do 
    echo "frame $i"
    ./main "frame/frame000$i.jpg"
done
for i in `seq 10 1 70`; do 
    echo "frame $i"
    ./main "frame/frame00$i.jpg"
done
