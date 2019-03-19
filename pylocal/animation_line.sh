#!/bin/bash
for img in *.png; do
    filename=${img%.*}
    echo $filename
    convert -trim "$filename.png" "$filename.png"
done

convert -delay 25 -loop 0 *.png animated.gif

echo **DONE**
