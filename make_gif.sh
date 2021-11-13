#!/bin/bash

ffmpeg -r 10 -i sample_human/unified_query%02d.png -y -filter_complex "[0:v] fps=10,split [a][b];[a] palettegen [p];[b][p] paletteuse" unified_human.gif 
#ffmpeg -r 10 -i sample_bird/unified_query%02d.png -y -filter_complex "[0:v] fps=10,split [a][b];[a] palettegen [p];[b][p] paletteuse" unified_bird.gif 

