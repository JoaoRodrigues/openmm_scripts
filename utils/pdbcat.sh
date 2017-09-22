#!/bin/bash

# Uses pymol to generate a .png file from a .pdb and uses imgcat to display it in terminal
# Assumes imgcat and pymol are both installed
# See:
# * https://www.iterm2.com/documentation-images.html
# * https://github.com/eddieantonio/imgcat
#
# Usage example: 
#  pdbcat 1crn.pdb

tmpdir=`mktemp -d -t pdbcat`
pymol -c -Q -d "load $1; as cartoon; spectrum; show sticks, hetatm; show nonbonded; bg_color white; orient; set ray_opaque_background, 0; png $tmpdir/pdbimg.png, width=800, height=600, ray=1"

# Trim the empty margins if ImageMagick is installed
convert $tmpdir/pdbimg.png -trim +repage $tmpdir/pdbimg_trimmed.png || cp $tmpdir/pdbimg.png $tmpdir/pdbimg_trimmed.png

# Print image to terminal
imgcat $tmpdir/pdbimg_trimmed.png || echo "The program 'imgcat' is required for this to work"
rm -rf $tmpdir

