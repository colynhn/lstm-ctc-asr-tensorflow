#!/usr/bin/env bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

H=`pwd`  #exp home
n=8      #parallel jobs

#corpus and trans directory
thchs=$1/thchs30

#you can obtain the database by uncommting the following lines
[ -d $thchs ] || mkdir -p $thchs  || exit 1
echo "downloading THCHS30 at $thchs ..."
./download_and_untar.sh $thchs  http://www.openslr.org/resources/18 data_thchs30  || exit 1
./download_and_untar.sh $thchs  http://www.openslr.org/resources/18 resource      || exit 1
./download_and_untar.sh $thchs  http://www.openslr.org/resources/18 test-noise    || exit 1
echo "download done."
