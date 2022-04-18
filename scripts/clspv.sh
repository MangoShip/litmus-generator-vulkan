#kernelList=("message-passing" "message-passing-barrier" "message-passing-storage-workgroup-barrier"
#            "message-passing-storage-workgroup" "message-passing-workgroup-barrier" "message-passing-workgroup")
kernelList=("store-buffer-rmw-barrier" "store-buffer-storage-workgroup-rmw-barrier" "store-buffer-workgroup-rmw-barrier")

for i in "${kernelList[@]}"
do
    /shared/clspv/build/bin/clspv --cl-std=CL2.0 --inline-entry-points $i.cl -o spv/$i.spv
    /usr/bin/spirv-opt --strip-reflect spv/$i.spv -o spv/$i.spv
done
