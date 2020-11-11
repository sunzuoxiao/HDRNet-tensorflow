# HDRNet tensorflow
code all used tensorflow 1.13 base on [hdrnet-wgangp](https://github.com/moyi7712/hdrnet-wgangp "hdrnet-wgangp")
# tensorflow bug
The OP tf.matmul(), when used this op  in GPU,you will get wrong result,bug in CPU will not. 
so, used with tf.device('/cpu:0'):, force this OP in CPU
