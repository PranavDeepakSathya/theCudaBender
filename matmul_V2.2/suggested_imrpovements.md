in GAU.NAURST or whatever his name (GUUCI GUY) is, in this code a lot of address computation is happening 
outside of loops whenever possible. 
we can also go towards that improvement
also faster parity based barriers seem produent, not sure though. 
The leftover things are
Address precomputation
is there any possibility of prefetching without warp spec being fast?)
am I literally on a shit rtx6000pro? I could run gau's code here
Deeper understanding of pipelining
Maybe async stores actually help with persistence?
fp32 C is too fat to async store though
one thing could be that the extra consumer warp shoots down block occupancy on SM which is why V2 is slow and maybe persistance is not good without async stores which is why V3 is slow. 
At least we got torch working, which means async stores and persistence can be possible on fp8fp8fp16. 
